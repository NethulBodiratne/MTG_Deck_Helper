# --- TRAINING BLOCK ---
import json
import unicodedata
import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import sympy
import os
import re
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
ORACLE_DATA_PATH = 'oracle-cards.json'
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
API_BASE = "https://json.edhrec.com/pages"
EMBEDDINGS_CACHE_PATH = 'card_embeddings.pt'

# Device Selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# --- FEATURE EXTRACTION ---
# Define the keywords and types we want to track mechanically
KEYWORDS_LIST = [
    # Evergreen & Core Actions
    'flying', 'trample', 'ward', 'haste', 'deathtouch', 'lifelink', 'vigilance', 'menace', 'reach',
    'double strike', 'first strike', 'indestructible', 'hexproof', 'flash', 'defender', 'prowess',
    'scry', 'surveil', 'exile', 'mill', 'discard', 'counter', 'sacrifice', 'token', 'tap', 'untap',
    # High-Synergy Actions & Ability Words
    'proliferate', 'investigate', 'landfall', 'convoke', 'delve', 'cycling', 'flashback', 'kicker',
    'cascade', 'discover', 'amass', 'poisonous', 'infect', 'toxic', 'corrupted', 'dredge', 'storm',
    # Modern & Future Era (Bloomburrow through Edge of Eternities 2026)
    'offspring', 'gift', 'forage', 'expend', 'manifest dread', 'survival', 'impending', 'warp',
    'vivid', 'plot', 'saddle', 'disguise', 'cloak', 'collect evidence', 'suspect', 'lander'
]
TYPES_LIST = ['creature', 'enchantment', 'artifact', 'sorcery', 'instant', 'planeswalker', 'land', 'battle',
              'kindred', 'legendary', 'basic', 'snow', 'world', 'saga', 'vehicle', 'equipment', 'aura', 'room', 'case']

def get_feature_vector(card_data_raw):
    """
    Converts card metadata into a high-fidelity mechanical feature vector.
    Captures Types, Keywords, and Mana Production/Fixing.
    """
    # 1. Mana Value Normalized (0.0 to 1.0)
    cmc = min(card_data_raw.get('cmc', 0.0), 15.0) / 15.0
    # 2. Type Bitmask
    type_line = card_data_raw.get('type_line', '').lower()
    type_vec = [1.0 if t in type_line else 0.0 for t in TYPES_LIST]
    # 3. Text Preparation (Handles double-faced cards)
    keywords_array = [k.lower() for k in card_data_raw.get('keywords', [])]
    if 'card_faces' in card_data_raw:
        oracle_text = " ".join([f.get('oracle_text', '').lower() for f in card_data_raw['card_faces']])
    else: oracle_text = card_data_raw.get('oracle_text', '').lower()
    # 4. Keyword Vector (with Regex Word Boundaries)
    key_vec = []
    for k in KEYWORDS_LIST:
        if k in keywords_array or re.search(rf'\b{re.escape(k)}\b', oracle_text):
            key_vec.append(1.0)
        else: key_vec.append(0.0)
    # 5. Mana Production Vector
    mana_vec = []
    # Generic Ramp/Fixing Detection
    mana_vec.append(1.0 if "put" in oracle_text and "land" in oracle_text and "battlefield" in oracle_text else 0.0) # manaramp
    mana_vec.append(1.0 if "add" in oracle_text and ("mana" in oracle_text or "{" in oracle_text) else 0.0) # manafix
    mana_vec.append(1.0 if "mana of any color" in oracle_text else 0.0) # mana_any
    # Specific Color Production
    for color in ['w', 'u', 'b', 'r', 'g', 'c']:
        symbol = f"{{{color.upper()}}}"
        mana_vec.append(1.0 if symbol in oracle_text and "add" in oracle_text else 0.0)
    # Final combined vector: CMC (1) + Types (19) + Keywords (52) + Mana (9) = 81-dimensional
    full_vector = [cmc] + type_vec + key_vec + mana_vec
    return torch.tensor(full_vector, dtype=torch.float32)

# --- 1. MODEL DEFINITION ---
class SynergyRefiner(nn.Module):
    def __init__(self, input_dim):
        super(SynergyRefiner, self).__init__()
        # Deeper architecture to handle augmented features
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        # self.dropout = nn.Dropout(0.1) # Prevent overfitting to specific card names

    def forward(self, x):
        # Step 1: Normalize the raw input features
        x = self.input_norm(x)
        identity = x
        # Step 2: Pass through layers with LeakyRelu to prevent "dead neurons"
        out = F.leaky_relu(self.norm1(self.fc1(x)))
        out = F.leaky_relu(self.norm2(self.fc2(out)))
        out = self.norm3(self.fc3(out))
        # Step 3: Residual Connection (Crucial for lowering loss fast)
        # This allows the model to learn ONLY the "synergy refinement"
        out = out + identity
        return F.normalize(out, p=2, dim=1) # Keeps embeddings on a hypersphere

class MultiSimilarityLoss(nn.Module):
    """
    Implements Multi-Similarity Loss for deep metric learning.
    Captures pair-wise similarities and weights them based on relative hardness.
    """
    def __init__(self, alpha=20.0, beta=90.0, margin=1.0):
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.base_margin = margin

    def forward(self, anchors, positives, negatives, current_margin=None):
        m = current_margin if current_margin is not None else self.base_margin
        # Sim(A,P) and Sim(A,N)
        sim_pos = torch.sum(anchors * positives, dim=1)
        sim_neg = torch.sum(anchors * negatives, dim=1)
        # MS-Loss weighting
        pos_exp = torch.exp(-self.alpha * (sim_pos - m))
        neg_exp = torch.exp(self.beta * (sim_neg - m))
        loss = torch.log(1 + torch.sum(pos_exp)) / self.alpha + \
               torch.log(1 + torch.sum(neg_exp)) / self.beta
        return loss

# Model Initialization
print("Initializing SentenceTransformer...")
st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
if device == 'cuda':
    # st_model.half()
    print("⚡ GPU detected: FP32 Precision Maintained for Stability")
elif device == 'cpu':
    # Use Dynamic Quantization for CPU (INT8) - 2-3x faster on standard Colab CPUs
    st_model = torch.quantization.quantize_dynamic(st_model, {torch.nn.Linear}, dtype=torch.qint8)
    # Set max sequence length to save cycles on short card text
    st_model.max_seq_length = 128
    print("🧊 CPU detected: INT8 Dynamic Quantization enabled")

# Calculate dimension (Text Dim + Feature Dim)
text_dim = st_model.get_sentence_embedding_dimension()
feature_dim = 1 + len(TYPES_LIST) + len(KEYWORDS_LIST)
total_dim = text_dim + feature_dim
print(f"📊 Vector Architecture: {text_dim} (Text) + {feature_dim} (Mechanical) = {total_dim} Total Dim")
criterion = MultiSimilarityLoss().to(device)
scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

# Load Scryfall
print("Loading Scryfall data...")
card_db = {}
card_features = {} # Stores the bitmask/stats
try:
    with open(ORACLE_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        # If raw_data is a dict, check for a 'data' key (some Scryfall API responses)
        # If it's a list, use it directly.
        if isinstance(raw_data, dict):
            cards_list = raw_data.get('data', [])
        elif isinstance(raw_data, list):
            cards_list = raw_data
        else:
            print(f"Unexpected data format: {type(raw_data)}")
            exit()
        for c in cards_list:
            # Add a safety check: skip if the entry isn't a dictionary
            if not isinstance(c, dict):
                continue
            name = c.get('name', 'Unknown')
            if not name: continue
            # Skip cards that are not legal in the Commander format
            legalities = c.get('legalities', {})
            if legalities.get('commander') != 'legal': continue
            if c.get('border_color') == 'silver' or c.get('security_stamp') == 'acorn': continue
            # 1. Handle Colors & Color Identity (Crucial for Commander)
            colors_list = c.get('colors', [])
            if not colors_list and 'card_faces' in c:
                colors_list = []
                for face in c['card_faces']: colors_list.extend(face.get('colors', []))
            colors = "".join(sorted(list(set(colors_list)))) or 'C'
            # Added: Color Identity (Symbols in text box + colors)
            identity = "".join(sorted(c.get('color_identity', []))) or 'C'
            cmc = c.get('cmc', 0.0)

            # 2. Extract Gameplay Stats Helper
            def get_face_info(face_data):
                f_type = face_data.get('type_line', 'Unknown Type')
                f_text = face_data.get('oracle_text', '')
                f_mana = face_data.get('mana_cost', '')
                p, t = face_data.get('power', ''), face_data.get('toughness', '')
                loyalty = face_data.get('loyalty', '')
                defense = face_data.get('defense', '')
                stats = ""
                if p and t: stats = f" [{p}/{t}]"
                elif loyalty: stats = f" [Loyalty: {loyalty}]"
                elif defense: stats = f" [Defense: {defense}]"
                return f"{f_mana} | {f_type}{stats} | {f_text}"

            # 3. Process Card Faces
            if 'card_faces' in c and c.get('layout') not in ['flip', 'meld']:
                gameplay_info = " // ".join([get_face_info(f) for f in c['card_faces']])
            else: gameplay_info = get_face_info(c)

            # 4. Supplemental Gameplay Data (Produced Mana & Keywords)
            keywords = ", ".join(c.get('keywords', []))
            keyword_str = f"Keywords: {keywords} | " if keywords else ""
            produced = ", ".join(c.get('produced_mana', []))
            produced_str = f"Produces: {produced} | " if produced else ""
            has_parts = "Has Related Tokens/Parts | " if 'all_parts' in c else ""

            # 5. Final Assembly (Augmented with Identity and Production)
            context_str = (
                f"Colors: {colors} | Identity: {identity} | CMC: {cmc} | "
                f"{keyword_str}{produced_str}{has_parts}{gameplay_info}"
            )
            lowered_name = name.lower()
            card_db[lowered_name] = context_str
            card_features[lowered_name] = get_feature_vector(c)

    if not card_db:
        print("⚠️ Data Warning: No cards parsed from JSON.")
    else:
        print(f"✅ Library Loaded: {len(card_db)} unique cards mapped.")
except json.JSONDecodeError as e:
    print(f"❌ JSON Error: {e}")
    print("Tip: Check if the file downloaded completely. Scryfall files must be full arrays [ ... ].")
except FileNotFoundError:
    print(f"FATAL: {ORACLE_DATA_PATH} not found. Please download the Scryfall Oracle bulk data.")
    exit()

# --- PRE-COMPUTE EMBEDDINGS ---
if os.path.exists(EMBEDDINGS_CACHE_PATH):
    print("📂 Loading cached embeddings from disk...")
    precomputed_embeddings = torch.load(EMBEDDINGS_CACHE_PATH, map_location=device)
    all_names = list(precomputed_embeddings.keys())
    print(f"✅ Loaded {len(precomputed_embeddings)} card vectors.")
else:
    print(f"Pre-computing augmented embeddings for {len(card_db)} cards...")
    all_names = list(card_db.keys())
    all_texts = [card_db[name] for name in all_names]
    encoded_text = st_model.encode(all_texts, batch_size=128, convert_to_tensor=True, show_progress_bar=True, device=device)
    precomputed_embeddings = {}
    for i, name in enumerate(all_names):
        feat_vec = card_features[name].to(device)
        # NORMALIZATION STEP: Scale text and features to have the same "energy" (L2 Norm = 1)
        text_norm = F.normalize(encoded_text[i].unsqueeze(0), p=2, dim=1)
        feat_norm = F.normalize(feat_vec.unsqueeze(0), p=2, dim=1)
        # Concatenate normalized vectors
        precomputed_embeddings[name] = torch.cat([text_norm.flatten(), feat_norm.flatten()])
    torch.save(precomputed_embeddings, EMBEDDINGS_CACHE_PATH)
    print(f"✅ Pre-computation complete and saved to {EMBEDDINGS_CACHE_PATH}.")
all_names = list(precomputed_embeddings.keys())

# --- 2. UTILITY FUNCTIONS ---
session = requests.Session()
session.headers.update({'User-Agent': USER_AGENT})

def sanitize_slug(name):
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii').lower()
    name = name.replace("//", "-").replace("&", "-")
    name = re.sub(r'[^a-z0-9\s-]', '', name)
    return re.sub(r'[\s-]+', '-', name).strip('-')

def fetch_commander_training_data(name):
    avg = get_average_deck(name)
    as_cmdr = get_as_commander_data(name)
    # Combine and remove duplicates by card name
    combined = {c['name']: c for c in (avg + as_cmdr)}.values()
    return list(combined)

def get_top_commanders(timeframe='year', offset=0, limit=100):
    """timeframe: 'year', 'month', or 'week'"""
    url = f"{API_BASE}/commanders/{timeframe}.json"
    print(f"🌐 Fetching top commanders for the {timeframe}...")
    try:
        r = session.get(url, timeout=10)
        data = r.json()
        cardlists = data.get('container', {}).get('json_dict', {}).get('cardlists', [])
        if not cardlists: return ["FAIL TO FIND"]
        # Logic to find the list—usually the first entry contains the rankings
        raw_cards = cardlists[0].get('cardlist') or cardlists[0].get('cardviews', [])
        #print(json.dumps(first_list, indent=2)[:100]) #Debugging
        return [c['name'] for c in raw_cards[offset : offset + limit] if 'name' in c]
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error fetching commander list: {e}")
    except KeyError as e:
        print(f"❌ Structure Error: Missing key {e} in EDHREC JSON.")
    except Exception as e:
        print(f"❌ Error fetching {timeframe} list: {e}")
        return []

def get_average_deck(commander_name):
    # Sanitize name for URL (lowercase, hyphens, remove special chars)
    slug = sanitize_slug(commander_name)
    url = f"{API_BASE}/average-decks/{slug}.json"
    print(slug)
    # print(url)
    try:
        r = session.get(f"{API_BASE}/average-decks/{slug}.json", timeout=10)
        cardlists = r.json().get('container', {}).get('json_dict', {}).get('cardlists', [])
        deck = []
        for clist in cardlists:
            for card in clist.get('cardviews', []):
                if card.get('name'): deck.append({'name': card['name'].strip(), 'synergy': 0.25}) # default synergy of 0.25
        return deck
    except Exception as e:
        print(f" ⚠️ Error fetching deck for {commander_name}: {e}")
        return []

def get_as_commander_data(commander_name):
    slug = sanitize_slug(commander_name)
    url = f"{API_BASE}/commanders/{slug}.json"
    try:
        r = session.get(url, timeout=10)
        cardlists = r.json().get('container', {}).get('json_dict', {}).get('cardlists', [])
        synergy_cards = []
        for clist in cardlists:
            for card in (clist.get('cardlist') or clist.get('cardviews', [])):
                synergy_cards.append({'name': card.get('name'),'synergy': card.get('synergy', 0.0)})
        return synergy_cards
    except Exception as e:
        print(f"      ⚠️ Error fetching commander page for {commander_name}: {e}")
        return []

def get_similar_commanders(commander_name, limit=3):
    """Fetches similar commanders from EDHREC to expand discovery."""
    slug = sanitize_slug(commander_name)
    url = f"{API_BASE}/commanders/{slug}.json"

    try:
        r = session.get(url, timeout=10)
        data = r.json()
        similar_data = data.get('similar', []) or data.get('container', {}).get('json_dict', {}).get('similar', [])
        similar = [c['name'] for c in similar_data[:limit] if 'name' in c]
        if similar:
            print(f"    🧬 Found {len(similar)} similar commanders for {commander_name}")
        return similar
    except Exception as e:
        print(f" ⚠️ Could not find similar commanders for {commander_name}: {e}")
        return []

def expand_pool_pre_train(initial_names, interval):
    """Network Lag Fix: Expands the list of commanders BEFORE training starts."""
    expanded = set(initial_names)
    discovery_list = list(initial_names)
    print(f"🌐 Running Discovery Phase on {len(discovery_list)} commanders...")
    # We only check similarity for the top ones to prevent infinite growth
    for i in range(0, len(discovery_list), interval):
        name = discovery_list[i]
        sims = get_similar_commanders(name)
        for s in sims:
            if s not in expanded:
                expanded.add(s)
    print(f"✅ Discovery Complete: Pool grew from {len(initial_names)} to {len(expanded)}")
    return list(expanded)

def get_hard_negative(commander_key, deck_names, all_names, cards_data):
    """
    Finds a card that matches colors but ISN'T in the deck.
    This forces the model to learn mechanics over just colors.
    """
    # Create a quick lookup for synergy scores in this specific deck
    synergy_map = {c['name'].lower(): c.get('synergy', 0.0) for c in cards_data}# 1. Try to find a 'Hard' negative
    if random.random() > 0.5:
        commander_context = card_db.get(commander_key, "")
        # Extract color string (e.g., "WUBRG") and convert to a set
        raw_target = commander_context.split('|')[1] if '|' in commander_context else ""
        target_colors = set(raw_target.replace("Identity: ", "").strip())
        for _ in range(10):
            candidate = random.choice(all_names)
            candidate_synergy = synergy_map.get(candidate, -1.0)
            if candidate not in deck_names or candidate_synergy <= 0.0:
                candidate_context = card_db.get(candidate, "")
                raw_candidate = candidate_context.split('|')[1] if '|' in candidate_context else ""
                candidate_colors = set(raw_candidate.replace("Identity: ", "").strip())
                # Check if candidate fits within the commander's color identity
                if candidate_colors and candidate_colors.issubset(target_colors):
                    return precomputed_embeddings[candidate]
    # 2. Fallback to random
    return precomputed_embeddings[random.choice(all_names)]

def save_weights_to_json(model, filepath):
    """Converts PyTorch state_dict tensors to lists and saves as JSON."""
    state_dict = model.state_dict()
    weights_printable = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
    with open(filepath, 'w') as f: json.dump(weights_printable, f)
    print(f"✅ Weights also saved as JSON to: {filepath}")

def save_training_corpus(corpus, filename="training_corpus_log.json"):
    log_data = {"total_count": len(corpus), "commanders": list(corpus.keys()), "last_updated": time.ctime()}
    with open(filename, 'w') as f: json.dump(log_data, f, indent=4)
    print(f"📋 Training corpus log updated: {filename}")

# --- 3. TRAINING LOGIC ---
def train_on_synergy(commander_name, cards_data, negative_ratio, optimizer, era_progress):
    refiner.train()
    # DYNAMIC MARGIN: Increase margin as training progresses
    margin_start = 0.75
    margin_end = 1.0
    current_margin = margin_start + (margin_end - margin_start) * era_progress

    commander_key = commander_name.lower()
    if commander_key not in precomputed_embeddings:
        print(f"  [Skip] {commander_name} not found in precomputed embeddings.")
        return 0.0

    anchor_emb = precomputed_embeddings[commander_key]
    deck_names = {c['name'].lower() for c in cards_data}
    anchors, positives, negatives = [], [], []

    for card in cards_data:
        pos_name = card['name'].lower()
        if pos_name in precomputed_embeddings and card['synergy'] > 0.1:
            pos_emb = precomputed_embeddings[pos_name]
            for _ in range(negative_ratio):
                neg_emb = get_hard_negative(commander_key, deck_names, all_names, cards_data)
                anchors.append(anchor_emb)
                positives.append(pos_emb)
                negatives.append(neg_emb)

    if not anchors:
        print(f"  [Skip] No valid synergy pairs for {commander_name}")
        return 0.0

    anchors_t, positives_t, negatives_t = torch.stack(anchors).to(device), torch.stack(positives).to(device), torch.stack(negatives).to(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=('cuda' if device == 'cuda' else 'cpu'), enabled=(device == 'cuda')):
        a_out = refiner(anchors_t)
        p_out = refiner(positives_t)
        n_out = refiner(negatives_t)
        loss = criterion(a_out, p_out, n_out, current_margin=current_margin)

    if scaler:
        scaler.scale(loss).backward()
        # GRADIENT CLIPPING: Prevents loss spikes from breaking the model
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"    ✅ Batch: {len(positives)} pairs | Loss: {loss.item():.5f} | Margin: {current_margin:.2f}")
    return loss.item()

# --- 4. MAIN LOOP ---
def run_cycle(cycle_number, batch_size, optimizer, training_corpus, era_progress, commanders_pool, similarity_interval):
    # Calculate current batch range
    idx = (cycle_number - 1) * batch_size
    batch_end = min(idx + batch_size, len(commanders_pool))
    batch_commanders = commanders_pool[idx:batch_end]
    negative_ratio = 10
    cycle_losses = []

    print(f"\n--- Starting Cycle #{cycle_number} (Rank {idx + 1} to {batch_end}) ---")
    print(f"Commanders: {'|'.join(batch_commanders)}")

    if not batch_commanders:
        print("⚠️ No commanders found in this batch range.")
        return []

    # Local counter for this cycle to determine similarity triggers
    # If you want it to be global across cycles, you'd pass a persistent counter in
    for i, name in enumerate(batch_commanders):
        # Global-style index for the similarity check
        global_count = idx + i + 1
        print(f"#{global_count}: 🔍 Processing: {name}")

        # Use the pre-fetched data which already combines Average Deck + As Commander Page
        if name in training_corpus and training_corpus[name]:
            print(f"  -> Training on Pre-fetched Data (Average Deck + As Commander)...")
            loss_val = train_on_synergy(name, training_corpus[name], negative_ratio, optimizer, era_progress)
            if isinstance(loss_val, (int, float)) and loss_val > 0:
                cycle_losses.append(loss_val)
        else:
            # Fallback if for some reason the data wasn't pre-fetched
            print(f"  -> Data missing for {name}. Fetching now...")
            combined_data = fetch_commander_training_data(name)
            if combined_data:
                loss_val = train_on_synergy(name, training_corpus[name], negative_ratio, optimizer, era_progress)
                if isinstance(loss_val, (int, float)) and loss_val > 0:
                    cycle_losses.append(loss_val)
        # Similarity Logic (Every X commanders)
        if global_count % similarity_interval == 0:
            print(f"🧬 Counter reached {global_count}: Finding similar commanders for {name}...")
            similar_list = get_similar_commanders(name)
            for similar_name in (similar_list or []):
                if similar_name not in training_corpus:
                    print(f"  ✨ New Discovery: {similar_name}. Fetching data...")
                    training_corpus[similar_name] = fetch_commander_training_data(similar_name)
                    commanders_pool.append(similar_name)
                print(f"   -> Training on Similar Commander: {similar_name}")
                loss_val = train_on_synergy(name, training_corpus[name], negative_ratio, optimizer, era_progress)
                if isinstance(loss_val, (int, float)) and loss_val > 0:
                    cycle_losses.append(loss_val)
    return cycle_losses

if __name__ == "__main__":
    batch_sz = 50
    total_eras = 5
    MODEL_BASE_NAME = "synergy_refiner"
    CORPUS_LOG_PATH = "trained_commanders_log.json"
    SIMILARITY_CHECK_INTERVAL = 25  # Check for similar commanders every X commanders

    # Pre-fetch week, month, and top-100 year
    print("🌐 Pre-fetching deck data for Week, Month, and Year Top 100...")
    master_names = set()
    for tf in ['year', 'month', 'week']:
        print(f"Fetching top 100 of the {tf}...")
        master_names.update(get_top_commanders(timeframe=tf, limit=100))

    # 2. EXPAND POOL (Network Phase 2)
    # Convert to list so we can shuffle
    shuffled_list = list(master_names)
    random.shuffle(shuffled_list)
    final_pool = expand_pool_pre_train(shuffled_list, SIMILARITY_CHECK_INTERVAL)

    # 3. PARALLEL DATA FETCH
    print(f"📥 Fetching deck data for {len(final_pool)} commanders using Multi-threading...")
    training_corpus = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        # map function allows parallel execution of the fetch function
        results = list(executor.map(fetch_commander_training_data, final_pool)) # might need rate limiting
        training_corpus = dict(zip(final_pool, results))

    # 4. START TRAINING (GPU/CPU Phase - NO MORE IDLING)
    actual_dim = precomputed_embeddings[all_names[0]].shape[0]
    refiner = SynergyRefiner(actual_dim).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=5e-4, weight_decay=1e-3)
    # Add a scheduler to drop the rate if loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1,
        threshold=5e-2,       # <--- % improvement required
        threshold_mode='rel', # 'rel' for relative, 'abs' for absolute
        # verbose=True
    )
    # Dictionary to store hyperparameter performance
    performance_metrics = {}
    print(f"🏗️ Refiner Setup: Input Dimension = {actual_dim}")
    print(f"Unique commanders to train: {len(master_names)}")

    try:
        for era in range(1, total_eras + 1):
            print(f"\n\n{'='*10} STARTING ERA {era} of {total_eras} {'='*10}")
            commanders_pool = list(training_corpus.keys())
            random.shuffle(commanders_pool)

            era_losses = []
            idx = 0
            while idx < len(commanders_pool):
                current_progress = (era - 1) / total_eras
                cycle_num = (idx // batch_sz) + 1

                # Execute the refactored cycle
                losses = run_cycle(
                    cycle_number=cycle_num,
                    batch_size=batch_sz,
                    optimizer=optimizer,
                    training_corpus=training_corpus,
                    era_progress=current_progress,
                    commanders_pool=commanders_pool,
                    similarity_interval=SIMILARITY_CHECK_INTERVAL
                )
                era_losses.extend(losses)

                idx += batch_sz
                num_cycles = (len(commanders_pool) + batch_sz - 1) // batch_sz
                print(f"Cycle {cycle_num}/{num_cycles} in Era {era} complete.")
                print(f"Total Unique Commanders in Knowledge Base: {len(training_corpus)}")

            # --- ERA STATS ---
            avg_era_loss = sum(era_losses) / len(era_losses) if era_losses else 0
            performance_metrics[f"era_{era}"] = avg_era_loss
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_era_loss)
            print(f"\n📊 ERA {era} SUMMARY")
            print(f"Average Loss: {avg_era_loss:.6f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Total Samples: {len(era_losses)}")

            era_save_path = f"{MODEL_BASE_NAME}_era_{era}.pth"
            torch.save(refiner.state_dict(), era_save_path)
            print(f"✅ Era {era} weights saved to: {era_save_path}")

        # --- FINAL EXPORT ---
        # Final Standard Save
        torch.save(refiner.state_dict(), f"{MODEL_BASE_NAME}_final.pth")
        save_weights_to_json(refiner, f"{MODEL_BASE_NAME}_final.json")
        save_training_corpus(training_corpus, CORPUS_LOG_PATH)
        print("\n🏆 Training Sequence Complete. All eras finished and final weights exported.")
    except KeyboardInterrupt:
        print("\n⚠️ Shutdown signal received. Saving emergency checkpoint...")
        torch.save(refiner.state_dict(), f"{MODEL_BASE_NAME}_interrupted.pth")
        print("Progress saved to: synergy_refiner_interrupted.pth")
