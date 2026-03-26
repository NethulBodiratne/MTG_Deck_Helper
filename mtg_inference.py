# --- INFERENCE BLOCK ---
import torch
import torch.nn as nn
import numpy as np

ORACLE_DATA_PATH = 'oracle-cards.json'
EMBEDDINGS_CACHE_PATH = 'card_embeddings.pt'

print("Loading card metadata...")
with open(ORACLE_DATA_PATH, 'r', encoding='utf-8') as f:
    cards_data = json.load(f)
# Map name (lowercase) to its type_line
# We use .lower() to match your cmd_key logic
card_metadata = {card['name'].lower(): card.get('type_line', '') for card in cards_data}
print(f"Metadata loaded for {len(card_metadata)} cards.")

def get_synergy_score(commander_name, card_name):
    # 1. Set to evaluation mode
    refiner.eval()

    # 2. Retrieve from precomputed_embeddings (already on the correct device)
    # We use .get() to avoid KeyError if a card name is slightly off
    cmd_emb = precomputed_embeddings.get(commander_name.lower())
    card_emb = precomputed_embeddings.get(card_name.lower())

    if cmd_emb is None:
        return f"❌ Commander '{commander_name}' not in cache."
    if card_emb is None:
        return f"❌ Card '{card_name}' not in cache."

    # 3. Vectorize and Refine
    with torch.no_grad():
        # Ensure we are on the same device as the refiner
        # unsqueeze(0) adds the batch dimension required by Linear layers
        c_ref = refiner(cmd_emb.unsqueeze(0))
        t_ref = refiner(card_emb.unsqueeze(0))

        # Cosine similarity on refined vectors
        score = F.cosine_similarity(c_ref, t_ref).item()
    return score

# --- ENTER COMMANDER FOR RECOMMENDATIONS ---
def recommend_cards(commander_name, top_n=10, filter_colors=True):
    refiner.eval()

    # 1. Prepare Commander Vector
    cmd_key = commander_name.lower()
    if cmd_key not in precomputed_embeddings:
        return f"❌ Commander '{commander_name}' not found."

    with torch.no_grad():
        cmd_vec = refiner(precomputed_embeddings[cmd_key].unsqueeze(0))
        all_keys = list(precomputed_embeddings.keys())
        all_raw_embs = torch.stack([precomputed_embeddings[k] for k in all_keys])
        all_refined_embs = refiner(all_raw_embs)
        scores = torch.matmul(all_refined_embs, cmd_vec.T).squeeze()

    # 2. Package results and check if they are Lands
    results = []
    for i in range(len(all_keys)):
        name = all_keys[i]
        # Check metadata for "Land" in the type line
        type_line = card_metadata.get(name, "").lower()
        is_land = "land" in type_line

        results.append({
            "name": name,
            "score": scores[i].item(),
            "is_land": is_land
        })

    # Sort by score descending
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # 3. Filter out the commander and split into Land/Non-Land
    filtered_results = [r for r in results if r['name'] != cmd_key]

    non_lands = [r for r in filtered_results if not r['is_land']][:top_n]
    lands_only = [r for r in filtered_results if r['is_land']][:top_n]

    # 4. Print Two Tables
    def print_section(title, items):
        print(f"\n📊 {title} for {commander_name}:")
        print(f"{'Rank':<5} | {'Card Name':<35} | {'Match'}")
        print("-" * 55)
        for idx, res in enumerate(items, 1):
            print(f"{idx:<5} | {res['name'].title():<35} | {res['score']:.4f}")

    print_section(f"Top {top_n} Non-Land Recommendations", non_lands)
    print_section(f"Top {top_n} Land Recommendations", lands_only)

# --- TEST CASES ---
test_scenarios = {
    "Kambal, Profiteering Mayor": [
        "Adeline, Resplendent Cathar",   # 🔥 High: Frequent token generation
        "Mondrak, Glory Dominus",       # 🔥 High: Token doubling payoff
        "Rest in Peace",                # ❄️ Low: Anti-Synergy (shuts off your own death triggers)
        "Colossal Dreadmaw",            # ❌ Invalid: Wrong colors / No synergy
        "Suntail Hawk"                  # ❄️ Low: Right colors, but zero impact
    ],
    "The Archimandrite": [
        "Aetherflux Reservoir",         # 🔥 High: Lifegain payoff/win-con
        "Ivory Tower",                  # 🔥 High: Redundant "Lifegain from cards" effect
        "Tainted Remedy",               # ❄️ Low: Anti-Synergy (blocks your own lifegain logic)
        "Krenko, Tin Street Kingpin",   # ❌ Invalid: Wrong colors
        "Giant Spider"                  # ❄️ Low: Right colors, but zero synergy
    ],
    "Arcades, the Strategist": [
        "Wall of Omens",                # 🔥 High: Low-cost Defender with draw
        "High Alert",                   # 🔥 High: Redundant "Toughness matters" effect
        "Humility",                     # ❄️ Low: Anti-Synergy (wipes your toughness-to-power ability)
        "Urabrask the Hidden",          # ❌ Invalid: Wrong colors
        "Charging Badger"               # ⚠️ Mid: Right colors, but lacks Defender tag
    ],
    "Omnath, Locus of All": [
        "Maelstrom Wanderer",           # 🔥 High: WUBRG synergy / Cascade
        "Leyline of the Guildpact",     # 🔥 High: Fixing and color-matters payoff
        "Blood Moon",                   # ❄️ Low: Anti-Synergy (shuts off your 5-color mana base)
        "Island",                       # ✅ Valid: Basic land / Fixing
        "Healing Salve"                 # ❄️ Low: Right colors, but zero impact/pip value
    ]
}

# --- EXECUTION ---
print(f"{'Commander':<30} | {'Card':<30} | {'Synergy Score'}")
print("-" * 80)
for cmd, cards in test_scenarios.items():
    # Filter out strings and only keep floats for calculation
    raw_results = [get_synergy_score(cmd, c) for c in cards]
    scores = [s for s in raw_results if isinstance(s, (int, float))]
    if not scores:
        print(f"⚠️ No valid scores found for {cmd}")
        continue
    avg = sum(scores) / len(scores)
    mi, ma = min(scores), max(scores)
    # Thresholds (avoiding division by zero if all scores are identical)
    range_val = ma - mi if ma != mi else 1.0
    # Quartile Thresholds
    q1 = mi + 0.25 * range_val  # Bottom 25%
    q2 = mi + 0.50 * range_val  # Middle (Median point)
    q3 = mi + 0.75 * range_val  # Top 25%
    for card, result in zip(cards, raw_results):
        if isinstance(result, str):
            print(f"{cmd:<30} | {card:<30} | {result}")
        else:
            # Determine label based on quartiles
            if result >= q3: label = "🔥 High"
            elif result >= q2: label = "Neutral"
            elif result >= q1: label = "❄️ Low"
            else: label = "❌ Bad"
            print(f"{cmd:<30} | {card:<30} | {result:.4f} ({label})")
    print(f"{'-' * 70} AVG: {avg:.4f}\n")
    print("-" * 80)

# --- EXECUTION ---
recommend_cards("Kambal, Profiteering Mayor", top_n=25)
recommend_cards("The Archimandrite", top_n=25)
recommend_cards("Arcades, the Strategist", top_n=25)
recommend_cards("Omnath, Locus of All", top_n=25)
recommend_cards("Jodah, the Unifier", top_n=25)
