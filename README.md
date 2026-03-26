# MTG_Deck_Helper
# Synergy Refiner: Deep Metric Learning for Commander

This project is a machine learning pipeline designed to calculate the mechanical and thematic relationships between Magic: The Gathering cards. It specifically focuses on the Commander (EDH) format, where the relationship between a "Special" card (the Commander) and the rest of the deck is the primary driver of gameplay.

The system uses a custom neural network to transform raw card data into high-dimensional mathematical vectors. In this mathematical space, cards that work well together are placed close to each other, while cards that do not have synergy are pushed far apart.

---

## Technical Architecture

The model builds a "Game-Sense" vector for every card by combining two distinct types of data. This hybrid approach ensures the AI understands both the written English on the card and the underlying game rules.

### 1. Semantic Layer (Text Meaning)
The system uses a pre-trained model called a SentenceTransformer. This model analyzes the "Oracle Text" (the official wording of a card's abilities) and converts it into a numerical representation of its meaning. This allows the AI to recognize that "Draw a card" and "Put the top card of your library into your hand" are functionally similar, even though the words are different.

### 2. Mechanical Layer (Game Rules)
Because text models can sometimes miss specific game-rule nuances, an 81-dimensional "bitmask" is added. This layer explicitly tracks:
* **Keywords:** 70+ specific mechanics ranging from "Flying" to 2026-era mechanics like "Manifest Dread" and "Warp."
* **Card Types:** Identification of whether a card is a Creature, Enchantment, Battle, Room, etc.
* **Mana Production:** Whether a card produces specific colours of mana or provides "ramp" (placing extra lands onto the battlefield).
* **Mana Value:** The casting cost of the card, normalized for the model.

---

## Training Process

The model is trained using a technique called **Multi-Similarity Loss**. This is a method of teaching the AI by showing it three things at once:
1.  **The Anchor:** A specific Commander (e.g., *Jodah, the Unifier*).
2.  **The Positive:** A card known to be good with that Commander based on community data.
3.  **The Negative:** A card that is in the same colours but does not actually help the Commander's strategy.

By comparing these three, the model learns to ignore coincidental similarities (like sharing a colour) and focus on functional similarities (like sharing a mechanic).

### The 5-Era Cycle
Training is divided into five "Eras." During each era, the model's "Margin" is increased. This means that as the AI learns, the requirements for what counts as a "match" become stricter, forcing the model to find deeper and more subtle connections between cards.

---

## Data Integration

The system pulls data from two primary sources:
* **Scryfall:** Used for official technical specifications, high-resolution text, and legalities of every card in the game.
* **EDHREC:** Used to gather real-world deck-building trends. The system uses multi-threading to quickly download "Average Deck" lists and synergy scores for hundreds of different Commanders simultaneously.

---

## Output and Usage

Upon completion, the model exports two types of files:
* **PyTorch Weights (.pth):** For continued training or use in Python applications.
* **JSON Weights (.json):** A standardized format that allows the mathematical "brain" of the model to be used in web browsers or mobile applications without requiring a powerful server.

The final result is a system that can take any card—even a brand-new one—and immediately identify which existing cards it supports best based on its mathematical "location" in the synergy map.

---

## Inference and Recommendation System

Beyond training, the project includes an inference engine that allows users to query the model for specific card relationships or generate full deck-building recommendations.

### Synergy Scoring
The system can compare any two cards—typically a Commander and a potential deck addition—to produce a **Synergy Score**. This score is calculated using cosine similarity on the "refined" vectors.
* **High Scores:** Indicate strong mechanical alignment (e.g., a token-generating card paired with a Commander that rewards token creation).
* **Low Scores:** Indicate anti-synergy or a total lack of mechanical overlap, even if the cards share the same colours.

### Automated Recommendations
The recommendation engine can scan the entire database to find the best matches for a specific Commander. To provide practical utility for deck building, the system automatically categorizes these results into two groups:
1.  **Non-Land Recommendations:** Creatures, spells, and artifacts that drive the deck's primary strategy.
2.  **Land Recommendations:** Optimized mana-base suggestions that align with the Commander’s colour identity and utility needs.

### Validation and Testing
The project includes a testing suite with "Test Scenarios" to ensure the model's accuracy. These scenarios verify that the AI correctly identifies:
* **High Synergy:** Cards that are community-proven staples for a specific strategy.
* **Anti-Synergy:** Cards that might share a colour but actually disable the Commander's abilities (e.g., a card that stops lifegain paired with a lifegain Commander).
* **Invalid/Low Impact:** Cards that are technically legal to play but offer no strategic benefit.

---
