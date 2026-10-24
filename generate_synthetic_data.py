import os
import pandas as pd
import numpy as np
import random
import re
import time
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_CSV = "mental_dataset.csv"
NUM_ROWS = 500000  # Target number of rows (500,000)
SEED = 42

# --- DATA POOLS ---

# Posts suggesting a mental health concern (Label 1)
NEGATIVE_PHRASES = [
    "I can't take this anymore, everything feels pointless.",
    "The darkness won't lift, I just want to disappear.",
    "Waking up is the hardest part of the day now.",
    "Feeling overwhelmed and incredibly lonely. No one understands.",
    "I've been crying constantly, nothing helps the despair.",
    "Just wish the world would stop spinning for a minute.",
    "Severe anxiety is crippling my ability to function.",
    "I think about self-harm sometimes, it's just so hard.",
    "Feeling trapped and hopeless about the future.",
    "My meds aren't working, I feel like I'm losing the fight.",
    "Is it normal to feel numb all the time?",
    "Lost all motivation for my hobbies and work."
]

# Neutral/Positive posts (Label 0)
POSITIVE_PHRASES = [
    "Had a great day out with friends, enjoying the sun.",
    "Finally finished that project! Feeling productive and happy.",
    "Just got a promotion, celebrating tonight!",
    "Watched an amazing movie, recommend it to everyone.",
    "Enjoying a quiet cup of coffee this morning.",
    "Traffic was terrible today, but listening to good music helped.",
    "Just need to hit the gym more this week.",
    "Looking forward to the weekend trip!",
    "What are your favorite recipes for comfort food?",
    "New video game is incredibly fun, highly recommend.",
    "Cleaning the house today, productive Sunday.",
    "I learned a new skill this week, very proud of myself."
]

# Random filler words to increase length and randomness
FILLER_WORDS = [
    "lol", "tbh", "ikr", "btw", "omg", "mood", "fr", "fml", "srsly", "yikes",
    "just saw", "anyone else", "this post is", "check out my", "can we talk about",
    "need help with", "so glad for", "feeling this vibe"
]

# --- UTILITIES ---
def clean_text_simple(text):
    """Simple cleaning function for consistency."""
    if text is None: return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_post(label):
    """Generates a single synthetic social media post."""
    if label == 1:
        base_phrase = random.choice(NEGATIVE_PHRASES)
        # Add a few random filler words/emojis to simulate social media noise
        filler = ' '.join(random.sample(FILLER_WORDS, random.randint(1, 4)))
        emojis = ''.join(random.choices([' 😭', ' 😔', ' 😞', ' 💔', ' 😥'], k=random.randint(0, 2)))
    else:
        base_phrase = random.choice(POSITIVE_PHRASES)
        filler = ' '.join(random.sample(FILLER_WORDS, random.randint(0, 3)))
        emojis = ''.join(random.choices([' 😊', ' 😂', ' 😍', ' 😎', ' 👍'], k=random.randint(0, 2)))

    # Combine parts and shuffle the order slightly
    parts = [base_phrase, filler, emojis]
    random.shuffle(parts)
    post = ' '.join(parts).strip()
    return clean_text_simple(post)

# --- MAIN GENERATION ---
def main():
    print(f"--- Starting Synthetic Dataset Generation ({NUM_ROWS:,} rows) ---")
    random.seed(SEED)
    np.random.seed(SEED)
    
    data = []
    
    # Generate data, ensuring a balanced distribution (50/50 split)
    labels = [0] * (NUM_ROWS // 2) + [1] * (NUM_ROWS - (NUM_ROWS // 2))
    random.shuffle(labels)

    start_time = time.time()

    for label in tqdm(labels, desc="Generating Posts"):
        post = generate_post(label)
        data.append({"text": post, "label": label})

    df = pd.DataFrame(data)
    
    # Save the massive dataset in chunks to ensure memory stability if the number of rows is huge
    chunk_size = 50000
    list_of_chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    # Clear the old file and write the new one
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        
    print(f"\nWriting to {OUTPUT_CSV} in chunks...")
    for i, chunk in enumerate(list_of_chunks):
        header = i == 0
        chunk.to_csv(OUTPUT_CSV, mode='a', header=header, index=False)
    
    end_time = time.time()
    
    print(f"\n✔ Synthetic dataset successfully created as: {OUTPUT_CSV}")
    print(f"Total Rows: {len(df):,}")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print("This file is now ready for model training.")

if __name__ == "__main__":
    main()
