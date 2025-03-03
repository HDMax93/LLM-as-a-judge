import openai
import pandas as pd
import os
import time
from tqdm import tqdm

# ========== Configuration ==========
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

openai.api_key = API_KEY  # Use the API key from the environment

# Get the path to the repo's root directory (relative to where the script is running)
repo_root = os.path.dirname(os.path.abspath(__file__))  # Path to the directory of the current script

# Define the relative paths to input and output files
INPUT_FILE = os.path.join(repo_root, '../../data/processed/balanced_prompts.csv')
OUTPUT_FILE = os.path.join(repo_root, '../../data/processed/balanced_prompts_answers.csv')

MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 150  # Approx. 100 words
BATCH_SIZE = 100  # Number of requests per batch
SLEEP_TIME = 1  # Delay between requests to avoid rate limits
TEST_MODE = True  # Set to False to process all prompts
TEST_SAMPLE_SIZE = 10  # Number of prompts to process in test mode
ESTIMATED_COST_PER_1K_TOKENS = 0.0015  # GPT-3.5 cost estimate

# ========== Load Data ==========
df = pd.read_csv(INPUT_FILE)

# Add answer column if not already present
if "answer" not in df.columns:
    df["answer"] = ""

# Select subset if running in test mode
if TEST_MODE:
    df = df.sample(n=TEST_SAMPLE_SIZE, random_state=42)

# Filter prompts that still need responses
df_to_process = df[df["answer"].isna() | (df["answer"] == "")].copy()

# ========== API Query Function ==========
def query_chatgpt(prompt):
    """Query OpenAI API for a given prompt and return the response."""
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "system", "content": "You are an AI assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return None

# ========== Process Prompts in Batches ==========
total_tokens_used = 0
total_cost = 0.0

print(f"Processing {len(df_to_process)} prompts...")

for i in tqdm(range(0, len(df_to_process), BATCH_SIZE)):
    batch = df_to_process.iloc[i:i + BATCH_SIZE]
    
    for idx, row in batch.iterrows():
        prompt = row["prompt"]
        response = query_chatgpt(prompt)
        
        if response:
            df.at[idx, "answer"] = response
            tokens_used = len(prompt.split()) + MAX_TOKENS
            total_tokens_used += tokens_used
            total_cost += (tokens_used / 1000) * ESTIMATED_COST_PER_1K_TOKENS
        
        time.sleep(SLEEP_TIME)  # Rate limit protection
    
    # Save progress after each batch
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Batch saved. Estimated cost so far: ${total_cost:.4f}")

# ========== Final Summary ==========
print(f"Finished processing. Total tokens used: {total_tokens_used}")
print(f"Estimated total cost: ${total_cost:.4f}")
print(f"Results saved to {OUTPUT_FILE}")