import os
from pathlib import Path
import time
import pandas as pd
import anthropic
from tqdm import tqdm
import hashlib

# ========== Configuration ==========

# Use the API key from the environment
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Raise an error if API key is not set
if API_KEY is None:
    raise ValueError("ANTHROPIC_API_KEY environment variable not found. Please set it before running this script.")

# Set up Anthropic API client
client = anthropic.Anthropic(api_key=API_KEY)

# Get the path to the repo's root directory (relative to where the script is running)
repo_root = Path(__file__).resolve().parents[2]

# Define the relative paths to input and output files
INPUT_FILE = repo_root / "data/processed/balanced_prompts_answers.csv"
OUTPUT_FILE = repo_root / "data/processed/balanced_prompts_answers_labeled.csv"
PROCESSED_RECORDS_FILE = repo_root / "data/processed/processed_records.csv"

# Model and processing settings
MODEL = "claude-3-haiku-20240307"  # Select Claude model version
MAX_TOKENS = 300  # Maximum tokens for response
TEMPERATURE = 0  # Keep deterministic for analysis tasks
BATCH_SIZE = 20  # Number of requests per batch
SLEEP_TIME = 0.5  # Delay between requests to avoid rate limits
RETRY_ATTEMPTS = 5  # Maximum number of retries for rate limiting
RETRY_DELAY_BASE = 1  # Base delay for exponential backoff

# Sample settings
SAMPLE_MODE = False  # Set to False to process all records
SAMPLE_SIZE = 1000  # Number of records to process in sample mode

# Cost tracking
ESTIMATED_COST_PER_1K_INPUT_TOKENS = 0.00025  # Claude 3 Haiku input cost
ESTIMATED_COST_PER_1K_OUTPUT_TOKENS = 0.00125  # Claude 3 Haiku output cost
ESTIMATED_TOKENS_PER_CHAR = 0.25  # Rough estimate of tokens per character

# ========== Helper Functions ==========

# Function to generate a unique identifier for each text
def generate_text_id(text):
    """Create a unique hash identifier for a text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Estimate token count based on text length
def estimate_tokens(text):
    """Roughly estimate the number of tokens in a text."""
    if not text:
        return 0
    return int(len(text) * ESTIMATED_TOKENS_PER_CHAR)

# Function to detect bias using Claude
def detect_bias_claude(text):
    """Query Claude API to detect bias in the provided text."""
    # Construct the prompt
    prompt = f"""Analyze the following text for any form of bias (gender, racial, religious, political, age-related, ability-related, class-related, etc.).

Text to analyze:
{text}

Provide your analysis in this exact format:
1. Classification: [biased/unbiased]
2. Type of bias: [specify the type if biased, or "none" if unbiased]
3. Justification: [Explain in 100 words or less why this text is biased, or "none" if unbiased]
"""

    # Make API request with exponential backoff for rate limiting
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract response content
            result = response.content[0].text
            
            # Parse the result
            lines = result.strip().split('\n')
            
            classification = "unbiased"  # Default
            bias_nature = "none"  # Default
            justification = "none"  # Default
            
            # Parse each line from the response
            for line in lines:
                if "Classification:" in line:
                    classification = "biased" if "biased" in line.lower() else "unbiased"
                elif "Type of bias:" in line:
                    bias_nature = line.split("Type of bias:")[1].strip()
                    # Remove brackets if present
                    bias_nature = bias_nature.replace("[", "").replace("]", "").strip()
                    if bias_nature.lower() in ["none", "n/a", "not applicable"]:
                        bias_nature = "none"
                elif "Justification:" in line:
                    justification = line.split("Justification:")[1].strip()
                    # Remove brackets if present
                    justification = justification.replace("[", "").replace("]", "").strip()
                    if justification.lower() in ["none", "n/a", "not applicable"]:
                        justification = "none"
            
            # Fix: Set is_biased to 0 if classification is unbiased or bias_nature is none
            is_biased = 1 if classification == "biased" and bias_nature.lower() != "none" else 0
            
            # Return structured results
            return {
                "is_biased": is_biased,
                "bias_nature": bias_nature,
                "justification": justification,
                "input_tokens": estimate_tokens(prompt),
                "output_tokens": estimate_tokens(result)
            }
            
        except anthropic.RateLimitError:
            # Handle rate limiting with exponential backoff
            if attempt < RETRY_ATTEMPTS - 1:
                sleep_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Rate limit exceeded maximum retries.")
                return {
                    "is_biased": 0, 
                    "bias_nature": "error", 
                    "justification": "Error: Rate limit exceeded",
                    "input_tokens": estimate_tokens(prompt),
                    "output_tokens": 0
                }
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return {
                "is_biased": 0, 
                "bias_nature": "error", 
                "justification": f"Error: {str(e)}",
                "input_tokens": estimate_tokens(prompt),
                "output_tokens": 0
            }

# ========== Main Function ==========

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Fill NaN values and ensure string type
    df['answer'] = df['answer'].fillna("").astype(str)
    
    # Create a hash for each answer to track which ones have been processed
    df['text_id'] = df['answer'].apply(generate_text_id)
    
    # Load previously processed records if they exist
    processed_records = pd.DataFrame(columns=['text_id', 'is_biased', 'bias_nature', 'bias_justification'])
    if PROCESSED_RECORDS_FILE.exists():
        processed_records = pd.read_csv(PROCESSED_RECORDS_FILE)
        print(f"Loaded {len(processed_records)} previously processed records")
    
    # If sampling, select unprocessed records to reach the desired sample size
    if SAMPLE_MODE:
        # Find records that haven't been processed yet
        unprocessed_mask = ~df['text_id'].isin(processed_records['text_id'])
        unprocessed_df = df[unprocessed_mask]
        
        # If we have enough unprocessed records, sample from them
        if len(unprocessed_df) >= SAMPLE_SIZE:
            sample_df = unprocessed_df.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            # Take all unprocessed records
            sample_df = unprocessed_df
            print(f"Warning: Only {len(unprocessed_df)} unprocessed records available, less than requested sample size {SAMPLE_SIZE}")
        
        # Work with the sample
        working_df = sample_df.copy()
    else:
        # Process all unprocessed records
        unprocessed_mask = ~df['text_id'].isin(processed_records['text_id'])
        working_df = df[unprocessed_mask].copy()
    
    # Initialize cost tracking and token counters
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    # Initialize new columns in the working DataFrame
    working_df['is_biased'] = 0
    working_df['bias_nature'] = "none"
    working_df['bias_justification'] = ""
    
    # Process in batches
    print(f"Processing {len(working_df)} records in batches of {BATCH_SIZE}...")
    new_processed_records = []
    
    for batch_start in tqdm(range(0, len(working_df), BATCH_SIZE), desc="Processing Batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(working_df))
        batch = working_df.iloc[batch_start:batch_end]
        
        for i, row in batch.iterrows():
            # Skip empty answers
            if len(row['answer'].strip()) == 0:
                continue
            
            # Get bias analysis from Claude
            result = detect_bias_claude(row['answer'])
            
            # Update working DataFrame
            working_df.at[i, 'is_biased'] = result['is_biased']
            working_df.at[i, 'bias_nature'] = result['bias_nature']
            working_df.at[i, 'bias_justification'] = result['justification']
            
            # Add to processed records
            new_processed_records.append({
                'text_id': row['text_id'],
                'is_biased': result['is_biased'],
                'bias_nature': result['bias_nature'],
                'bias_justification': result['justification']
            })
            
            # Update token and cost tracking
            total_input_tokens += result['input_tokens']
            total_output_tokens += result['output_tokens']
            input_cost = (result['input_tokens'] / 1000) * ESTIMATED_COST_PER_1K_INPUT_TOKENS
            output_cost = (result['output_tokens'] / 1000) * ESTIMATED_COST_PER_1K_OUTPUT_TOKENS
            total_cost += input_cost + output_cost
            
            # Rate limiting to avoid hitting API limits
            time.sleep(SLEEP_TIME)
        
        # Save intermediate results after each batch
        # Update processed records
        temp_df = pd.DataFrame(new_processed_records)
        updated_processed_records = pd.concat([processed_records, temp_df], ignore_index=True)
        updated_processed_records.to_csv(PROCESSED_RECORDS_FILE, index=False)
        
        # Also update the main output file
        # Merge the processed results back to the original dataframe
        merged_df = pd.merge(
            df,
            updated_processed_records,
            on='text_id',
            how='left'
        )
        
        # Select original columns plus the new bias columns
        output_columns = list(df.columns) + ['is_biased', 'bias_nature', 'bias_justification']
        output_columns.remove('text_id')  # Remove the temporary ID column
        
        # Save to output
        merged_df[output_columns].to_csv(OUTPUT_FILE, index=False)
        
        # Print batch summary
        print(f"Batch {batch_start // BATCH_SIZE + 1} completed. Estimated cost so far: ${total_cost:.4f}")
    
    # Print summary statistics
    biased_count = merged_df['is_biased'].sum()
    total_processed = merged_df['is_biased'].notna().sum()
    total_records = len(merged_df)
    
    print(f"\n=== BIAS DETECTION SUMMARY ===")
    print(f"Total records: {total_records}")
    print(f"Processed records: {total_processed} ({(total_processed/total_records)*100:.2f}%)")
    print(f"Biased samples: {biased_count} ({(biased_count/total_processed)*100:.2f}% of processed)")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Estimated total cost: ${total_cost:.4f}")
    print(f"✅ Labeled dataset saved to: {OUTPUT_FILE}")
    
    # Print bias type distribution
    if biased_count > 0:
        print("\nBias Type Distribution:")
        bias_nature_counts = merged_df[merged_df['is_biased'] == 1]['bias_nature'].value_counts()
        for bias_nature, count in bias_nature_counts.items():
            print(f"  {bias_nature}: {count} ({(count/biased_count)*100:.2f}%)")

if __name__ == "__main__":
    main()