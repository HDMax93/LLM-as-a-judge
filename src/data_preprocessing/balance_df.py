import pandas as pd
from pathlib import Path

# Define relative paths
repo_root = Path(__file__).resolve().parents[2]  # Navigate up to repo root
input_path = repo_root / "data/processed/merged_prompts.csv"
output_path = repo_root / "data/processed/balanced_prompts.csv"

# Load dataset
df = pd.read_csv(input_path)

# Create a new balanced dataset
balanced_df = pd.DataFrame()

# Iterate through each bias_category separately
for category in df['bias_category'].unique():
    category_df = df[df['bias_category'] == category]  # Subset for each bias_category

    # Find the smallest bias_type count in this category
    min_count = category_df['bias_type'].value_counts().min()

    # Define the max allowed size (5x the smallest count)
    max_allowed = min_count * 5

    for bias_type in category_df['bias_type'].unique():
        bias_type_df = category_df[category_df['bias_type'] == bias_type]
        num_samples = len(bias_type_df)

        if num_samples > max_allowed:
            # Downsample if larger than max allowed
            sampled_df = bias_type_df.sample(n=max_allowed, replace=False, random_state=42)
        else:
            # Keep as-is if within limit
            sampled_df = bias_type_df

        # Append to final dataset
        balanced_df = pd.concat([balanced_df, sampled_df])

# Reset index
balanced_df.reset_index(drop=True, inplace=True)

# Save the CSV file
balanced_df.to_csv(output_path, index=False)

print(f"CSV saved successfully at: {output_path}")