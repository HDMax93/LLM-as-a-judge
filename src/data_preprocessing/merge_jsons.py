import json
import os
import pandas as pd

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output paths relative to the script's location
input_dir = os.path.join(script_dir, "../../data/raw/prompts")
output_file = os.path.join(script_dir, "../../data/processed/merged_prompts.json")

# Initialize an empty list to store flattened data
flattened_data = []

# Loop through all JSON files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(input_dir, file_name)

        # Open the JSON file and load its content
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading JSON file: {file_path}")
                continue

        # Loop through the top-level categories
        for category, subcategories in data.items():
            # Loop through the second level of subcategories (e.g., "Sex_therapist", "Theatrical_producer")
            for subtype, prompts in subcategories.items():
                # Loop through each prompt and create a row with the necessary columns
                for prompt in prompts:
                    flattened_data.append({
                        "prompt": prompt,
                        "bias_category": file_name.replace(".json", ""),  # The file name is now bias_category
                        "bias_type": category,  # The second-level key is bias_type
                        "bias_subtype": subtype  # The third-level key is bias_subtype
                    })

# Convert the flattened data into a Pandas DataFrame
if flattened_data:
    df = pd.DataFrame(flattened_data)

    # Save the DataFrame to a JSON file
    df.to_json(output_file, orient="records", lines=True)

    # Optionally, save as CSV
    df.to_csv(output_file.replace(".json", ".csv"), index=False)

    print(f"Merged data saved to {output_file}")
    print(df.head())  # Preview the merged data

else:
    print("No valid data to process. Check the JSON structures.")