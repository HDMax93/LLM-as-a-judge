from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="copying from a non-meta parameter")

# Define relative paths
repo_root = Path(__file__).resolve().parents[2]  # Navigate up to repo root
input_path = repo_root / "data/processed/balanced_prompts_answers.csv"
output_path = repo_root / "data/processed/bias_labeled_dataset.csv"

# Load dataset
SAMPLE_MODE = False
SAMPLE_SIZE = 1000

df = pd.read_csv(input_path)
if SAMPLE_MODE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)

# Clean the dataset - fill NaN values in the 'answer' column with empty strings
df['answer'] = df['answer'].fillna("")

# Make sure all values are strings
df['answer'] = df['answer'].astype(str)

print(f"Dataset loaded with {len(df)} rows")

print("Setting up specialized bias detection models...")

# Function to get predictions in batches to avoid memory issues
def get_predictions_in_batches(texts, tokenizer, model, batch_size=8):
    all_predictions = []
    
    # Check if tokenizer has pad_token, if not, set it
    if tokenizer.pad_token is None:
        # Use eos_token as pad_token if available, otherwise add a [PAD] token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize model embeddings to account for new token
            model.resize_token_embeddings(len(tokenizer))
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting bias"):
        batch_texts = texts[i:i+batch_size]
        
        # Convert all items to strings and handle NaN values
        batch_texts = [str(text) if not pd.isna(text) else "" for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(model.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            all_predictions.append(predictions.cpu().numpy())
    
    return np.vstack(all_predictions)

# Define general bias detection models
general_bias_models = {
    'toxic_bert': "unitary/toxic-bert",
    'bias_detection': "D1V1DE/bias-detection",
    'distilbert_toxicity': "martin-ha/toxic-comment-model"
}

# Process each general bias model
for bias_model_name, model_path in general_bias_models.items():
    print(f"Processing with general model: {bias_model_name} ({model_path})...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if tokenizer has pad_token, if not, set it
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
        
        # Get bias predictions
        bias_scores = get_predictions_in_batches(df['answer'].tolist(), tokenizer, model)
        
        # Most models have bias as the positive class at index 1
        prediction_column = 1 if bias_scores.shape[1] > 1 else 0
        
        # Use standard threshold of 0.5 for general models
        df[f'{bias_model_name}_present'] = (bias_scores[:, prediction_column] > 0.8).astype(int)
        df[f'{bias_model_name}_confidence'] = bias_scores[:, prediction_column]
        
    except Exception as e:
        print(f"Error processing general model {bias_model_name}: {str(e)}")
        print("Skipping this model")
        df[f'{bias_model_name}_present'] = 0
        df[f'{bias_model_name}_confidence'] = 0.0

# Define specialized models for different bias types
bias_type_models = {
    'abusive_language': "tomh/toxigen_hatebert",
    'hate_speech': "tomh/toxigen_roberta",
    'political_bias': "bucketresearch/politicalBiasBERT",
    'gender_bias': "monologg/koelectra-base-v3-gender-bias"
}

# Custom threshold for each model
bias_thresholds = {
    'abusive_language': 0.9,
    'hate_speech': 0.9,
    'political_bias': 0.9,
    'gender_bias': 0.9
}

# Special handling for problematic models that need specific tokenizers
model_specific_tokenizers = {
    "tomh/toxigen_hatebert": "bert-base-uncased"
}

# Process each bias type with appropriate model
for bias_type, model_name in bias_type_models.items():
    print(f"Processing {bias_type} using {model_name}...")
    
    try:
        # Check if this model needs a specific tokenizer
        if model_name in model_specific_tokenizers:
            tokenizer_name = model_specific_tokenizers[model_name]
            print(f"Using specific tokenizer {tokenizer_name} for model {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Normal loading for other models
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if tokenizer has pad_token, if not, set it
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
        
        # Get predictions
        bias_type_scores = get_predictions_in_batches(df['answer'].tolist(), tokenizer, model)
        
        # Use appropriate column for prediction
        prediction_column = 1 if bias_type_scores.shape[1] > 1 else 0
        
        # Assign binary label based on threshold
        df[bias_type] = (bias_type_scores[:, prediction_column] > bias_thresholds[bias_type]).astype(int)
        df[f'{bias_type}_confidence'] = bias_type_scores[:, prediction_column]
        
    except Exception as e:
        print(f"Error processing {bias_type}: {str(e)}")
        print("Using fallback approach for this bias type")
        # Set default values for this bias type
        df[bias_type] = 0
        df[f'{bias_type}_confidence'] = 0.0

# Add a summary column that identifies any type of bias detected
bias_columns = [col for col in df.columns if col.endswith('_present') or col in bias_type_models.keys()]
df['any_bias_detected'] = df[bias_columns].max(axis=1)

# Add timestamp to output for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = repo_root / f"data/processed/bias_labeled_dataset_{timestamp}.csv"

# --- Save the labeled dataset ---
df.to_csv(output_path, index=False)
print(f"✅ Labeled dataset saved to: {output_path}")

# Print a summary of bias detection
print("\n=== BIAS DETECTION SUMMARY ===")
for col in bias_columns:
    bias_count = df[col].sum()
    bias_percent = (bias_count / len(df)) * 100
    print(f"{col}: {bias_count} instances ({bias_percent:.2f}%)")

print(f"Any bias detected: {df['any_bias_detected'].sum()} instances ({(df['any_bias_detected'].sum() / len(df)) * 100:.2f}%)")