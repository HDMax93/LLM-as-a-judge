import pandas as pd
import numpy as np
import re
import string
import nltk
from pathlib import Path
from textblob import TextBlob
from nltk import pos_tag, word_tokenize, sent_tokenize, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

# Ensure necessary NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("wordnet")
nltk.download("stopwords")

# Paths
repo_root = Path(__file__).resolve().parents[2]
input_path = repo_root / "data/processed/balanced_prompts_answers.csv"
output_path = repo_root / "data/processed/df_features.csv"
tfidf_output_path = repo_root / "data/processed/tfidf_features.csv"
embeddings_output_path = repo_root / "data/processed/embeddings_features.npy"

# Load dataset
SAMPLE_MODE = True
SAMPLE_SIZE = 1000

df = pd.read_csv(input_path)
if SAMPLE_MODE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)

# --- Cleaning ---

# Drop null or invalid rows
df.dropna(inplace=True)
df = df[~df.isin(["", "N/A"]).any(axis=1)]

# Binary column for specific phrases
df["contains_apology"] = df["answer"].str.contains("I'm sorry|I don't know", case=False, na=False).astype(int)

# Convert column types
df["bias_category"] = df["bias_category"].astype("category")
df["bias_type"] = df["bias_type"].astype("category")
df["bias_subtype"] = df["bias_subtype"].astype("category")
df["prompt"] = df["prompt"].astype(str)
df["answer"] = df["answer"].astype(str)

# Set up stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

# Clean and tokenize functions
def clean_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word not in stop_words and word not in punctuation]

def extra_clean_text(text):
    text = re.sub(r"[-']", " ", text.lower())
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and word not in punctuation]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]

# Lowercase
df['prompt_lc'] = df['prompt'].str.lower()
df['answer_lc'] = df['answer'].str.lower()

# Token cleaning
df['prompt_cleaned_tokens'] = df['prompt_lc'].apply(clean_text)
df['prompt_extra_cleaned_tokens'] = df['prompt_lc'].apply(extra_clean_text)
df['answer_cleaned_tokens'] = df['answer_lc'].apply(clean_text)
df['answer_extra_cleaned_tokens'] = df['answer_lc'].apply(extra_clean_text)

# Lemmatization
df['prompt_clean_lemmatized_tokens'] = df['prompt_cleaned_tokens'].apply(lemmatize_tokens)
df['prompt_extra_clean_lemmatized_tokens'] = df['prompt_extra_cleaned_tokens'].apply(lemmatize_tokens)
df['answer_clean_lemmatized_tokens'] = df['answer_cleaned_tokens'].apply(lemmatize_tokens)
df['answer_extra_clean_lemmatized_tokens'] = df['answer_extra_cleaned_tokens'].apply(lemmatize_tokens)
df = df[df['answer_extra_clean_lemmatized_tokens'].apply(lambda x: len(x) > 0)]

# --- Feature Engineering ---

# Length-based features
df["answer_word_count"] = df["answer"].apply(lambda x: len(word_tokenize(str(x))))
df["answer_sentence_count"] = df["answer"].apply(lambda x: len(sent_tokenize(str(x))))
df["answer_avg_sentence_length"] = df.apply(
    lambda row: row["answer_word_count"] / row["answer_sentence_count"]
    if row["answer_sentence_count"] > 0 else 0, axis=1
)

# Sentiment features
df["answer_polarity"] = df["answer"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df["answer_subjectivity"] = df["answer"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Named Entity Recognition (count of named entities)
def count_named_entities(text):
    try:
        chunks = ne_chunk(pos_tag(word_tokenize(text)))
        named_entities = [chunk for chunk in chunks if isinstance(chunk, Tree)]
        return len(named_entities)
    except Exception:
        return 0

df["answer_named_entities"] = df["answer"].apply(count_named_entities)

# POS tagging ratios
def get_pos_ratios(text):
    try:
        tagged = pos_tag(word_tokenize(text))
        total = len(tagged)
        if total == 0:
            return (0, 0, 0, 0)
        counts = {"JJ": 0, "RB": 0, "NN": 0, "VB": 0}
        for word, tag in tagged:
            if tag.startswith("JJ"): counts["JJ"] += 1
            elif tag.startswith("RB"): counts["RB"] += 1
            elif tag.startswith("NN"): counts["NN"] += 1
            elif tag.startswith("VB"): counts["VB"] += 1
        return tuple(counts[pos] / total for pos in ["JJ", "RB", "NN", "VB"])
    except:
        return (0, 0, 0, 0)

df[["adj_ratio", "adv_ratio", "noun_ratio", "verb_ratio"]] = df["answer"].apply(get_pos_ratios).apply(pd.Series)

# Bias indicator (keyword-based) - UPDATED to use lemmatized tokens
# Lemmatized bias keywords (adjusted from original list)
lemmatized_bias_keywords = set([
    "should", "must", "deserve", "obligate", "entitle", "ought", "need", "have", 
    "typically", "usually", "often", "generally", "most", "always", "case", 
    "rule", "average", "everyone", "every", "never", "all", "naturally", 
    "innately", "nature", "because", "due", "result"
])

# Updated bias_features to work with lemmatized tokens
def bias_features_lemmatized(tokens):
    match_count = sum(token in lemmatized_bias_keywords for token in tokens)
    return pd.Series([1 if match_count > 0 else 0, match_count])

# Apply the updated bias detection to lemmatized tokens
df[["bias_indicator", "bias_term_count"]] = df["answer_clean_lemmatized_tokens"].apply(bias_features_lemmatized)

# Prompt complexity using Flesch Reading Ease
df["prompt_complexity"] = df["prompt"].apply(lambda x: flesch_reading_ease(str(x)))

# --- NEW CODE: TF-IDF for Top 20 Terms ---
print("Calculating TF-IDF features for top 20 terms...")

# Convert the lemmatized token lists to strings for TF-IDF
df['answer_lemma_strings'] = df['answer_extra_clean_lemmatized_tokens'].apply(lambda x: ' '.join(x))

# Create and fit TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['answer_lemma_strings'])

# Get feature names (top 20 terms)
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"Top 20 TF-IDF terms: {feature_names}")

# Convert TF-IDF sparse matrix to DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{term}' for term in feature_names]
)

# Save TF-IDF features separately
tfidf_df.to_csv(tfidf_output_path, index=False)
print(f"✅ TF-IDF features saved to: {tfidf_output_path}")

# Add TF-IDF features to the main dataframe
for column in tfidf_df.columns:
    df[column] = tfidf_df[column].values
print(f"✅ Added {len(tfidf_df.columns)} TF-IDF features to main dataframe")

# --- NEW CODE: Embedding Model ---
print("Generating embeddings...")

# Initialize the sentence transformer model
# Using mpnet-base-v2 for a good balance of quality and performance
model_name = 'all-mpnet-base-v2'
embedding_model = SentenceTransformer(model_name)

# Function to generate embeddings in batches to avoid memory issues
def generate_embeddings_in_batches(texts, model, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Show progress
        print(f"Processing embeddings batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

# Generate embeddings for answers
answer_embeddings = generate_embeddings_in_batches(df['answer'].tolist(), embedding_model)

# Save embeddings to disk
np.save(embeddings_output_path, answer_embeddings)
print(f"✅ Embeddings saved to: {embeddings_output_path}, shape: {answer_embeddings.shape}")

# Add embedding dimension info to the dataframe
df['embedding_dimensions'] = embedding_model.get_sentence_embedding_dimension()

# Save the main dataframe with all features to file
df.to_csv(output_path, index=False)
print(f"✅ Data cleaned and features engineered. Saved to: {output_path}")

print("✅ Processing complete!")