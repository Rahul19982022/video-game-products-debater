import os
import pandas as pd
import numpy as np
import re
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import torch
from dotenv import load_dotenv

# 1. Load Environment Variables and Initialize Clients
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please ensure it is defined in your .env file.")

print("Loading Engine and Local Models...")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16})

# Load Datasets
df_products = pd.read_parquet("data/meta_clean.parquet")
df_reviews = pd.read_parquet("data/semantic_metadata.parquet")
df_stats = pd.read_parquet("data/aspect_sentiments.parquet")
index = faiss.read_index("data/video_games_bge.index")

df_reviews['year'] = pd.to_datetime(df_reviews['timestamp']).dt.year

# 2. Universal Category Taxonomy Mapping
category_topics = {
    'Gaming Mice': ["Battery Life", "Ergonomics and Shape", "Build Quality", "Software and Drivers", "Switch and Click Quality", "Sensor Tracking", "Scroll Wheel"],
    'Gaming Keyboards': ["Typing and Switch Feel", "Build Quality", "RGB Lighting", "Software and Drivers", "Wrist Rest and Ergonomics", "Wireless Connectivity"],
    'Headsets': ["Sound and Audio Quality", "Microphone Quality", "Comfort and Earcups", "Battery Life", "Build Quality", "Connectivity and Lag"],
    'Controllers': ["Stick Drift and Thumbsticks", "Ergonomics and Grip", "Button and Trigger Feel", "Battery Life", "Build Quality", "D-Pad Quality"],
    'Consoles': ["Performance and Framerate", "Cooling and Fan Noise", "UI and Dashboard", "Storage Capacity", "Controller Quality", "Build Quality"],
    'Games': ["Story and Narrative", "Gameplay Mechanics", "Graphics and Visuals", "Bugs and Performance", "Replayability", "Multiplayer and Servers"]
}

# 3. Target Selection
# Swap this out with any parent_asin from your meta_clean database
target_asin = "B00D7B9M0I"

product_info = df_products[df_products['parent_asin'] == target_asin].iloc[0]
category = product_info['leaf_category']

if category not in category_topics:
    raise ValueError(f"Category '{category}' is not supported by our current taxonomy.")

# Automatically pick a relevant topic for this analysis loop
valid_topics = category_topics[category]
target_topic = valid_topics[2]  # Dynamically loads the first topic (e.g., Battery Life)

print("\n" + "="*80)
print(f"📦 PRODUCT UNDER AUDIT: {product_info['title']}")
print(f"🗂️ CATEGORY: {category}")
print(f"🎯 ANALYSIS ASPECT: {target_topic}")
print("="*80 + "\n")

# 4. Evidence Extraction Function (Time-Weighted + 2-Sentence Sliding Window + 50-Word Cap)
def extract_single_product_evidence(asin, topic, max_chunks=20):
    product_indices = np.where(df_reviews['parent_asin'] == asin)[0]
    if len(product_indices) == 0: 
        return []
    
    # Reconstruct vectors from FAISS index for this product
    product_vectors = np.array([index.reconstruct(int(i)) for i in product_indices])
    topic_vector = model.encode(topic, normalize_embeddings=True).reshape(1, -1)
    
    # 1. Fetch raw similarity scores
    similarities = np.dot(topic_vector, product_vectors.T)[0]
    pool_size = min(100, len(similarities))
    top_pool_local_idx = np.argsort(similarities)[::-1][:pool_size]
    
    # 2. Extract candidate details for time weighting
    candidate_global_idx = product_indices[top_pool_local_idx]
    candidate_scores = similarities[top_pool_local_idx]
    candidate_years = df_reviews.iloc[candidate_global_idx]['year'].values
    
    # 3. Calculate Exponential Decay (5% penalty per year older than 2023)
    df_temporal = pd.DataFrame({
        'local_idx': top_pool_local_idx,
        'raw_score': candidate_scores,
        'year': candidate_years
    })
    baseline_year = 2023
    df_temporal['years_old'] = (baseline_year - df_temporal['year']).clip(lower=0)
    df_temporal['weighted_score'] = df_temporal['raw_score'] * (0.95 ** df_temporal['years_old'])
    
    # 4. Filter down to top time-weighted review blocks
    final_local_idx = df_temporal.sort_values(by='weighted_score', ascending=False).head(max_chunks)['local_idx'].values
    
    best_chunks = []
    for local_idx in final_local_idx:
        text = df_reviews.iloc[product_indices[local_idx]]['text']
        raw_sentences = [s.strip() + "." for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        if not raw_sentences: 
            continue
        
        # Build 2-sentence sliding windows to maintain pronoun context for the LLM
        chunks = []
        if len(raw_sentences) > 1:
            for j in range(len(raw_sentences)-1):
                combined = " ".join(raw_sentences[j:j+2])
                chunks.append(combined)
        else:
            chunks = raw_sentences
            
        # Hard token/word cap protection
        capped_chunks = []
        for c in chunks:
            words = c.split()
            if len(words) > 50:
                capped_chunks.append(" ".join(words[:50]) + "...")
            else:
                capped_chunks.append(c)
                
        # Sub-select the exact sentence pair that carries the highest match weight
        chunk_vectors = model.encode(capped_chunks, normalize_embeddings=True, show_progress_bar=False)
        chunk_scores = np.dot(topic_vector, chunk_vectors.T)[0]
        best_chunks.append(capped_chunks[np.argmax(chunk_scores)])
        
    return best_chunks

# 5. Gather Data Assets
print("Extracting time-weighted qualitative snippets...")
evidence = extract_single_product_evidence(target_asin, target_topic)

# Load pre-computed metrics safely
stats_query = df_stats[(df_stats['parent_asin'] == target_asin) & (df_stats['aspect'] == target_topic)]
if not stats_query.empty:
    stats = stats_query.iloc[0]
else:
    stats = {'positive': 0, 'neutral': 0, 'negative': 0}

# 6. Prompt Generation (Balanced Single-Target Variant)
system_prompt = """You are a ruthless, objective video game hardware and software analyst. 
Your job is to read the quantitative sentiment data and qualitative user evidence for a specific product regarding a SPECIFIC TOPIC, and deliver an unvarnished performance report.

CRITICAL DIRECTIVES:
1. FOCUS ONLY ON THE PROVIDED DEBATE TOPIC. Completely ignore any user comments about other aspects of the product.
2. Use the actual title of the product in your response. Do not use generic placeholders.
3. MATCH THE TONE TO THE NUMBERS. If the sentiment distribution is overwhelmingly positive, emphasize its strengths and contextualize complaints as isolated edge cases. If it is poor, treat user frustrations as widespread systemic issues.

Format your response exactly with these markdown headers:
## The Numbers
(State the exact Positive/Neutral/Negative sentiment distributions provided for this topic).

## The Evidence
(Cite specific quotes from the provided user snippets that directly validate the topic numbers).

## The Real Takeaway
(Provide a balanced, data-justified breakdown of the real-world performance for this product regarding the Topic of Debate. Realistically summarize its core capabilities and limitations based solely on the provided evidence)."""

user_prompt = f"""
TOPIC OF ANALYSIS: {target_topic}

=== {product_info['title']} ===
Quantitative Sentiment: {stats['positive']} Positive | {stats['neutral']} Neutral | {stats['negative']} Negative

User Evidence Snippets:
{chr(10).join([f"- {e}" for e in evidence])}

Analyze the data and deliver your topic-focused report using the product title.
"""

# 7. Groq Evaluation Pipeline Execution
print(f"Pinging Groq Llama-3.3-70B for topic extraction...")
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

print("\n" + "="*80)
print(chat_completion.choices[0].message.content)
print("="*80 + "\n")