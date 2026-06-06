import os
import pandas as pd
import numpy as np
import re
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# 1. Initialize Clients and Local Models
print("Loading Engine...")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16})

df_products = pd.read_parquet("data/meta_clean.parquet")
df_reviews = pd.read_parquet("data/semantic_metadata.parquet")
df_stats = pd.read_parquet("data/aspect_sentiments.parquet")
index = faiss.read_index("data/video_games_bge.index")

df_reviews['year'] = pd.to_datetime(df_reviews['timestamp']).dt.year

# 2. The Universal Taxonomy
category_topics = {
    'Gaming Mice': ["Battery Life", "Ergonomics and Shape", "Build Quality", "Software and Drivers", "Switch and Click Quality", "Sensor Tracking", "Scroll Wheel"],
    'Gaming Keyboards': ["Typing and Switch Feel", "Build Quality", "RGB Lighting", "Software and Drivers", "Wrist Rest and Ergonomics", "Wireless Connectivity"],
    'Headsets': ["Sound and Audio Quality", "Microphone Quality", "Comfort and Earcups", "Battery Life", "Build Quality", "Connectivity and Lag"],
    'Controllers': ["Stick Drift and Thumbsticks", "Ergonomics and Grip", "Button and Trigger Feel", "Battery Life", "Build Quality", "D-Pad Quality"],
    'Consoles': ["Performance and Framerate", "Cooling and Fan Noise", "UI and Dashboard", "Storage Capacity", "Controller Quality", "Build Quality"],
    'Games': ["Story and Narrative", "Gameplay Mechanics", "Graphics and Visuals", "Bugs and Performance", "Replayability", "Multiplayer and Servers"]
}

# 3. Dynamic Product Selection
product_a_id = "B001UQ704C"  # Replace with any ASIN
product_b_id = "B00D7B9M0I"  # Replace with any competing ASIN

product_a_info = df_products[df_products['parent_asin'] == product_a_id].iloc[0]
product_b_info = df_products[df_products['parent_asin'] == product_b_id].iloc[0]

category = product_a_info['leaf_category']

# Safety Check: Prevent comparing a mouse to a video game
if category != product_b_info['leaf_category']:
    raise ValueError(f"Category mismatch! You are trying to compare a {category} to a {product_b_info['leaf_category']}.")

print(f"Arena Matched: {category} Category")
print(f"Contender A: {product_a_info['title']}")
print(f"Contender B: {product_b_info['title']}")

# 4. Select Topic from the valid list for this category
valid_topics = category_topics[category]
print(f"\nAvailable Topics for {category}: {valid_topics}")

# Let's dynamically pick the first one, or you can hardcode one like "Build Quality"
target_topic = valid_topics[1] 
print(f"\nDEBATING TOPIC: {target_topic}\n")

# 5. Extraction Function (Time-Weighted + 2-Sentence context + 50-word cap)
def extract_evidence(asin, target_topic, max_chunks=20):
    product_indices = np.where(df_reviews['parent_asin'] == asin)[0]
    if len(product_indices) == 0: return []
    
    product_vectors = np.array([index.reconstruct(int(i)) for i in product_indices])
    topic_vector = model.encode(target_topic, normalize_embeddings=True).reshape(1, -1)
    
    similarities = np.dot(topic_vector, product_vectors.T)[0]
    pool_size = min(100, len(similarities))
    top_pool_local_idx = np.argsort(similarities)[::-1][:pool_size]
    
    candidate_global_idx = product_indices[top_pool_local_idx]
    candidate_scores = similarities[top_pool_local_idx]
    candidate_years = df_reviews.iloc[candidate_global_idx]['year'].values
    
    df_temporal = pd.DataFrame({
        'local_idx': top_pool_local_idx,
        'raw_score': candidate_scores,
        'year': candidate_years
    })
    baseline_year = 2023
    df_temporal['years_old'] = (baseline_year - df_temporal['year']).clip(lower=0)
    df_temporal['weighted_score'] = df_temporal['raw_score'] * (0.95 ** df_temporal['years_old'])
    
    final_local_idx = df_temporal.sort_values(by='weighted_score', ascending=False).head(max_chunks)['local_idx'].values
    
    best_chunks = []
    for local_idx in final_local_idx:
        text = df_reviews.iloc[product_indices[local_idx]]['text']
        raw_sentences = [s.strip() + "." for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        if not raw_sentences: continue
        
        # Groq needs context: Strictly 2-sentence sliding window
        chunks = []
        if len(raw_sentences) > 1:
            for j in range(len(raw_sentences)-1):
                combined = " ".join(raw_sentences[j:j+2])
                chunks.append(combined)
        else:
            chunks = raw_sentences
            
        capped_chunks = []
        for c in chunks:
            words = c.split()
            if len(words) > 50:
                capped_chunks.append(" ".join(words[:50]) + "...")
            else:
                capped_chunks.append(c)
                
        chunk_vectors = model.encode(capped_chunks, normalize_embeddings=True, show_progress_bar=False)
        chunk_scores = np.dot(topic_vector, chunk_vectors.T)[0]
        best_chunks.append(capped_chunks[np.argmax(chunk_scores)])
        
    return best_chunks

# 6. Gather the Evidence
print(f"Extracting evidence...")
evidence_a = extract_evidence(product_a_id, target_topic)
evidence_b = extract_evidence(product_b_id, target_topic)

# Get pre-computed stats safely
stats_a_query = df_stats[(df_stats['parent_asin'] == product_a_id) & (df_stats['aspect'] == target_topic)]
stats_b_query = df_stats[(df_stats['parent_asin'] == product_b_id) & (df_stats['aspect'] == target_topic)]

stats_a = stats_a_query.iloc[0] if not stats_a_query.empty else {'positive': 0, 'neutral': 0, 'negative': 0}
stats_b = stats_b_query.iloc[0] if not stats_b_query.empty else {'positive': 0, 'neutral': 0, 'negative': 0}

# 7. The Prompt Architecture (Data-Driven Balanced Analysis)
system_prompt = """You are a ruthless, objective video game hardware and software analyst. 
Your job is to read the quantitative sentiment data and qualitative user evidence for two competing products regarding a SPECIFIC TOPIC, and deliver an unvarnished comparative verdict.

CRITICAL DIRECTIVES:
1. FOCUS ONLY ON THE PROVIDED DEBATE TOPIC. Completely ignore any user comments about other aspects of the products.
2. Use the actual titles of the products in your response. Never refer to them as "Game A", "Product B", etc.
3. MATCH THE TONE TO THE NUMBERS. If a product has high positive sentiment, emphasize its strengths and treat any complaints as minor edge cases. If a product has high negative sentiment, ruthlessly expose its flaws as widespread systemic issues.

Format your response exactly with these headers:
## The Verdict
(Declare the outcome regarding the specific topic in 1-2 sentences. Explicitly state if it is a clear win, a tie, or just the lesser of two evils, and why based strictly on this topic).

## The Numbers
(Compare the exact Positive/Neutral/Negative sentiment distributions provided for this topic).

## The Evidence
(Cite specific quotes from the provided user snippets that directly support the topic of debate).

## The Real Takeaway
(Provide a balanced, data-justified breakdown of the real-world performance for both products regarding the Topic of Debate. If a product's numbers are overwhelmingly positive or very positive, highlight its core strengths first before mentioning any minor limitations. If its numbers are poor, focus heavily on its major failures. Do not invent flaws of your own)."""

user_prompt = f"""
TOPIC OF DEBATE: {target_topic}

=== {product_a_info['title']} ===
Quantitative Sentiment: {stats_a['positive']} Positive | {stats_a['neutral']} Neutral | {stats_a['negative']} Negative
User Evidence Snippets:
{chr(10).join([f"- {e}" for e in evidence_a])}

=== {product_b_info['title']} ===
Quantitative Sentiment: {stats_b['positive']} Positive | {stats_b['neutral']} Neutral | {stats_b['negative']} Negative
User Evidence Snippets:
{chr(10).join([f"- {e}" for e in evidence_b])}

Analyze the data and deliver your topic-focused verdict using their actual titles.
"""

# 8. Groq Execution
print(f"Passing context to Groq for {product_a_info['title'][:30]}... vs {product_b_info['title'][:30]}...")
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
print("="*80)