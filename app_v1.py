import streamlit as st
import os
import pandas as pd
import numpy as np
import re
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import torch
from dotenv import load_dotenv

# Set page config first
st.set_page_config(page_title="Hardware Debate Arena", layout="wide", page_icon="🎮")

load_dotenv()

# --- CACHING THE HEAVY ASSETS ---
# This guarantees your RTX 4060 only loads the model into VRAM once.
@st.cache_resource
def load_engine():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16})
    index = faiss.read_index("data/video_games_bge.index")
    return client, model, index

@st.cache_data
def load_data():
    df_products = pd.read_parquet("data/meta_clean.parquet")
    df_reviews = pd.read_parquet("data/semantic_metadata.parquet")
    df_stats = pd.read_parquet("data/aspect_sentiments.parquet")
    return df_products, df_reviews, df_stats

client, model, index = load_engine()
df_products, df_reviews, df_stats = load_data()

df_reviews['year'] = pd.to_datetime(df_reviews['timestamp']).dt.year

# --- TAXONOMY ---
category_topics = {
    'Games': ["Story and Narrative", "Gameplay Mechanics", "Graphics and Visuals", "Bugs and Performance", "Replayability", "Multiplayer and Servers"],
    'Headsets': ["Sound and Audio Quality", "Microphone Quality", "Comfort and Earcups", "Battery Life", "Build Quality", "Connectivity and Lag"],
    'Gaming Mice': ["Ergonomics and Shape", "Build Quality", "Software and Drivers", "Switch and Click Quality", "Sensor Tracking", "Scroll Wheel","Battery Life"],
    'Gaming Keyboards': ["Typing and Switch Feel", "Build Quality", "RGB Lighting", "Software and Drivers", "Wrist Rest and Ergonomics", "Wireless Connectivity"],
    'Controllers': ["Stick Drift and Thumbsticks", "Ergonomics and Grip", "Button and Trigger Feel", "Battery Life", "Build Quality", "D-Pad Quality"],
    'Consoles': ["Performance and Framerate", "Cooling and Fan Noise", "UI and Dashboard", "Storage Capacity", "Controller Quality", "Build Quality"]
}

# --- CORE EXTRACTION ENGINE ---
def extract_evidence(asin, target_topic, max_chunks=20, is_comparison=True):
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
    
    df_temporal['years_old'] = (2023 - df_temporal['year']).clip(lower=0)
    df_temporal['weighted_score'] = df_temporal['raw_score'] * (0.95 ** df_temporal['years_old'])
    final_local_idx = df_temporal.sort_values(by='weighted_score', ascending=False).head(max_chunks)['local_idx'].values
    
    best_chunks = []
    for local_idx in final_local_idx:
        text = df_reviews.iloc[product_indices[local_idx]]['text']
        raw_sentences = [s.strip() + "." for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        if not raw_sentences: continue
        
        chunks = []
        if len(raw_sentences) > 1:
            for j in range(len(raw_sentences)-1):
                chunks.append(" ".join(raw_sentences[j:j+2]))
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

# --- UI LAYOUT ---
st.title("🎮 Product Debater & Audit AI")
st.markdown("Analyze raw hardware sentiment powered by local BGE-M3 embeddings and Groq Llama-3.3-70B.")

# Sidebar Navigation
mode = st.sidebar.radio("Select Engine Mode:", ["Head-to-Head Debate", "Single Product Audit"])

# Category Selection
valid_categories = list(category_topics.keys())
selected_category = st.selectbox("1. Select Hardware Category", valid_categories)

# Filter products based on selected category
category_products = df_products[df_products['leaf_category'] == selected_category]
product_list = category_products['title'].tolist()
product_map = dict(zip(category_products['title'], category_products['parent_asin']))

# Topic Selection
selected_topic = st.selectbox("2. Select Target Topic", category_topics[selected_category])

st.divider()

if mode == "Head-to-Head Debate":
    col1, col2 = st.columns(2)
    with col1:
        title_a = st.selectbox("Contender A", product_list, index=0)
    with col2:
        title_b = st.selectbox("Contender B", product_list, index=1 if len(product_list) > 1 else 0)
        
    if st.button("Initiate Debate", type="primary"):
        if title_a == title_b:
            st.error("You cannot debate a product against itself.")
        else:
            asin_a, asin_b = product_map[title_a], product_map[title_b]
            
            with st.spinner("Extracting time-weighted vectors & pinging Groq..."):
                ev_a = extract_evidence(asin_a, selected_topic)
                ev_b = extract_evidence(asin_b, selected_topic)
                
                stats_a = df_stats[(df_stats['parent_asin'] == asin_a) & (df_stats['aspect'] == selected_topic)].iloc[0] if not df_stats[(df_stats['parent_asin'] == asin_a) & (df_stats['aspect'] == selected_topic)].empty else {'positive': 0, 'neutral': 0, 'negative': 0}
                stats_b = df_stats[(df_stats['parent_asin'] == asin_b) & (df_stats['aspect'] == selected_topic)].iloc[0] if not df_stats[(df_stats['parent_asin'] == asin_b) & (df_stats['aspect'] == selected_topic)].empty else {'positive': 0, 'neutral': 0, 'negative': 0}

                sys_prompt = """You are a ruthless, objective video game hardware and software analyst. 
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
                (Provide a balanced, data-justified breakdown of both the products regarding the specific topic of Debate only, nothing else. If a product's numbers are overwhelmingly positive or very positive, highlight its core strengths(regarding the Topic of Debate)
                first before mentioning any minor limitations (regarding the Topic of Debate). If its numbers are poor, focus heavily on its major failures (regarding the Topic of Debate). Do not invent flaws of your own)."""


                usr_prompt = f"""TOPIC OF DEBATE: {selected_topic}
                === {title_a} ===
                Sentiment: {stats_a['positive']} Pos | {stats_a['neutral']} Neu | {stats_a['negative']} Neg
                Evidence: {chr(10).join([f"- {e}" for e in ev_a])}
                
                === {title_b} ===
                Sentiment: {stats_b['positive']} Pos | {stats_b['neutral']} Neu | {stats_b['negative']} Neg
                Evidence: {chr(10).join([f"- {e}" for e in ev_b])}"""

                response = client.chat.completions.create(
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                )
                
                st.success("Analysis Complete.")
                st.markdown(response.choices[0].message.content)

elif mode == "Single Product Audit":
    title_single = st.selectbox("Select Product to Audit", product_list)
    
    if st.button("Run Full Audit", type="primary"):
        asin_single = product_map[title_single]
        
        with st.spinner("Compiling dossier & pinging Groq..."):
            ev_single = extract_evidence(asin_single, selected_topic)
            stats_s = df_stats[(df_stats['parent_asin'] == asin_single) & (df_stats['aspect'] == selected_topic)].iloc[0] if not df_stats[(df_stats['parent_asin'] == asin_single) & (df_stats['aspect'] == selected_topic)].empty else {'positive': 0, 'neutral': 0, 'negative': 0}
            
            sys_prompt = """You are a ruthless, objective video game hardware and software analyst. 
            Your job is to read the quantitative sentiment data and qualitative user evidence for a specific product regarding a SPECIFIC TOPIC, and deliver an unvarnished performance report.

            CRITICAL DIRECTIVES:
            1. FOCUS ONLY ON THE PROVIDED TOPIC of Analysis. Completely ignore any user comments about other aspects of the product.
            2. Use the actual title of the product in your response. Do not use generic placeholders.
            3. MATCH THE TONE TO THE NUMBERS. If the sentiment distribution is overwhelmingly positive, emphasize its strengths and contextualize complaints as isolated edge cases. If it is poor, treat user frustrations as widespread systemic issues.

            Format your response exactly with these markdown headers:
            ## The Numbers
            (State the exact Positive/Neutral/Negative sentiment distributions provided for this topic).

            ## The Evidence
            (Cite specific quotes from the provided user snippets that directly validate the topic numbers).

            ## The Real Takeaway
            (Provide a balanced, data-justified breakdown of the real-world performance for this product regarding the Topic of Analysis. Realistically summarize its core capabilities and limitations based solely on the provided evidence)."""

            usr_prompt = f"""TOPIC of Analysis: {selected_topic}
            === {title_single} ===
            Sentiment: {stats_s['positive']} Pos | {stats_s['neutral']} Neu | {stats_s['negative']} Neg
            Evidence: {chr(10).join([f"- {e}" for e in ev_single])}"""

            response = client.chat.completions.create(
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
            
            st.success("Audit Complete.")
            st.markdown(response.choices[0].message.content)