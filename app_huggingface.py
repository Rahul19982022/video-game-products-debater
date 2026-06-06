import streamlit as st
import os
import pandas as pd
import numpy as np
import re
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import torch
import groq
import urllib.request  # Built-in utility to download the FAISS index file

st.set_page_config(page_title="Video Game Products Debate Arena", layout="wide", page_icon="🎮")

# --- HUGGING FACE ASSETS CONFIGURATION ---
# Replace this with your actual Hugging Face username and dataset name
HF_USER = "RahulGoyal2111"
HF_DATASET = "product-debater-assets"

# Construct direct download URLs for your parquet files
META_URL = f"https://huggingface.co/datasets/{HF_USER}/{HF_DATASET}/resolve/main/meta_clean.parquet"
REVIEWS_URL = f"https://huggingface.co/datasets/{HF_USER}/{HF_DATASET}/resolve/main/semantic_metadata.parquet"
STATS_URL = f"https://huggingface.co/datasets/{HF_USER}/{HF_DATASET}/resolve/main/aspect_sentiments.parquet"
INDEX_URL = f"https://huggingface.co/datasets/{HF_USER}/{HF_DATASET}/resolve/main/video_games_bge.index"

@st.cache_resource
def load_engine():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    # Load model on CPU for Hugging Face free tier compatibility
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    
    # FAISS indices cannot be read directly from a URL string, so we download it to a temporary local path
    local_index_path = "video_games_bge.index"
    if not os.path.exists(local_index_path):
        with st.spinner("Downloading FAISS Index from Hugging Face Hub..."):
            urllib.request.urlretrieve(INDEX_URL, local_index_path)
            
    index = faiss.read_index(local_index_path)
    return client, model, index

@st.cache_data
def load_data():
    with st.spinner("Streaming analytical dataframes from Hugging Face Hub..."):
        df_products = pd.read_parquet(META_URL)
        df_reviews = pd.read_parquet(REVIEWS_URL)
        df_stats = pd.read_parquet(STATS_URL)
    return df_products, df_reviews, df_stats

client, model, index = load_engine()
df_products, df_reviews, df_stats = load_data()

df_reviews['year'] = pd.to_datetime(df_reviews['timestamp']).dt.year

# --- TAXONOMY ---
category_topics = {
    'Games': ["Story and Narrative", "Gameplay Mechanics", "Graphics and Visuals", "Bugs and Performance", "Replayability", "Multiplayer and Servers"],
    'Headsets': ["Sound and Audio Quality", "Build Quality", "Microphone Quality", "Comfort and Earcups", "Battery Life", "Connectivity and Lag"],
    'Gaming Mice': ["Ergonomics and Shape", "Build Quality", "Software and Drivers", "Switch and Click Quality", "Sensor Tracking", "Scroll Wheel","Battery Life"],
    'Gaming Keyboards': ["Typing and Switch Feel", "Build Quality", "Software and Drivers", "Wrist Rest and Ergonomics", "RGB Lighting", "Wireless Connectivity"],
    'Controllers': ["Button and Trigger Feel", "Ergonomics and Grip", "Build Quality", "Battery Life", "Stick Drift and Thumbsticks", "D-Pad Quality"],
    'Consoles': ["Performance and Framerate", "Build Quality", "UI and Dashboard", "Storage Capacity", "Controller Quality", "Cooling and Fan Noise"]
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


# --- FALLBACK ENGINE ---
# --- FALLBACK ENGINE ---
def generate_verdict_with_fallback(system_prompt, user_prompt):
    # The cascading defense strategy: ordered by reasoning capability and TPD limits
    models_to_try = [
        "llama-3.3-70b-versatile",                   # Primary Flagship (100k TPD)
        "qwen/qwen3-32b",                            # Tactical Mid-Weight (500k TPD)
        "openai/gpt-oss-120b",                       # Heavyweight Backup (200k TPD)
        "meta-llama/llama-4-scout-17b-16e-instruct", # Context Scout (500k TPD)
        "llama-3.1-8b-instant"                       # Workhorse Safety Net (500k TPD)
    ]
    
    for i, model_name in enumerate(models_to_try):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=model_name,
                temperature=0.2,
            )
            
            # If we had to drop down a tier, trigger a silent toast notification
            if i > 0:
                st.toast(f"⚠️ Primary AI maxed out. Switched to: {model_name}", icon="🔄")
                
            raw_text = response.choices[0].message.content
            
            # Scrub <think> blocks generated by reasoning models like Qwen
            cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            
            return cleaned_text
            
        except groq.RateLimitError:
            # If the absolute last model on the list fails, kill the app cleanly
            if i == len(models_to_try) - 1:
                st.error("🚨 **Total Limit Reached!** All backup models have exhausted their free tokens for today. The arena is closed.", icon="🛑")
                st.stop()
            else:
                continue # Catch the error and immediately fire the next model in the list
                
        except Exception as e:
            st.error(f"An unexpected Groq API error occurred: {e}")
            st.stop()

# --- UI LAYOUT ---
st.title("🎮 Video Game Products Debate Arena")
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
        
    if st.button("⚡ Initiate Debate", type="primary", use_container_width=True):
        if title_a == title_b:
            st.error("You cannot debate a product against itself.")
        else:
            asin_a, asin_b = product_map[title_a], product_map[title_b]
            
            with st.spinner("🔍 Extracting time-weighted vectors & evaluating sentiment..."):
                ev_a = extract_evidence(asin_a, selected_topic)
                ev_b = extract_evidence(asin_b, selected_topic)
                
                stats_a = df_stats[(df_stats['parent_asin'] == asin_a) & (df_stats['aspect'] == selected_topic)].iloc[0] if not df_stats[(df_stats['parent_asin'] == asin_a) & (df_stats['aspect'] == selected_topic)].empty else {'positive': 0, 'neutral': 0, 'negative': 0}
                stats_b = df_stats[(df_stats['parent_asin'] == asin_b) & (df_stats['aspect'] == selected_topic)].iloc[0] if not df_stats[(df_stats['parent_asin'] == asin_b) & (df_stats['aspect'] == selected_topic)].empty else {'positive': 0, 'neutral': 0, 'negative': 0}

                # --- VISUAL IMPROVEMENT: QUANTITATIVE SCOREBOARD ---
                st.subheader("📊 Quantitative Baseline - Sentiment Scores")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    with st.container(border=True):
                        st.markdown(f"**{title_a[:50]}...**")
                        m_a1, m_a2, m_a3 = st.columns(3)
                        m_a1.metric("Positive", f"{stats_a['positive']}")
                        m_a2.metric("Neutral", f"{stats_a['neutral']}")
                        m_a3.metric("Negative", f"{stats_a['negative']}")
                        
                with metric_col2:
                    with st.container(border=True):
                        st.markdown(f"**{title_b[:50]}...**")
                        m_b1, m_b2, m_b3 = st.columns(3)
                        m_b1.metric("Positive", f"{stats_b['positive']}")
                        m_b2.metric("Neutral", f"{stats_b['neutral']}")
                        m_b3.metric("Negative", f"{stats_b['negative']}")

                # Execution Prompts
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

                usr_prompt = f"""TOPIC: {selected_topic}
                === {title_a} ===
                Sentiment: {stats_a['positive']} Pos | {stats_a['neutral']} Neu | {stats_a['negative']} Neg
                Evidence: {chr(10).join([f"- {e}" for e in ev_a])}
                
                === {title_b} ===
                Sentiment: {stats_b['positive']} Pos | {stats_b['neutral']} Neu | {stats_b['negative']} Neg
                Evidence: {chr(10).join([f"- {e}" for e in ev_b])}"""

                # Use our new fallback engine
                final_text = generate_verdict_with_fallback(sys_prompt, usr_prompt)
                
                # --- VISUAL IMPROVEMENT: AI OUTPUT CONTAINER ---
                st.subheader("🤖 Analyst Deep-Dive Verdict")
                with st.container(border=True):
                    st.markdown(final_text)

elif mode == "Single Product Audit":
    title_single = st.selectbox("Select Product to Audit", product_list)
    
    if st.button("🕵️‍♂️ Run Full Audit", type="primary", use_container_width=True):
        asin_single = product_map[title_single]
        
        with st.spinner("🔍 Compiling dossier & parsing temporal chunks..."):
            ev_single = extract_evidence(asin_single, selected_topic)
            stats_s = df_stats[(df_stats['parent_asin'] == asin_single) & (df_stats['aspect'] == selected_topic)].iloc[0] if not df_stats[(df_stats['parent_asin'] == asin_single) & (df_stats['aspect'] == selected_topic)].empty else {'positive': 0, 'neutral': 0, 'negative': 0}
            
            # --- VISUAL IMPROVEMENT: SINGLE METRIC CARD ---
            st.subheader("📊 Sentiment Footprint")
            with st.container(border=True):
                st.markdown(f"**{title_single}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Positive Mentions", f"{stats_s['positive']}", delta=None)
                m2.metric("Neutral Mentions", f"{stats_s['neutral']}", delta=None)
                m3.metric("Negative Mentions", f"{stats_s['negative']}", delta=None, delta_color="inverse")

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
            (Provide a balanced, data-justified breakdown of the real-world performance for this product regarding the Topic of Analysis only. Realistically summarize its core capabilities (regarding the Topic of Analysis) and limitations (Regarding the Topic of Analysis) based solely on the provided evidence)."""

            usr_prompt = f"""TOPIC: {selected_topic}
            === {title_single} ===
            Sentiment: {stats_s['positive']} Pos | {stats_s['neutral']} Neu | {stats_s['negative']} Neg
            Evidence: {chr(10).join([f"- {e}" for e in ev_single])}"""

            # Use our new fallback engine
            final_text = generate_verdict_with_fallback(sys_prompt, usr_prompt)
            
            # --- VISUAL IMPROVEMENT: SINGLE AUDIT CONTAINER ---
            st.subheader("📋 Audit Report")
            with st.container(border=True):
                st.markdown(final_text)