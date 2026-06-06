# 🎮 Video Game Products Debater (Data Pipeline & AI Engine)

This repository contains the backend data engineering, sentiment analysis, and vector embedding pipeline for the **Video Game Products Debate Arena**, a production-grade AI application.

🔗 **[Live Application on Hugging Face Spaces](https://huggingface.co/spaces/RahulGoyal2111/video-game-products-debater)**

## 🧠 Project Architecture

This project moves beyond standard LLM wrappers by implementing a custom, rigorous Retrieval-Augmented Generation (RAG) architecture tailored for hardware reviews:

* **Data Ingestion:** Processing massive `.parquet` datasets containing consumer reviews and product metadata.
* **Vector Indexing:** Utilizing `BAAI/bge-m3` to map qualitative user reviews into a high-dimensional FAISS index.
* **Quantitative Sentiment:** Pre-computing RoBERTa-based sentiment scores across specific product taxonomy aspects (e.g., "Stick Drift", "Battery Life").
* **Temporal Decay Algorithm:** Applying a 5% exponential time-decay penalty to older reviews to ensure the LLM prioritizes the most current state of the hardware.
* **Multi-LLM Fallback Engine:** A highly resilient routing script that cascades from `llama-3.3-70b-versatile` down to `openai/gpt-oss-120b` and `qwen/qwen3-32b` to gracefully handle API rate limits without dropping user queries.

## 📁 Repository Structure

* `ingest_data.py` - A ingestion script streaming the massive raw Amazon Reviews dataset (JSONL) directly from the Hugging Face CDN into local storage.
* `/notebooks/` - Contains the Jupyter notebooks for raw data processing, RoBERTa sentiment classification, and FAISS index generation.
* `app_huggingface.py` - The production Streamlit interface configured for Hugging Face deployment.

## 🚀 Note on Data Storage

To maintain a lightweight repository, the massive `.parquet` datasets and the `.index` FAISS files are **not** hosted on GitHub. They are stored natively in a Hugging Face Dataset repository, where the live Streamlit app dynamically streams them into memory upon initialization.

## 📊 Dataset Attribution

The raw data processed in this pipeline is sourced from the **Amazon Reviews 2023** dataset, provided by the McAuley Lab. 

* **Documentation & Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
* **Citation:** This project utilizes the "Video Games" sub-category metadata and user review datasets. Full credit for the data collection and curation belongs to the original authors. 