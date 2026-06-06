import os
import requests
from tqdm import tqdm

def download_amazon_data_direct():
    data_dir = "/home/rahul_goyal/product-debater-video-games/data"
    
    # Direct CDN links bypassing the Hugging Face datasets library
    files_to_download = {
        "video_games_reviews.jsonl": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Video_Games.jsonl",
        "video_games_meta.jsonl": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl"
    }

    for filename, url in files_to_download.items():
        filepath = os.path.join(data_dir, filename)
        print(f"\nConnecting to Hugging Face CDN for {filename}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the total file size from the headers for the progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            # Download in 1MB chunks to keep memory usage low
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
                    
    print("\nData extraction complete. Files are physically located in your /data folder.")

if __name__ == "__main__":
    download_amazon_data_direct()