import os
import pandas as pd
import logging
from urllib.parse import urlparse
from supabase import create_client, Client
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import secrets

# Load environment variables from the .env file
load_dotenv()

# Initialize Supabase client using environment variables for the URL and key
def init_supabase():
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY") # Supabase API key from environment
    if not url or not key:
        raise ValueError("Supabase URL and API key must be set in the environment variables.")
    
    supabase: Client = create_client(url, key)

    return supabase

# Extract the filepath from the expired URL
def extract_filepath(audio_url):
    parsed_url = urlparse(audio_url)
    return parsed_url.path.replace("/storage/v1/object/sign/audios/processed/", "").split("?")[0]

# Generate a new signed URL for a given filepath
def generate_signed_url(supabase: Client, bucket_name: str, filepath: str, expiry_duration: int = 3600):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = supabase.storage.from_(bucket_name).create_signed_url(filepath, expiry_duration)
            if 'signedURL' in response:
                return response['signedURL']
            else:
                raise ValueError(f"Failed to generate signed URL for {filepath}")
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(secrets.randbelow(3) + 1)  # Secure random backoff between 1-3 seconds
                continue
            else:
                raise

# Process the dataset and generate new signed URLs
def update_presigned_urls(data: pd.DataFrame, supabase: Client, bucket_name: str, expiry_duration: int = 604800):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _, row in data.iterrows():
            future = executor.submit(process_row, row, supabase, bucket_name, expiry_duration)
            futures.append(future)

        new_urls = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URLs"):
            new_urls.append(future.result())

    data['new_audio_url'] = new_urls
    return data

def process_row(row, supabase: Client, bucket_name: str, expiry_duration: int = 3600):
    audio_url = row['audio_url']
    filepath = extract_filepath(audio_url)
    try:
        new_signed_url = generate_signed_url(supabase, bucket_name, filepath, expiry_duration)
        return new_signed_url
    except Exception as e:
        logging.error(f"Error generating signed URL for {filepath}: {e}")
        return None

# Main function
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load dataset
    input_file = "sampledata.csv"
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        return

    # Initialize Supabase client
    try:
        supabase = init_supabase()
    except ValueError as e:
        logging.error(f"Error initializing Supabase client: {e}")
        return

    # Supabase bucket name
    bucket_name = "audios/processed"

    # Update dataset with new signed URLs
    try:
        updated_data = update_presigned_urls(data, supabase, bucket_name, expiry_duration=3600)
    except Exception as e:
        logging.error(f"Error updating presigned URLs: {e}")
        return

    # Save the updated dataset
    output_file = "updated_dataset_with_presigned_urls.csv"
    try:
        updated_data.to_csv(output_file, index=False)
        logging.info(f"Updated dataset saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving updated dataset: {e}")

if __name__ == "__main__":
    main()
