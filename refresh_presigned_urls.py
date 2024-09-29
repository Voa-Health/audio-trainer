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
import argparse

# Load environment variables from the .env file
load_dotenv()

# Initialize Supabase client using environment variables for the URL and key
def init_supabase():
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")  # Supabase API key from environment
    if not url or not key:
        raise ValueError("Supabase URL and API key must be set in the environment variables.")

    supabase: Client = create_client(url, key)
    return supabase

# Extract the filepath from the expired URL
def extract_filepath(audio_url):
    parsed_url = urlparse(audio_url)
    return parsed_url.path.replace("/storage/v1/object/sign/audios/processed/", "").split("?")[0]

# Generate a new signed URL for a given filepath
# Generate a new signed URL for a given filepath
def generate_signed_url(supabase: Client, bucket_name: str, filepath: str, expiry_duration: int = 3600):
    max_retries = 5

    for attempt in range(max_retries):
        try:
            # Initialize response before making the request
            response = None

            # Attempt to create the signed URL
            response = supabase.storage.from_(bucket_name).create_signed_url(filepath, expiry_duration)

            # Check if the 'signedURL' key exists in the response
            if response and isinstance(response, dict) and 'signedURL' in response:
                return response['signedURL']
            else:
                # Log detailed response info in case of failure to retrieve 'signedURL'
                logging.error(f"Unexpected response format or missing 'signedURL' key for {filepath}. Response: {response}")
                raise ValueError(f"Failed to generate signed URL for {filepath}")

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {filepath}: {e}, Response: {response}")

            # If not the last attempt, retry with backoff
            if attempt < max_retries - 1:
                time.sleep(secrets.randbelow(3) + 1)  # Secure random backoff between 1-3 seconds
            else:
                # Raise the final exception after all retries
                raise ValueError(f"Failed to generate signed URL for {filepath} after {max_retries} retries") from None

# Process the dataset and generate new signed URLs
def update_presigned_urls(data: pd.DataFrame, supabase: Client, bucket_name: str, expiry_duration: int = 604800):
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for _, row in data.iterrows():
            future = executor.submit(process_row, row, supabase, bucket_name, expiry_duration)
            futures.append(future)

        new_urls = []
        failed_rows = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URLs"):
            result = future.result()
            if result is None:
                failed_rows.append(row)
            new_urls.append(result)

    data['new_audio_url'] = new_urls

    # Log the rows that failed to generate URLs
    if failed_rows:
        logging.error(f"Failed to generate signed URLs for {len(failed_rows)} rows.")
        pd.DataFrame(failed_rows).to_csv("failed_rows.csv", index=False)
        logging.info("Failed rows saved to failed_rows.csv")

    return data

def process_row(row, supabase: Client, bucket_name: str, expiry_duration: int = 3600):
    audio_url = row['audio_url']
    filepath = extract_filepath(audio_url)
    try:
        new_signed_url = generate_signed_url(supabase, bucket_name, filepath, expiry_duration)
        return new_signed_url
    except Exception as e:
        logging.error(f"Error generating signed URL for {filepath} after retries: {e}")
        return None  # Or return a fallback URL if needed

# Main function
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Update signed URLs in a CSV file")
    parser.add_argument('input_csv', type=str, help="Path to the input CSV file")
    parser.add_argument('--output_csv', type=str, default="updated_dataset_with_presigned_urls.csv",
                        help="Path to save the updated CSV file (default: updated_dataset_with_presigned_urls.csv)")
    args = parser.parse_args()

    input_file = args.input_csv
    output_file = args.output_csv

    # Load dataset
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
    try:
        updated_data.to_csv(output_file, index=False)
        logging.info(f"Updated dataset saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving updated dataset: {e}")

if __name__ == "__main__":
    main()
