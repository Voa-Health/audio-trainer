import os
import pandas as pd
import logging
from urllib.parse import urlparse
from supabase import create_client, Client
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import time
import secrets
import json

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

# Batch creation for processing in chunks
def batch(iterable, batch_size=1000):
    length = len(iterable)
    for i in range(0, length, batch_size):
        yield iterable[i:i + batch_size]

# Create signed URLs for a batch of filepaths with retries
def create_signed_urls_batch(supabase: Client, batch_items, expiry_duration: int, max_retries=5):
    for attempt in range(max_retries):
        try:
            # Send the request to create signed URLs for the batch of items
            response = supabase.storage.from_("audios").create_signed_urls(batch_items, expiry_duration)

            # Check if response is a list or a dictionary
            if isinstance(response, list):
                return [item["signedURL"] for item in response if "signedURL" in item]
            elif isinstance(response, dict) and "signedURLs" in response:
                return response["signedURLs"]
            else:
                logging.error(f"Unexpected response format: {response}")
                raise ValueError("Unexpected response format when creating signed URLs.")
        except Exception as e:
            logging.error(f"Error creating signed URLs for batch: {e}, Attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(secrets.randbelow(3) + 1)  # Retry with backoff
            else:
                raise  # If all retries fail, propagate the exception

# Process the dataset and generate new signed URLs in batches, saving intermediate results
def update_presigned_urls(data: pd.DataFrame, supabase: Client, expiry_duration: int, batch_size: int = 1000, temp_file='temp_results.jsonl', failed_file='failed_rows.csv'):
    audio_urls = []
    failed_rows = []

    chunks_id_list = data['audio_url'].tolist()
    processed_audio_id_list = ["processed/" + extract_filepath(audio_url) for audio_url in chunks_id_list]

    total_items = len(processed_audio_id_list)

    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        futures = []

        # Submit batch jobs
        with tqdm(total=total_items, desc="Submitting batch jobs") as pbar:
            for batch_items in batch(processed_audio_id_list, batch_size):
                future = executor.submit(create_signed_urls_batch, supabase, batch_items, expiry_duration)
                futures.append(future)
                pbar.update(len(batch_items))

        results = []

        # Collect results from the futures
        with tqdm(total=len(futures), desc="Processing results") as pbar:
            for future in as_completed(futures):
                try:
                    batch_urls = future.result()
                    results.append(batch_urls)
                    # Save intermediate results
                    save_intermediate_results(temp_file, batch_urls)
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                    failed_rows.append(batch_items)  # Append the batch that failed
                pbar.update(1)

        # Flatten results and add to audio_urls
        audio_urls = [url for batch_urls in results for url in batch_urls]

    # Log the rows that failed to generate URLs
    if failed_rows:
        logging.error(f"Failed to generate signed URLs for {len(failed_rows)} rows.")
        pd.DataFrame(failed_rows).to_csv(failed_file, index=False)
        logging.info(f"Failed rows saved to {failed_file}")

    # Save the final results
    save_intermediate_results(temp_file, audio_urls, is_final=True)

    # Append the new signed URLs to the original dataset
    if len(audio_urls) == len(data):
        data['new_audio_url'] = audio_urls
    else:
        logging.error(f"Length mismatch between input data and generated URLs: {len(audio_urls)} vs {len(data)}")
        # Handle the case when lengths don't match by saving a partial file
        data['new_audio_url'] = audio_urls[:len(data)]

    return data

# Save intermediate results to a file (JSONL format for simplicity)
def save_intermediate_results(temp_file, results, is_final=False):
    mode = 'w' if is_final else 'a'  # Overwrite on final save, append otherwise
    with open(temp_file, mode) as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    if is_final:
        logging.info(f"Final results saved to {temp_file}")
    else:
        logging.info(f"Intermediate results saved to {temp_file}")

# Main function
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Update signed URLs in a CSV file")
    parser.add_argument('input_csv', type=str, help="Path to the input CSV file")
    parser.add_argument('--output_csv', type=str, default="updated_dataset_with_presigned_urls.csv",
                        help="Path to save the updated CSV file (default: updated_dataset_with_presigned_urls.csv)")
    parser.add_argument('--temp_file', type=str, default="temp_results.jsonl", help="Path to save intermediate results")
    parser.add_argument('--failed_file', type=str, default="failed_rows.csv", help="Path to save failed rows")
    args = parser.parse_args()

    input_file = args.input_csv
    output_file = args.output_csv
    temp_file = args.temp_file
    failed_file = args.failed_file

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

    # Update dataset with new signed URLs in batches, with intermediate saving
    try:
        updated_data = update_presigned_urls(data, supabase, expiry_duration=604800, batch_size=1000, temp_file=temp_file, failed_file=failed_file)
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
