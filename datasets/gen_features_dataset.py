import os
import time
import requests
import pandas as pd
import torchaudio
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import logging
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import json
from tqdm import tqdm

# Configuration
target_sample_rate = 16000
max_duration_sec = 30  # Maximum duration of the audio in seconds
max_workers = 4  # Number of parallel workers (adjust based on your system)
do_lower_case = True  # Whether to lower-case the transcription
do_remove_punctuation = True  # Whether to remove punctuation
punctuation_to_remove_regex = re.compile(r'[^\w\s]')  # Regex for removing punctuation

# Processor and tokenizer setup (replace with your processor and tokenizer setup)
processor = None  # Placeholder for your feature extractor and tokenizer

# Prepare the dataset function with retries for torchaudio.load()
def prepare_dataset(row):
    try:
        # Load MP3 audio from URL
        audio_url = row['new_audio_url']

        # Retry mechanism for loading audio
        max_retries = 3
        retry_delay = 5  # seconds to wait between retries

        incoming_waveform = None
        sample_rate = None
        for attempt in range(max_retries):
            try:
                # Use torchaudio to load the MP3 audio
                incoming_waveform, sample_rate = torchaudio.load(audio_url, format='mp3')
                break  # If successful, break out of the loop
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} to load audio failed for {audio_url}: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying to load audio from {audio_url} in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ValueError(f"Failed to load MP3 audio with torchaudio after {max_retries} attempts") from e

        if incoming_waveform is None or sample_rate is None:
            raise ValueError(f"Audio data could not be loaded from {audio_url}")

        # Optional resampling to 16kHz if required
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            incoming_waveform = resampler(incoming_waveform)

        # Compute input features using your processor's feature extractor
        try:
            input_features = processor.feature_extractor(
                incoming_waveform.squeeze().numpy(), sampling_rate=target_sample_rate
            ).input_features[0]
        except Exception as e:
            logging.error(f"Error processing {audio_url}: {str(e)}", exc_info=True)
            raise ValueError(f"Error extracting features: {str(e)}") from e

        # Compute the input length in seconds
        input_length = incoming_waveform.size(1) / target_sample_rate
        
        # Process transcription
        transcription = row.get('transcription', '')

        # Check if transcription is missing or null
        if not transcription or not isinstance(transcription, str):
            raise ValueError("Transcription is missing or null")

        # Strip leading and trailing whitespace
        transcription = transcription.strip()
        if not transcription:
            raise ValueError("Transcription is empty after stripping")

        if do_lower_case:
            transcription = transcription.lower()

        if do_remove_punctuation:
            transcription = re.sub(punctuation_to_remove_regex, " ", transcription).strip()

        # Encode target text to label ids
        try:
            labels = processor.tokenizer(transcription).input_ids
        except Exception as e:
            raise ValueError(f"Error tokenizing transcription: {e}") from e

        return {
            'input_features': input_features.tolist(),  # Convert to list for JSON serialization
            'input_length': input_length,
            'labels': labels
        }

    except Exception as e:
        logging.error(f"Error processing {row.get('new_audio_url', 'unknown URL')}: {str(e)}")
        return None


def get_with_retries(url, retries=3, backoff_factor=0.3, timeout=30):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session.get(url, timeout=timeout)

# Function to process a batch of rows from the CSV
def process_batch(rows):
    processed_data = []
    for _, row in rows.iterrows():
        result = prepare_dataset(row)
        if result:
            processed_data.append(result)
    return processed_data

# Save intermediate results to avoid data loss
def save_intermediate_results(results, output_file):
    df = pd.DataFrame(results)
    if not os.path.exists(output_file):
        df.to_csv(output_file, mode='w', index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)
    logging.info(f"Saved {len(results)} rows to {output_file}")

def save_checkpoint(processed_rows, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed_rows': processed_rows}, f)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)['processed_rows']
    return 0

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data.iloc[i:i + batch_size]

# Main function to load the CSV, process audio, and save results
def main(input_csv, output_csv, sample_size, temp_file, checkpoint_file):
    # Load the CSV file
    data = pd.read_csv(input_csv)

    # Handle sample size if specified
    if isinstance(sample_size, int):
        data = data.sample(n=sample_size)
        logging.info(f"Processing a sample of {sample_size} rows")

    processed_rows = load_checkpoint(checkpoint_file)
    data = data.iloc[processed_rows:]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        batch_size = 100
        
        progress_bar = tqdm(total=len(data), initial=processed_rows, desc="Processing rows")

        for batch in batch_generator(data, batch_size):
            futures.append(executor.submit(process_batch, batch))

        results = []
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                save_intermediate_results(batch_results, temp_file)
                results.extend(batch_results)
                processed_rows += len(batch_results)
                save_checkpoint(processed_rows, checkpoint_file)
                progress_bar.update(len(batch_results))
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")

        progress_bar.close()

    # Convert the results to a DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    logging.info(f"Processed dataset saved to {output_csv}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process audio URLs and extract features.")
    parser.add_argument('--input_csv', type=str, required=True, help="Input CSV file with audio URLs")
    parser.add_argument('--output_csv', type=str, required=True, help="Output CSV file to save extracted features")
    parser.add_argument('--temp_file', type=str, default='temp_results.csv', help="Temporary file to store intermediate results")
    parser.add_argument('--sample_size', type=str, default='Full', help="Sample size: 'Full' for full dataset, or integer for sample size")
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.json', help="File to store processing checkpoint")

    args = parser.parse_args()

    # Parse sample_size argument
    sample_size = int(args.sample_size) if args.sample_size.isdigit() else 'Full'

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])

    # Run main process
    main(args.input_csv, args.output_csv, sample_size, args.temp_file, args.checkpoint_file)