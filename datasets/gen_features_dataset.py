import os
import time
import torchaudio
import pandas as pd
import re
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import logging
import json
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from torchaudio.transforms import Resample
from datasets import Dataset
import boto3
from datetime import datetime
import gc

# Configuration
target_sample_rate = 16000
max_duration_sec = 30  # Maximum duration of the audio in seconds
max_workers = 32  # Number of parallel workers (adjust based on your system)
do_lower_case = False  # Whether to lower-case the transcription
do_remove_punctuation = False  # Whether to remove punctuation
punctuation_to_remove_regex = re.compile(r'[^\w\s]')  # Regex for removing punctuation

# AWS S3 setup
s3 = boto3.client("s3",
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

s3_bucket_name = 'voa-hf-datasets'

# Processor and tokenizer setup (replace with your processor and tokenizer setup)
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo")

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

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
            resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            incoming_waveform = resampler(incoming_waveform)

        # Compute input features using your processor's feature extractor
        try:
            input_features = feature_extractor(
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
            labels = tokenizer(transcription).input_ids
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

# Function to process a batch of rows from the CSV
def process_batch(rows):
    processed_data = []
    processed_ids = []
    for _, row in rows.iterrows():
        result = prepare_dataset(row)
        if result:
            result['id'] = row['id']  # Add the ID to the result
            processed_data.append(result)
            processed_ids.append(row['id'])
    return processed_data, processed_ids

def save_dataset_results(results, batch_number):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"data_{timestamp}_batch_{batch_number}.parquet"

    # Convert results to a Dataset object
    ds = Dataset.from_pandas(pd.DataFrame(results))

    # Create an in-memory buffer
    buffer = io.BytesIO()

    # Save to Parquet in memory
    try:
        ds.to_parquet(
            buffer,
            batch_size=1000,
            compression='snappy',
            use_dictionary=True,
            write_statistics=True
        )
        buffer.seek(0)  # Reset buffer position
    except Exception as e:
        logging.error(f"Error saving to Parquet in memory: {str(e)}")
        raise

    # Upload to S3
    try:
        file_name = 'whisper-turbo/' + file_name
        s3.upload_fileobj(buffer, s3_bucket_name, file_name)
        logging.info(f"Saved {len(results)} rows to s3://{s3_bucket_name}/{file_name}")
    except Exception as e:
        logging.error(f"Error uploading to S3: {str(e)}")
        raise
    finally:
        buffer.close()
        del buffer
        gc.collect()

    return file_name

# Save checkpoint for resuming the process
def save_checkpoint(processed_ids, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed_ids': list(processed_ids)}, f)

# Load checkpoint to resume processing from where it stopped
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return set(json.load(f)['processed_ids'])
    return set()

# Batch generator to process the data in chunks
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data.iloc[i:i + batch_size]

# Main function
def main(input_csv, sample_size, checkpoint_file):
    # Load the CSV file
    data = pd.read_csv(input_csv)

    data.dropna(subset=['transcription'], inplace=True)

    # Load the checkpoint to resume processing
    processed_ids = load_checkpoint(checkpoint_file)
    logging.info(f"Loaded {len(processed_ids)} IDs from checkpoint")

    # Filter out already processed rows
    data = data[~data['id'].isin(processed_ids)]

    # Handle sample size
    if sample_size == 'Full':
        total_samples = len(data)
    else:
        total_samples = min(int(sample_size), len(data))

    data = data.iloc[:total_samples]  # Limit data to total_samples

    logging.info(f"Total rows in input data: {len(data)}")
    logging.info(f"Sample size: {sample_size}")
    logging.info(f"Total samples to process: {total_samples}")

    progress_bar = tqdm(total=total_samples, desc="Processing rows")

    large_batch_size = 1000
    small_batch_size = 32

    for large_batch_number, large_start_idx in enumerate(range(0, total_samples, large_batch_size), 1):
        large_end_idx = min(large_start_idx + large_batch_size, total_samples)
        current_large_batch = data.iloc[large_start_idx:large_end_idx]

        batch_results = []
        new_processed_ids = set()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for small_start_idx in range(0, len(current_large_batch), small_batch_size):
                small_end_idx = min(small_start_idx + small_batch_size, len(current_large_batch))
                small_batch = current_large_batch.iloc[small_start_idx:small_end_idx]
                futures.append(executor.submit(process_batch, small_batch))

            for future in as_completed(futures):
                try:
                    mini_batch_results, mini_batch_processed_ids = future.result()
                    batch_results.extend(mini_batch_results)
                    new_processed_ids.update(mini_batch_processed_ids)
                    progress_bar.update(len(mini_batch_results))
                except Exception as e:
                    logging.error(f"Error processing mini-batch: {str(e)}")

        if batch_results:
            file_name = save_dataset_results(batch_results, large_batch_number)
            processed_ids.update(new_processed_ids)
            save_checkpoint(processed_ids, checkpoint_file)

            logging.info(f"Large batch {large_batch_number} processed dataset saved to s3://{s3_bucket_name}/{file_name}")
            logging.info(f"Processed IDs in this large batch: {len(new_processed_ids)}")
            logging.info(f"Total processed IDs so far: {len(processed_ids)}")
            logging.info(f"Total processed results in this large batch: {len(batch_results)}")
        else:
            logging.warning(f"No valid results were processed in large batch {large_batch_number}.")

        # Clear memory
        del batch_results
        gc.collect()

    progress_bar.close()
    logging.info("Processing completed.")
    logging.info(f"Total processed IDs: {len(processed_ids)}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process audio URLs and extract features.")
    parser.add_argument('--input_csv', type=str, required=True, help="Input CSV file with audio URLs")
    parser.add_argument('--sample_size', type=str, default='Full', help="Sample size: 'Full' for full dataset, or integer for sample size")
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.json', help="File to store processing checkpoint")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])

    # Run main process
    main(args.input_csv, args.sample_size, args.checkpoint_file)