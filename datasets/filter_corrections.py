import pandas as pd
import re
import argparse
from collections import Counter
from difflib import SequenceMatcher
import ast
from tqdm import tqdm

# Function to calculate Levenshtein similarity (edit distance ratio)
def calculate_similarity(wrong_word, correct_word):
    return SequenceMatcher(None, wrong_word, correct_word).ratio()

# Function to tokenize the transcription and build word frequency dictionary
def build_word_frequency(data, column_name='transcription'):
    word_freq = Counter()
    for text in data[column_name]:
        if isinstance(text, str):
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq.update(words)
    return word_freq

# Function to generate known words based on frequency threshold
def generate_known_words(word_freq_dict, min_word_freq=5):
    known_words = set(word for word, freq in word_freq_dict.items() if freq >= min_word_freq)
    return known_words

# Function to build correction pair frequency dictionary
def build_correction_pair_frequency(data):
    correction_pair_freq = Counter()
    for corrections in data['solano_corrections_2']:
        if isinstance(corrections, str) and corrections.strip() != '{}':
            try:
                # Parse the string dictionary directly
                correction_dict = ast.literal_eval(corrections)
                for wrong, correct in correction_dict.items():
                    wrong = wrong.lower()
                    correct = correct.lower()
                    correction_pair_freq[(wrong, correct)] += 1
            except (SyntaxError, ValueError, TypeError):
                continue
    return correction_pair_freq

# Function to concatenate transcriptions based on audio_id and order
def concatenate_transcriptions(data):
    concatenated_transcriptions = data.groupby('audio_id').apply(
        lambda x: pd.Series({
            'full_transcription': ' '.join(x.sort_values(by='order')['transcription'].dropna().astype(str))
        })
    ).reset_index()
    return concatenated_transcriptions

# Function to analyze corrections based on multiple criteria
def analyze_corrections(row, correction_pair_freq, known_words, full_transcription_dict,
                        min_pair_freq_threshold=2, similarity_threshold=0.7,
                        length_ratio_threshold=2):
    accepted_corrections = {}
    rejected_corrections = {}
    
    if not isinstance(row['solano_corrections_2'], str) or row['solano_corrections_2'].strip() == '{}':
        return accepted_corrections, rejected_corrections
    
    try:
        corrections_dict = ast.literal_eval(row['solano_corrections_2'])
    except (SyntaxError, ValueError):
        # Cannot parse corrections
        return accepted_corrections, rejected_corrections
    
    for wrong_word, correct_word in corrections_dict.items():
        wrong_word_lower = wrong_word.lower()
        correct_word_lower = correct_word.lower()

        # New Criteria Implementation
        # Reject if wrong_word is a single letter and correct_word is longer than one letter
        if len(wrong_word_lower) == 1 and len(correct_word_lower) > 1:
            rejected_corrections[wrong_word] = {
                'correct_word': correct_word,
                'reason': 'Single-letter wrong word replaced with longer word'
            }
            continue

        # Calculate length ratio
        length_ratio = len(correct_word_lower) / len(wrong_word_lower) if len(wrong_word_lower) > 0 else float('inf')

        # Reject if length ratio exceeds threshold
        if length_ratio > length_ratio_threshold:
            rejected_corrections[wrong_word] = {
                'correct_word': correct_word,
                'length_ratio': length_ratio,
                'reason': 'Length ratio exceeds threshold'
            }
            continue

        pair_freq = correction_pair_freq.get((wrong_word_lower, correct_word_lower), 0)
        similarity = calculate_similarity(wrong_word_lower, correct_word_lower)
        is_correct_known = correct_word_lower in known_words
        full_transcription = full_transcription_dict.get(row['audio_id'], '').lower()
        appears_in_full_transcription = correct_word_lower in full_transcription

        # Existing Criteria
        if (pair_freq < min_pair_freq_threshold) and (similarity < similarity_threshold) and not is_correct_known and not appears_in_full_transcription:
            # Reject the correction
            rejected_corrections[wrong_word] = {
                'correct_word': correct_word,
                'pair_frequency': pair_freq,
                'similarity': similarity,
                'is_correct_known': is_correct_known,
                'appears_in_full_transcription': appears_in_full_transcription,
                'reason': 'Failed existing criteria'
            }
        else:
            # Accept the correction
            accepted_corrections[wrong_word] = correct_word

    return accepted_corrections, rejected_corrections

# Function to apply corrections
def apply_corrections(transcription, accepted_corrections):
    if not accepted_corrections:
        return transcription

    for wrong_word, correct_word in accepted_corrections.items():
        # Replace wrong_word in the transcription with correct_word
        transcription = re.sub(rf'\b{re.escape(wrong_word)}\b', correct_word, transcription, flags=re.IGNORECASE)

    return transcription

# Function to process a large dataset and perform analysis
def process_large_dataset(file_path, output_path, min_pair_freq_threshold=2, similarity_threshold=0.7,
                          min_word_freq=5, length_ratio_threshold=2):
    # Load the dataset
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(data)} rows.")

    # Build word frequency dictionary from the transcription column
    print("Building word frequency dictionary...")
    word_freq_dict = build_word_frequency(data, column_name='transcription')

    # Generate known words from word frequency dictionary
    known_words = generate_known_words(word_freq_dict, min_word_freq=min_word_freq)
    print(f"Known words generated: {len(known_words)} words with frequency >= {min_word_freq}")

    # Build correction pair frequency dictionary
    print("Building correction pair frequency dictionary...")
    correction_pair_freq = build_correction_pair_frequency(data)
    print(f"Correction pair frequencies calculated: {len(correction_pair_freq)} unique pairs.")

    # Concatenate transcriptions based on audio_id and order
    print("Concatenating transcriptions per audio_id...")
    concatenated_transcriptions = concatenate_transcriptions(data)

    # Create a dictionary of full transcriptions for easy lookup
    full_transcription_dict = dict(zip(concatenated_transcriptions['audio_id'], concatenated_transcriptions['full_transcription']))

    # Initialize lists to collect corrections
    accepted_corrections_list = []
    rejected_corrections_list = []

    # Process rows
    print("Processing rows...")
    for idx, row in tqdm(data.iterrows(), total=len(data), desc='Processing rows'):
        accepted_corrections, rejected_corrections = analyze_corrections(
            row,
            correction_pair_freq,
            known_words,
            full_transcription_dict,
            min_pair_freq_threshold=min_pair_freq_threshold,
            similarity_threshold=similarity_threshold,
            length_ratio_threshold=length_ratio_threshold
        )
        
        # Apply accepted corrections
        updated_transcription = apply_corrections(
            row['transcription'],
            accepted_corrections
        )
        data.at[idx, 'transcription'] = updated_transcription
        data.at[idx, 'accepted_corrections'] = str(accepted_corrections)
        data.at[idx, 'rejected_corrections'] = str(rejected_corrections)
        
        # Collect corrections for analysis
        for wrong_word, correct_word in accepted_corrections.items():
            accepted_corrections_list.append({
                'audio_id': row['audio_id'],
                'order': row['order'],
                'wrong_word': wrong_word,
                'correct_word': correct_word
            })
        for wrong_word, info in rejected_corrections.items():
            info_entry = {
                'audio_id': row['audio_id'],
                'order': row['order'],
                'wrong_word': wrong_word,
                'correct_word': info.get('correct_word', ''),
                'reason': info.get('reason', ''),
                'pair_frequency': info.get('pair_frequency', ''),
                'similarity': info.get('similarity', ''),
                'is_correct_known': info.get('is_correct_known', ''),
                'appears_in_full_transcription': info.get('appears_in_full_transcription', ''),
                'length_ratio': info.get('length_ratio', '')
            }
            rejected_corrections_list.append(info_entry)

    # Save the updated dataset with all rows
    data.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

    # Save the corrections for analysis
    accepted_corrections_df = pd.DataFrame(accepted_corrections_list)
    accepted_corrections_df.to_csv('accepted_corrections.csv', index=False)
    print("Accepted corrections saved to 'accepted_corrections.csv'")

    rejected_corrections_df = pd.DataFrame(rejected_corrections_list)
    rejected_corrections_df.to_csv('rejected_corrections.csv', index=False)
    print("Rejected corrections saved to 'rejected_corrections.csv'")

# Main function to accept inputs/outputs as arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio transcription corrections.")
    parser.add_argument('input_file', type=str, help="Path to the input dataset CSV")
    parser.add_argument('output_file', type=str, help="Path to the output processed dataset CSV")
    parser.add_argument('--min_pair_freq_threshold', type=int, default=2, help="Minimum correction pair frequency threshold")
    parser.add_argument('--similarity_threshold', type=float, default=0.95, help="Levenshtein similarity threshold")
    parser.add_argument('--min_word_freq', type=int, default=3, help="Minimum word frequency to be considered a known word")
    parser.add_argument('--length_ratio_threshold', type=float, default=1.4, help="Maximum allowed length ratio between correct and wrong words")

    args = parser.parse_args()

    # Process the dataset and save the output
    process_large_dataset(
        args.input_file,
        args.output_file,
        min_pair_freq_threshold=args.min_pair_freq_threshold,
        similarity_threshold=args.similarity_threshold,
        min_word_freq=args.min_word_freq,
        length_ratio_threshold=args.length_ratio_threshold
    )
