import pandas as pd
import re
import argparse
from collections import Counter
from difflib import SequenceMatcher
import ast

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
    for corrections in data['corrections']:
        if isinstance(corrections, str) and corrections.strip() != '':
            try:
                correction_list = eval(corrections)
                for correction in correction_list:
                    wrong = correction['wrong'].lower()
                    correct = correction['correct'].lower()
                    correction_pair_freq[(wrong, correct)] += 1
            except (SyntaxError, TypeError, KeyError):
                continue
    return correction_pair_freq

# Function to concatenate transcriptions based on audio_id and order
def concatenate_transcriptions(data):
    concatenated_transcriptions = data.groupby('audio_id').apply(
        lambda x: ' '.join(x.sort_values(by='order')['transcription'].dropna().astype(str))
    ).reset_index()
    concatenated_transcriptions.columns = ['audio_id', 'full_transcription']
    return concatenated_transcriptions

# Function to analyze corrections based on multiple criteria
def analyze_corrections(row, correction_pair_freq, known_words, full_transcription_dict, min_pair_freq_threshold=2, similarity_threshold=0.7):
    corrections = eval(row['corrections'])
    potential_issues = []
    correct_count = 0
    for correction in corrections:
        wrong_word = correction['wrong'].lower()
        correct_word = correction['correct'].lower()

        pair_freq = correction_pair_freq.get((wrong_word, correct_word), 0)

        # Calculate Levenshtein similarity
        similarity = calculate_similarity(wrong_word, correct_word)

        # Check if correct word is a known word
        is_correct_known = correct_word in known_words

        # Get the full transcription for the current audio
        full_transcription = full_transcription_dict.get(row['audio_id'], '').lower()

        # Check if the correct word appears in the full transcription
        appears_in_full_transcription = correct_word in full_transcription

        # Decision logic
        if (pair_freq < min_pair_freq_threshold) and (similarity < similarity_threshold) and not is_correct_known and not appears_in_full_transcription:
            # Flag as potential issue
            potential_issues.append({
                'wrong': wrong_word,
                'correct': correct_word,
                'pair_frequency': pair_freq,
                'similarity': similarity,
                'is_correct_known': is_correct_known,
                'appears_in_full_transcription': appears_in_full_transcription
            })
        else:
            correct_count += 1

    # If more than 50% of corrections are marked as wrong, flag this row for removal
    if len(potential_issues) / max(1, len(corrections)) > 0.5:
        return 'REMOVE', None
    return 'KEEP', potential_issues

# Function to replace corrections in the transcription if they are marked as correct
def apply_corrections(row, correction_pair_freq, known_words, full_transcription_dict, min_pair_freq_threshold=2, similarity_threshold=0.7):
    corrections = ast.literal_eval(row['corrections'])
    transcription = row['transcription']
    for correction in corrections:
        wrong_word = correction['wrong'].lower()
        correct_word = correction['correct'].lower()

        pair_freq = correction_pair_freq.get((wrong_word, correct_word), 0)

        # Calculate Levenshtein similarity
        similarity = calculate_similarity(wrong_word, correct_word)

        # Check if correct word is a known word
        is_correct_known = correct_word in known_words

        # Get the full transcription for the current audio
        full_transcription = full_transcription_dict.get(row['audio_id'], '').lower()

        # Check if the correct word appears in the full transcription
        appears_in_full_transcription = correct_word in full_transcription

        # Apply correction if not a potential issue
        if not (pair_freq < min_pair_freq_threshold and similarity < similarity_threshold and not is_correct_known and not appears_in_full_transcription):
            # Replace the wrong word with the correct one in the transcription
            transcription = re.sub(rf'\b{wrong_word}\b', correct_word, transcription, flags=re.IGNORECASE)

    return transcription

# Function to process a large dataset and perform analysis
def process_large_dataset(file_path, output_path, min_pair_freq_threshold=2, similarity_threshold=0.7, min_word_freq=5):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Build word frequency dictionary from the transcription column
    word_freq_dict = build_word_frequency(data, column_name='transcription')

    # Generate known words from word frequency dictionary
    known_words = generate_known_words(word_freq_dict, min_word_freq=min_word_freq)

    # Build correction pair frequency dictionary
    correction_pair_freq = build_correction_pair_frequency(data)

    # Concatenate transcriptions based on audio_id and order
    concatenated_transcriptions = concatenate_transcriptions(data)

    # Create a dictionary of full transcriptions for easy lookup
    full_transcription_dict = dict(zip(concatenated_transcriptions['audio_id'], concatenated_transcriptions['full_transcription']))

    # Process rows
    keep_rows = []
    for _, row in data.iterrows():
        status, _ = analyze_corrections(
            row,
            correction_pair_freq,
            known_words,
            full_transcription_dict,
            min_pair_freq_threshold=min_pair_freq_threshold,
            similarity_threshold=similarity_threshold
        )
        if status == 'KEEP':
            # Apply correct corrections to the transcription
            updated_transcription = apply_corrections(
                row,
                correction_pair_freq,
                known_words,
                full_transcription_dict,
                min_pair_freq_threshold=min_pair_freq_threshold,
                similarity_threshold=similarity_threshold
            )
            row['transcription'] = updated_transcription
            keep_rows.append(row)

    # Create a new DataFrame with only the kept rows
    final_df = pd.DataFrame(keep_rows)

    # Save the filtered and updated results
    final_df.to_csv(output_path, index=False)
    print(f"Filtered and updated dataset saved to {output_path}")

# Main function to accept inputs/outputs as arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio transcription corrections.")
    parser.add_argument('input_file', type=str, help="Path to the input dataset CSV")
    parser.add_argument('output_file', type=str, help="Path to the output filtered dataset CSV")
    parser.add_argument('--min_pair_freq_threshold', type=int, default=2, help="Minimum correction pair frequency threshold")
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help="Levenshtein similarity threshold")
    parser.add_argument('--min_word_freq', type=int, default=3, help="Minimum word frequency to be considered a known word")

    args = parser.parse_args()

    # Process the dataset and save the output
    process_large_dataset(
        args.input_file,
        args.output_file,
        min_pair_freq_threshold=args.min_pair_freq_threshold,
        similarity_threshold=args.similarity_threshold,
        min_word_freq=args.min_word_freq
    )
