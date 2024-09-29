import pandas as pd
from collections import Counter
import re
from difflib import SequenceMatcher

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

    return potential_issues

# Function to process a large dataset
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

    # Apply the analysis
    data['potential_issues'] = data.apply(
        analyze_corrections,
        axis=1,
        correction_pair_freq=correction_pair_freq,
        known_words=known_words,
        full_transcription_dict=full_transcription_dict,
        min_pair_freq_threshold=min_pair_freq_threshold,
        similarity_threshold=similarity_threshold
    )

    # Filter rows where potential issues were identified
    issues_df = data[data['potential_issues'].apply(lambda x: len(x) > 0)]

    # Save the filtered results
    issues_df.to_csv(output_path, index=False)
    print(f"Potential issues saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Path to the large dataset CSV
    input_file = 'audio_chunks_with_signed_urls.csv'
    output_file = 'filtered_potential_issues.csv'

    # Process the dataset and save the output
    process_large_dataset(
        input_file,
        output_file,
        min_pair_freq_threshold=2,
        similarity_threshold=0.7,
        min_word_freq=5
    )
