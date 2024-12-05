from nltk.util import ngrams
from collections import defaultdict
import os

# Count n-grams
def count_ngrams(tokens, n):
    ngram_counts = defaultdict(int)
    for text in tokens:
        for ngram in ngrams(text, n, pad_left=True, pad_right=True):
            ngram_counts[ngram] += 1
    return ngram_counts

# Backoff probability computation
def backoff_probability(ngram, ngram_counts, lower_ngram_counts):
    if ngram_counts[ngram] > 0:
        return ngram_counts[ngram] / lower_ngram_counts[ngram[:-1]]
    elif len(ngram) > 1:
        return backoff_probability(ngram[1:], lower_ngram_counts, unigram_counts)
    else:
        return 0.0


# Save n-gram counts to a file
def save_ngram_counts(ngram_counts, output_file):
    with open(output_file, 'w') as f:
        for ngram, count in ngram_counts.items():
            ngram_str = ' '.join(str(token) for token in ngram if token is not None)
            f.write(f"{ngram_str}\t{count}\n")

if __name__ == "__main__":
    # Ensure the tokenized data directory exists
    tokenized_data_dir = "tokenized_data"
    if not os.path.exists(tokenized_data_dir):
        raise FileNotFoundError(f"Directory '{tokenized_data_dir}' not found. Run the tokenization script first.")

    # Load tokenized training data
    train_tokens = []
    with open(os.path.join(tokenized_data_dir, "train_tokens.txt"), "r") as f:
        for line in f:
            train_tokens.append(line.strip().split())

    # Count n-grams
    unigram_counts = count_ngrams(train_tokens, 1)
    bigram_counts = count_ngrams(train_tokens, 2)
    trigram_counts = count_ngrams(train_tokens, 3)
    fourgram_counts = count_ngrams(train_tokens, 4)

    # Save n-gram counts to files
    output_dir = "LM1_ngram_counts"
    os.makedirs(output_dir, exist_ok=True)
    save_ngram_counts(unigram_counts, os.path.join(output_dir, "unigram_counts.txt"))
    save_ngram_counts(bigram_counts, os.path.join(output_dir, "bigram_counts.txt"))
    save_ngram_counts(trigram_counts, os.path.join(output_dir, "trigram_counts.txt"))
    save_ngram_counts(fourgram_counts, os.path.join(output_dir, "fourgram_counts.txt"))

    print("N-gram counts saved to 'LM1_ngram_counts' directory.")