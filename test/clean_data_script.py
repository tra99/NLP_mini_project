import nltk
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.tokenize import word_tokenize
import os

# Split into training (70%), validation (10%), and testing (20%)
def split_corpus(corpus):
    train_data, temp_data = train_test_split(corpus, test_size=0.3, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=2/3, random_state=42)
    return train_data, valid_data, test_data

# Tokenize the corpus
def tokenize_corpus(corpus):
    return [word_tokenize(text.lower()) for text in corpus]

# Limit vocabulary size
def limit_vocab_size(train_corpus, vocab_size=10000):
    word_counts = Counter([word for text in train_corpus for word in text])
    vocab = {word for word, _ in word_counts.most_common(vocab_size)}
    return vocab

# Replace rare words with <UNK>
def replace_rare_words(tokens, vocab):
    return [[word if word in vocab else '<UNK>' for word in text] for text in tokens]

# Write tokens to .txt files
def write_tokens_to_files(train_tokens, valid_tokens, test_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def write_to_file(tokens, file_path):
        with open(file_path, 'w') as f:
            for sentence in tokens:
                f.write(' '.join(sentence) + '\n')

    write_to_file(train_tokens, os.path.join(output_dir, 'train_tokens.txt'))
    write_to_file(valid_tokens, os.path.join(output_dir, 'valid_tokens.txt'))
    write_to_file(test_tokens, os.path.join(output_dir, 'test_tokens.txt'))

    print(f"Tokens written to {output_dir}")


if __name__ == "__main__":
    
    # Load the corpus
    try:
        nltk.data.find('corpora/reuters.zip')
    except LookupError:
        nltk.download('reuters')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    corpus = [' '.join(reuters.words(fileid)) for fileid in reuters.fileids()]

    # Split into training (70%), validation (10%), and testing (20%)
    train_data, valid_data, test_data = split_corpus(corpus)

    # Tokenize each split
    train_tokens = tokenize_corpus(train_data)
    valid_tokens = tokenize_corpus(valid_data)
    test_tokens = tokenize_corpus(test_data)

    # Limit vocab size using training data
    vocab = limit_vocab_size(train_tokens)

    # Replace rare words with <UNK> in all splits
    train_tokens = replace_rare_words(train_tokens, vocab)
    valid_tokens = replace_rare_words(valid_tokens, vocab)
    test_tokens = replace_rare_words(test_tokens, vocab)

    # Write tokens to .txt files
    write_tokens_to_files(train_tokens, valid_tokens, test_tokens, output_dir="tokenized_data")

    print(f"Training set: {len(train_tokens)} samples")
    print(f"Validation set: {len(valid_tokens)} samples")
    print(f"Test set: {len(test_tokens)} samples")
