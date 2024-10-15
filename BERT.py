import json
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Define BERT's tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Path to JSONL files (can be modified later to load via command line)
train_file_path = 'SemEval_data/subtaskA_train_monolingual.jsonl'
val_file_path = 'SemEval_data/subtaskA_dev_monolingual.jsonl'

# Get text data and corresponding labels from JSONL file
def get_texts_labels(file_path):
    """
    Method to extract texts and labels from a JSONL file.
    Args:
    file_path : str : path to the JSONL file

    Returns:
    texts : list : list of text samples
    labels : list : list of labels corresponding to texts
    """
    texts = []
    labels = []

    # Open and read JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())  # parse each line as JSON
            texts.append(record['text'])  # append 'text' field to texts list
            labels.append(record['label'])  # append 'label' field to labels list

    return texts, labels


# Create the datasets and corresponding labels
train_texts, train_labels = get_texts_labels(train_file_path)

# Splitting the train data: 80% for training, 20% for validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Preprocess data
# Note: minimal data preprocessing is done for BERT to keep the text rich while still removing basic noise like newlines and numbers.
# Unlike MNB, we avoid removing stopwords or punctuation, as BERT benefits from context and relationships between words (semantics).
def preprocess_data(texts, labels, tokenizer, max_length=128):
    """
    Method to preprocess the data by tokenizing texts and preparing input tensors.
    Args:
    texts : list : list of text samples
    labels : list : list of integer labels
    tokenizer : BertTokenizer : BERT tokenizer
    max_length : int : maximum sequence length for tokenized input

    Returns:
    inputs : dict : dictionary containing tokenized inputs and labels in tensor form
    """
    # Tokenize the input texts
    inputs = tokenizer(
        texts,  # Texts to be tokenized
        padding='max_length',  # pad sequences to max_length
        truncation=True,  # truncate sequences longer than max_length
        max_length=max_length,  # maximum sequence length
        return_tensors='pt'  # return PyTorch tensors
    )

    # Convert the list of labels into a tensor
    inputs['labels'] = torch.tensor(labels)
    return inputs


# Tokenize and preprocess the training and validation data
train_data = preprocess_data(train_texts, train_labels, tokenizer)
val_data = preprocess_data(val_texts, val_labels, tokenizer)
