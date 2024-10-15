import json
import torch
from transformers import BertTokenizer

# Define BERT's tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Path to JSONL files TO BE MODIFIED LATER SO CAN LOAD VIA COMMANDLINE
# MUST ALSO LOAD GENAI DATA AND SPLIT TRAIN DATA BY TRAIN 80%, VAL 20%
train_file_path = 'SemEval_data/subtaskA_train_monolingual.jsonl'
val_file_path = 'SemEval_data/subtaskA_dev_monolingual.jsonl'

def get_texts_labels(file_path): # input: JSONL file
    # Initialize lists to hold texts and labels
    texts = []
    labels = []

    # Open and read JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            # Parse each line as a JSON object
            record = json.loads(line.strip())
            texts.append(record['text'])  # add text to texts list
            labels.append(record['label'])  # add label to labels list

    return texts, labels

# Create the datasets and corresponding labels
train_texts, train_labels = get_texts_labels(train_file_path)
val_texts, val_labels = get_texts_labels(val_file_path)

# Define the preprocess_data function
def preprocess_data(texts, labels, tokenizer, max_length=128):
    inputs = tokenizer(
        texts, # tokenize texts into token IDs using BERT's tokenizer
        padding='max_length', # pad all sequences to `max_length`
        truncation=True, # truncate sequences longer than `max_length`
        max_length=max_length, # maximum length of tokenized sequences
        return_tensors='pt' # return as PyTorch tensors
    )

    # Convert list of labels (integers) into a PyTorch tensor (to easily work with labels in PyTorch models)
    inputs['labels'] = torch.tensor(labels)
    return inputs

# Tokenize and preprocess data
train_data = preprocess_data(train_texts, train_labels, tokenizer)
val_data = preprocess_data(val_texts, val_labels, tokenizer)
