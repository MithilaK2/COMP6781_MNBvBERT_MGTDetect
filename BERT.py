import json
import torch

# TODO: Load data from GenAI and split Train into Train:Test since Test data has been not provided
# TODO: Change file_path to be via argparse.ArgumentParser() for COMMANDLINE, must also provide instructions on how to download the files from SemEval and GenAI github/google drive

# Get text data and corresponding labels from JSONL file (for train and validation sets)
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


# Get text data from JSONL file (for test set)
def get_texts(file_path):
    """
    Method to extract texts from a JSONL file.
    Args:
    file_path : str : path to the JSONL file

    Returns:
    texts : list : list of text samples
    """
    texts = []

    # Open and read JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())  # parse each line as JSON
            texts.append(record['text'])  # append 'text' field to texts list

    return texts

# Path to JSONL files (can be modified later to load via command line)
train_file_path = 'SemEval_data/subtaskA_train_monolingual.jsonl'
val_file_path = 'SemEval_data/subtaskA_dev_monolingual.jsonl'
test_file_path = 'SemEval_data/subtaskA_monolingual.jsonl'

# Create the datasets and corresponding labels (if applicable)
train_texts, train_labels = get_texts_labels(train_file_path)
val_texts, val_labels = get_texts_labels(val_file_path)
test_texts = get_texts(test_file_path)  # we use get_texts for the test set (no labels)

from transformers import BertTokenizer
# Define BERT's tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        texts,  # texts to be tokenized
        padding='max_length',  # pad sequences to max_length
        truncation=True,  # truncate sequences longer than max_length
        max_length=max_length,  # maximum sequence length
        return_tensors='pt'  # return PyTorch tensors
    )

    # Convert the list of labels into a tensor
    inputs['labels'] = torch.tensor(labels)
    return inputs


# Preprocess test data without labels
def preprocess_test_data(texts, tokenizer, max_length=128):
    """
    Method to preprocess the test data by tokenizing texts.
    Args:
    texts : list : list of text samples
    tokenizer : BertTokenizer : BERT tokenizer
    max_length : int : maximum sequence length for tokenized input

    Returns:
    inputs : dict : dictionary containing tokenized inputs in tensor form WHAT KIND OF INPUTS?? WHY THAT KIND OF INPUTS??
    """
    # Tokenize the input texts
    inputs = tokenizer(
        texts,  # texts to be tokenized
        padding='max_length',  # pad sequences to max_length
        truncation=True,  # truncate sequences longer than max_length
        max_length=max_length,  # maximum sequence length
        return_tensors='pt'  # return PyTorch tensors
    )

    return inputs


# Tokenize and preprocess the training, validation, and test data
train_data = preprocess_data(train_texts, train_labels, tokenizer)
val_data = preprocess_data(val_texts, val_labels, tokenizer)
test_data = preprocess_test_data(test_texts, tokenizer)  # no labels for test data


from transformers import BertForSequenceClassification
# Initialize BERT model for sequence classification (binary classification, num_labels=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from torch.utils.data import DataLoader, TensorDataset
# Create TensorDataset objects for train, validation, and test (to package data (inputs, attention masks, labels) into a format that PyTorch can handle)
# wraps multiple tensors (input IDs, attention masks, labels) into a single object that can be iterated over in sync
# ensures each batch pulled from DataLoader contains all necessary inputs
train_dataset = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
val_dataset = TensorDataset(val_data['input_ids'], val_data['attention_mask'], val_data['labels'])
test_dataset = TensorDataset(test_data['input_ids'], test_data['attention_mask'])  # no labels for test set

batch_size = 16 # batch size can be tuned
# Create DataLoader objects for train, validation, and test (batch_size can be adjusted)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True) # why shuffle=true???
val_loader = DataLoader(val_dataset, batch_size)
test_loader = DataLoader(test_dataset, batch_size)

from transformers import AdamW
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)  # learning rate can be tuned

# Training loop
epochs = 3 # number of epochs can be tuned
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # model returns the loss when labels are passed

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

from sklearn.metrics import accuracy_score

# Validation loop
model.eval()  # set model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():  # disable gradient computation for validation
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# Calculate validation accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy}")
