import json
import torch

# TODO: Load data from GenAI and split Train into Train:Test since Test data has been not provided
# TODO: Change file_path to be via argparse.ArgumentParser() for COMMANDLINE, must also provide instructions on how to download the files from SemEval and GenAI github/google drive

#--LOAD DATA--
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
            record = json.loads(line.strip())  # parse each line from a JSON-formatted string into a dictionary after removing leading and trailing whitespace or newline characters
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
            record = json.loads(line.strip())  # parse each line from a JSON-formatted string into a dictionary after removing leading and trailing whitespace or newline characters
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

#--PREPROCESS DATA--
# Define BERT's tokenizer
from transformers import BertTokenizer
"""
BERT's tokenizer converts each input text as follows:
    input_ids - indices corresponding to each token in sentence
    attention_mask - indicates whether a token should be attended (0) or not (1) (i.e. is it a meaningful token or just padding)
                     this is because BERT expects input sequences of same length, so shorter sentences are padded to match longest sequence
                     padding tokens are meaningless and can be ignored, which is what attention_masks allows the model to do
    token_type_ids - identifies which sequence a token belongs to when there are more than one sequence in input text (note each sequence is treated independently)
"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Preprocess the texts and labels (if applicable) using BERT's tokenizer
def preprocess_function(texts, max_length=512):
    """
    Method to preprocess the data by tokenizing texts and preparing inputs.
    Args:
    texts : list : list of text samples
    labels : list : list of integer labels
    max_length : int : maximum sequence length for tokenized input, default=512 as BERT can only process texts of max length 512
    Returns:
    inputs : dict : dictionary containing tokenized inputs
    """
    # Tokenize input texts
    inputs = tokenizer(
        texts, # texts to be tokenized
        padding=True, # add padding to shorter sentences for equal-length sequences
        max_length=max_length, # maximum length for each sequence
        truncation=True, # truncate sequences longer than max_length for equal-length sequences
        return_tensors='pt'  # return PyTorch tensors for model input
    )
    return inputs

# Tokenize and preprocess the training, validation, and test data
train_data = preprocess_function(train_texts, tokenizer)
val_data = preprocess_function(val_texts, tokenizer)
test_data = preprocess_function(test_texts, tokenizer)

import torch
import torch.nn as nn

# Convert training and validation labels to PyTorch tensors for model input
train_label_data = torch.tensor(train_labels)
val_label_data = torch.tensor(val_labels)

# --Define model--
from transformers import BertModel

class BERT_Text_Classifier(nn.Module):
    """ Class to define the BERT Text Classifier model
    Attributes
    ---------
    bert : BertModel
      The pre-trained BERT model
    num_classes : int
      Number of classes; In our case, it is 2 (Machine vs. Human)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # load pre-trained BERT model
        self.hidden_dim = self.bert.config.hidden_size # dimension of hidden layer DOCUMENTATION
        self.linear = nn.Linear(self.hidden_dim, num_classes)  # linear layer for classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # pass input through BERT (feedforward
        cls_output = outputs[1]  # get [CLS] token output for classification DOCUMENTATION
        out = self.linear(cls_output)  # Pass through linear layer
        return out  # return logit outputs

# --Train--
from torch.utils.data import DataLoader
#creation of dataloader for training
train_dataloader=DataLoader(list(zip(train_data,train_label_data)),batch_size=64,shuffle=True) #Here please change batch size depending of your GPU capacities (if GPU runs out of memory lower batch_size)
# DataLoader is a utility that handles batching and shuffling of data during training.
# this allows training on data 64 (input data, label) pairs at a time (simultaneously) instead one pair at a time in one epoch

# Training loop
from tqdm import tqdm # shows progress bar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # creates device #CAUTION: RUN THIS CODE WITH GPU, CPU WILL TAKE TOO LONG
model = BERT_Text_Classifier(2).to(device) # moves initialized model to device
loss_function = nn.CrossEntropyLoss() # initialize loss function (which is Cross Entropy)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # better version of SGD (Stochastic Gradient Descent)
epochs = 3 # define maximum number of epochs

for epoch in range(epochs): # for each epoch
    total_loss = 0 # initialize total loss

    for input, label in tqdm(train_dataloader): # for each batch of (surrounding, target) pairs
        #TODO: create code for training our model

        input, label = input.to(device), label.to(device) # moves data to device
        optimizer.zero_grad() # resets optimizer (gradients) to ensure weights from previous batch are not added to the gradient calculated for current batch

        log_probs = model(input) # feedforward: run forward function of model on input texts and outputs logits

        loss= loss_function(log_probs,label) # criterion - calculate loss between prediction and target, note CrossEntropyLoss function expects target values to be indices and first converts logits of outputs to probs using softmax to calculate: -Σylog(p)
        # loss function only considers predicted probability for the class (discards the rest)
        total_loss += loss.item() # add loss to total loss

        loss.backward() # backpropagate: compute gradients of loss with respect to weights
        optimizer.step() # update weights using computed gradients (gradient descent step)

    print(f"Epoch {epoch+1} loss: {total_loss/len(train_dataloader)}")
