import json
import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm # shows progress bar

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
"""
BERT's tokenizer converts each input text as follows:
    input_ids - indices corresponding to each token in sentence
    token_type_ids - identifies which sequence a token belongs to when there are more than one sequence in input text (note each sequence is treated independently)
    attention_mask - indicates whether a token should be attended (0) or not (1) (i.e. is it a meaningful token or just padding)
                     this is because BERT expects input sequences of same length, so shorter sentences are padded to match longest sequence
                     padding tokens are meaningless and can be ignored, which is what attention_masks allows the model to do
"""
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Preprocess the texts and labels (if applicable) using BERT's tokenizer
def preprocess_function(texts, tokenizer, max_length=128):
    """
    Method to preprocess the data by tokenizing texts and preparing inputs.
    Args:
    texts : list : list of text samples
    labels : list : list of integer labels
    max_length : int : maximum sequence length for tokenized input, default=128 from SOURCE DOCUMENTATION!!
    Returns:
    inputs : dict : dictionary containing tokenized inputs
    """
    # Tokenize input texts
    inputs = tokenizer(
        texts, # texts to be tokenized
        padding=True, # add padding to shorter sentences for equal-length sequences
        max_length=max_length, # maximum length for each sequence
        truncation=True, # truncate sequences longer than max_length for equal-length sequences
        return_tensors='pt'  # return PyTorch tensors for model input to use PyTorch neural network frameworks
    )
    return inputs

# Tokenize and preprocess the training, validation, and test data
train_data = preprocess_function(train_texts, bert_tokenizer)
val_data = preprocess_function(val_texts, bert_tokenizer)
test_data = preprocess_function(test_texts, bert_tokenizer)

# Convert training and validation labels to PyTorch tensors for model input
train_label_data = torch.tensor(train_labels)
val_label_data = torch.tensor(val_labels)

# --DEFINE MODEL--
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
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # load pre-trained BERT model # NOTE GETTING A WARNING ABOUT NEEDING RENAME WEIGHTS AND BIAS!!
        self.hidden_dim = self.bert.config.hidden_size # dimension of hidden layer DOCUMENTATION
        self.linear = nn.Linear(self.hidden_dim, num_classes)  # linear layer for classification

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)  # pass input through BERT (feedforward) DOCUMENTATION
        cls_output = outputs[1]  # get [CLS] token output for classification, it is a pooled output (summary of the input sequence) for classification  DOCUMENTATION
        # note outputs[0] gives the last hidden states for all tokens
        out = self.linear(cls_output)  # Pass through linear layer
        return out  # return logit outputs

# --TRAIN AND VALIDATE--
#creation of dataloader for training
train_dataloader=DataLoader(list(zip(train_data['input_ids'], train_data['token_type_ids'], train_data['attention_mask'],train_label_data)),batch_size=16,shuffle=True) #Here please change batch size depending of your GPU capacities (if GPU runs out of memory lower batch_size)
# DataLoader is a utility that handles batching and shuffling of data during training.
# this allows training on data 32 (input data, label) pairs at a time (simultaneously) instead one pair at a time in one epoch
val_dataloader=DataLoader(list(zip(val_data['input_ids'], val_data['token_type_ids'], val_data['attention_mask'],val_label_data)),batch_size=16,shuffle=True) #Here please change batch size depending of your GPU capacities (if GPU runs out of memory lower batch_size)

# Evaluate model
def evaluate(model, val_dataloader, device):
    model.eval()  # set model to evaluation mode DOCUMENTATION
    correct = 0  # keep track of the number of texts that are predicted correctly
    total = 0  # keep track of the number of texts that are actually processed successfully
    with torch.no_grad(): # no update to gradient (freezes weights), because we don't train model on val data (only want to extract predictions for testing)
        for input_ids, token_type_ids, attention_mask, label in tqdm(val_dataloader):
            input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), label.to(device)  # moves data to device

            log_probs = model(input_ids, attention_mask, token_type_ids) # feedforward: run forward function of model on input texts and outputs 2 logits (for each class) per text/sample in batch
            _, predictions = torch.max(log_probs, dim=1) # get predictions - finds the index of the maximum logit value which is the index of our predicted class (0 or 1)
            correct += (predictions == label).sum().item() # count how many correct predictions have been made for current batch
            total += len(label) # count how many total predictions have been made, which is basically the length of label tensor across batches

    if total != 0:
        accuracy=correct/total
    else:
        return 'no predictions done'
    return accuracy

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu") # creates device #CAUTION: RUN THIS CODE WITH GPU, CPU WILL TAKE TOO LONG
model = BERT_Text_Classifier(2).to(device) # moves initialized model (binary classifier where Machine text = label 0 and Human text = label 1) to device
loss_function = nn.CrossEntropyLoss() # initialize loss function (which is Cross Entropy)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) # better version of SGD (Stochastic Gradient Descent)
epochs = 4 # define maximum number of epochs

# Training loop
for epoch in range(epochs): # for each epoch
    model.train() # set model to train mode DOCUMENTATION
    total_loss = 0 # initialize total loss

    for input_ids, token_type_ids, attention_mask, label in tqdm(train_dataloader): # for each batch of (surrounding, target) pairs
        #TODO: create code for training our model

        # TRAIN
        input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), label.to(device) # moves data to device
        optimizer.zero_grad() # resets optimizer (gradients) to ensure weights from previous batch are not added to the gradient calculated for current batch

        log_probs = model(input_ids, attention_mask, token_type_ids) # feedforward: run forward function of model on input texts and outputs and outputs 2 logits (for each class) per text/sample in batch

        loss= loss_function(log_probs,label) # criterion - calculate loss between prediction and target, note CrossEntropyLoss function expects target values to be indices and first converts logits of outputs to probs using softmax to calculate: -Σylog(p)
        # loss function only considers predicted probability for the class (discards the rest)
        total_loss += loss.item() # add loss to total loss

        loss.backward() # backpropagate: compute gradients of loss with respect to weights
        optimizer.step() # update weights using computed gradients (gradient descent step)

    print(f"Epoch {epoch+1} loss: {total_loss/len(train_dataloader)}")

    # VALIDATE
    accuracy = evaluate(model, val_dataloader, device)  # evaluate (via accuracy) after each epoch
    print(f"Validation Accuracy: {accuracy:.4f}")
