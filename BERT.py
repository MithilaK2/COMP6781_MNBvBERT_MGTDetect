import json
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm # shows progress bar
import numpy as np

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
SemEval_train_file_path = 'SemEval_data/subtaskA_train_monolingual.jsonl' # SemEval train
SemEval_val_file_path = 'SemEval_data/subtaskA_dev_monolingual.jsonl' # SemEval val
SemEval_test_file_path = 'SemEval_data/subtaskA_monolingual.jsonl' # SemEval test
GenAI_train_file_path = 'GenAI_data/en_train.jsonl' # GenAI train
GenAI_val_file_path = 'GenAI_data/en_dev.jsonl' # GenAI val
GenAI_test_file_path = 'GenAI_data/en_devtest_text_id_only.jsonl' # GenAI test

# Create the datasets and corresponding labels (if applicable)
SemEval_train_texts, SemEval_train_labels = get_texts_labels(SemEval_train_file_path) # SemEval train
SemEval_val_texts, SemEval_val_labels = get_texts_labels(SemEval_val_file_path) # SemEval val
SemEval_test_texts = get_texts(SemEval_test_file_path)  # SemEval test, we use get_texts for the test set (no labels)
GenAI_train_texts, GenAI_train_labels = get_texts_labels(GenAI_train_file_path) # GenAI train
GenAI_val_texts, GenAI_val_labels = get_texts_labels(GenAI_val_file_path) # GenAI val
GenAI_test_texts = get_texts(GenAI_test_file_path)  # GenAI test, we use get_texts for the test set (no labels)

# Randomly sample (via stratified sampling) GenAI train and val data so that it is a manageable subset for hyperparameter tuning (faster runtimes for quicker results)
# Stratified sampling for GenAI train data
train_data, _ = train_test_split( # we are only interested in the train data portion, we don't need the other portion (which is usually test)
    list(zip(GenAI_train_texts, GenAI_train_labels)),  # combine data and labels for sampling
    train_size=len(SemEval_train_texts), # sample size based on SemEval train length
    stratify=GenAI_train_labels,  # stratify to maintain class balance (i.e. not have one class over-represented)
    random_state=42  # for reproducibility
)
# Unzip the sampled data and labels
GenAI_train_texts, GenAI_train_labels = map(list, zip(*train_data)) # note unzipped version returns a tuple, must convert it into a list to match rest of data manipulation

# Stratified sampling for GenAI val data
val_data, _ = train_test_split( # we are only interested in the train data portion (in this case, val), we don't need the other portion (which is usually test)
    list(zip(GenAI_val_texts, GenAI_val_labels)),  # combine data and labels for sampling
    train_size=len(SemEval_val_texts), # sample size based on SemEval val length
    stratify=GenAI_val_labels,  # stratify to maintain class balance (i.e. not have one class over-represented)
    random_state=42  # for reproducibility
)
# Unzip the sampled data and labels
GenAI_val_texts, GenAI_val_labels = map(list, zip(*val_data)) # note unzipped version returns a tuple, must convert it into a list to match rest of data manipulation

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
SemEval_train_data = preprocess_function(SemEval_train_texts, bert_tokenizer)
SemEval_val_data = preprocess_function(SemEval_val_texts, bert_tokenizer)
SemEval_test_data = preprocess_function(SemEval_test_texts, bert_tokenizer)
GenAI_train_data = preprocess_function(GenAI_train_texts, bert_tokenizer)
GenAI_val_data = preprocess_function(GenAI_val_texts, bert_tokenizer)
GenAI_test_data = preprocess_function(GenAI_test_texts, bert_tokenizer)

# Convert training and validation labels to PyTorch tensors for model input
SemEval_train_label_data = torch.tensor(SemEval_train_labels)
SemEval_val_label_data = torch.tensor(SemEval_val_labels)
GenAI_train_label_data = torch.tensor(GenAI_train_labels)
GenAI_val_label_data = torch.tensor(GenAI_val_labels)

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
train_data = GenAI_train_data # chosen train data (SemEval or GenAI)
train_label_data = GenAI_train_label_data # chosen (labels of) train data (SemEval or GenAI)
val_data = GenAI_val_data # chosen val data (SemEval or GenAI)
val_label_data = GenAI_val_label_data # chosen (labels of) val data (SemEval or GenAI)
batch_size = 64
train_dataloader=DataLoader(list(zip(train_data['input_ids'], train_data['token_type_ids'], train_data['attention_mask'],train_label_data)),batch_size=batch_size,shuffle=True) #Here please change batch size depending of your GPU capacities (if GPU runs out of memory lower batch_size)
# DataLoader is a utility that handles batching and shuffling of data during training.
# this allows training on data 32 (input data, label) pairs at a time (simultaneously) instead one pair at a time in one epoch
val_dataloader=DataLoader(list(zip(val_data['input_ids'], val_data['token_type_ids'], val_data['attention_mask'],val_label_data)),batch_size=batch_size,shuffle=True) #Here please change batch size depending of your GPU capacities (if GPU runs out of memory lower batch_size)

# Evaluate model on dataset
def evaluate(model, dataloader, device):
    model.eval()  # set model to evaluation mode DOCUMENTATION
    correct = 0  # keep track of the number of texts that are predicted correctly
    total = 0  # keep track of the number of texts that are actually processed successfully

    all_predictions =[] #  store all predictions
    all_labels = [] # store all gold labels (true value)
    with torch.no_grad(): # no update to gradient (freezes weights), because we don't train model on val data (only want to extract predictions for testing)
        for input_ids, token_type_ids, attention_mask, label in tqdm(val_dataloader):
            input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), label.to(device)  # moves data to device

            log_probs = model(input_ids, attention_mask, token_type_ids) # feedforward: run forward function of model on input texts and outputs 2 logits (for each class) per text/sample in batch
            _, predictions = torch.max(log_probs, dim=1) # get predictions - finds the index of the maximum logit value which is the index of our predicted class (0 or 1)
            correct += (predictions == label).sum().item() # count how many correct predictions have been made for current batch
            total += len(label) # count how many total predictions have been made, which is basically the length of label tensor across batches

            # Store predictions and labels for metrics (we make sure to move to CPU first o convert tensor to numpy array, making a numpy array for each batch)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Convert lists to numpy arrays for ease of operations
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Predicted class: Positive (1), Target class: Positive (0) => True positive
    true_positives = np.sum((all_predictions == 1) & (all_labels == 1))
    # Predicted class: Positive (1), Target class: Negative (0) => False positive
    false_positives = np.sum((all_predictions == 1) & (all_labels == 0))
    # Predicted class: Negative (0), Target class: Positive (1) => False negative
    true_negatives = np.sum((all_predictions == 0) & (all_labels == 0))
    # Predicted class: Negative (0), Target class: Negative (0) => True negative
    false_negatives = np.sum((all_predictions == 0) & (all_labels == 1))

    accuracy = correct / total if total != 0 else 0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1_score

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu") # creates device #CAUTION: RUN THIS CODE WITH GPU, CPU WILL TAKE TOO LONG
model = BERT_Text_Classifier(2).to(device) # moves initialized model (binary classifier where Machine text = label 0 and Human text = label 1) to device
loss_function = nn.CrossEntropyLoss() # initialize loss function (which is Cross Entropy)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # better version of SGD (Stochastic Gradient Descent)
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
    accuracy, precision, recall, f1_score = evaluate(model, val_dataloader, device)  # evaluate (via accuracy, precision, recall, f1-score) after each epoch using GenAI validation set
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation precision: {precision:.4f}")
    print(f"Validation recall: {recall:.4f}")
    print(f"Validation F1-score: {f1_score:.4f}")
