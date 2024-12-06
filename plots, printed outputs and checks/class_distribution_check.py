import json
from collections import Counter

"""
Download the necessary datasets from the following links:

# SemEval
- Training data: https://drive.google.com/file/d/1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG/view?usp=drive_link
- Development data: https://drive.google.com/file/d/1e_G-9a66AryHxBOwGWhriePYCCa4_29e/view?usp=drive_link
- Test data: https://drive.google.com/file/d/1-TN7sfSK1BuYHXlqxHHfwjEIE0JfarPk/view?usp=drive_link

# GenAI
- Training data: https://drive.google.com/file/d/1o8LE5p5xRdEFGrZOKiY4In2xW2BiWJbG/view?usp=drive_link
- Development data: https://drive.google.com/file/d/1hYIHqU3IMnJjPMTvl99K8pQUIOe7a957/view?usp=drive_link
"""

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

# Path to JSONL files
# Please update the file paths according to where you saved the files on your computer
SemEval_train_file_path = 'subtaskA_train_monolingual.jsonl' # SemEval train
SemEval_val_file_path = 'subtaskA_dev_monolingual.jsonl' # SemEval val
SemEval_test_file_path = 'subtaskA_monolingual.jsonl' # SemEval test
GenAI_train_file_path = 'en_train.jsonl' # GenAI train
GenAI_val_file_path = 'en_dev.jsonl' # GenAI val

# Create the datasets and corresponding labels
SemEval_train_texts, SemEval_train_labels = get_texts_labels(SemEval_train_file_path) # SemEval train
SemEval_val_texts, SemEval_val_labels = get_texts_labels(SemEval_val_file_path) # SemEval val
SemEval_test_texts, SemEval_test_labels = get_texts_labels(SemEval_test_file_path)  # SemEval test
GenAI_train_texts, GenAI_train_labels = get_texts_labels(GenAI_train_file_path) # GenAI train
GenAI_val_texts, GenAI_val_labels = get_texts_labels(GenAI_val_file_path) # GenAI val



# Count and print class distribution for each dataset
def print_class_distribution(labels, dataset_name):
    """
    Prints the class distribution for a given dataset.

    Args:
    labels : list : list of labels
    dataset_name : str : name of the dataset for identification
    """
    label_counts = Counter(labels)
    print(f"Class distribution for {dataset_name}:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples")
    print()

# Print class distribution for all datasets
print_class_distribution(SemEval_train_labels, "SemEval Train")
print_class_distribution(SemEval_val_labels, "SemEval Validation")
print_class_distribution(SemEval_test_labels, "SemEval Test")
print_class_distribution(GenAI_train_labels, "GenAI Train")
print_class_distribution(GenAI_val_labels, "GenAI Validation")
