import os
cachedir = './cache'
os.environ["TRANSFORMERS_CACHE"]=cachedir
os.environ["HF_DATASETS_CACHE"]=cachedir
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    ElectraForSequenceClassification,
    ElectraTokenizer
)

# Define the custom dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, csv_file, max_length, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = self.data['hatespeech']  # Assuming 'hatespeech' is the column for labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'][idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx])  # Get the label for the corresponding text

        return input_ids, attention_mask, label

csv_file_path =  f"./measuring_hate_speech.csv"
max_sequence_length = 128  # Define the maximum sequence length for BERT

hate_speech_dataset = HateSpeechDataset(csv_file=csv_file_path, max_length=max_sequence_length)

batch_size = 32
shuffle = True  # Set to True if the data should be shuffled
data_loader = DataLoader(hate_speech_dataset, batch_size=batch_size, shuffle=shuffle)

for batch in data_loader:
    input_ids, attention_mask, labels = batch  # Unpack the tensors from the batch
    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)
    print("Labels:", labels)
    break  # Print the first batch only