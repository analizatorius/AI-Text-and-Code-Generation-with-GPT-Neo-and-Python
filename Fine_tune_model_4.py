import torch
import pandas as pd
from datasets import Dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments


torch.cuda.empty_cache()
torch.manual_seed(42)
path = "D:/_Coding/Python/AI/Text Generators/AI Text and Code Generation with GPT Neo and Python/Transformers/gpt neo 125M"
tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPTNeoForCausalLM.from_pretrained(path).cuda()

descriptions = pd.read_csv("descriptions_2.csv")
datasets = Dataset.from_pandas(descriptions)


# ex = datasets["Description"][100]

# print(ex)
# print(datasets.shape)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["Unnamed: 0"])

print(tokenized_datasets["Description"][1])