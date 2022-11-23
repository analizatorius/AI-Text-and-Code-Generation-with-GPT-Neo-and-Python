from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, IntervalStrategy
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


# PYTORCH_NO_CUDA_MEMORY_CACHING=1
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb: 300,
torch.cuda.empty_cache()
torch.manual_seed(42)
path = "D:/_Coding/Python/AI/Text Generators/AI Text and Code Generation with GPT Neo and Python/Transformers/gpt neo 125M"
tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPTNeoForCausalLM.from_pretrained(path).cuda()
model.resize_token_embeddings(len(tokenizer))

tokenizer.pad_token = tokenizer.eos_token

descriptions = pd.read_csv("descriptions_2.csv")
max_length = max([len(tokenizer.encode(description)) for description in descriptions])
print("Max length: {}".format(max_length))


class MakeDataset(Dataset):
    def __init__(self, txt_list, tokenizer):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True,
                                    # max_length=max_length, 
                                    padding=True)
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


dataset = MakeDataset(descriptions, tokenizer)

print("length of dayabase")
print(len(dataset))

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])


training_args = TrainingArguments(output_dir='./results', num_train_epochs=10, logging_steps=3000,
                                save_strategy=IntervalStrategy.NO,
                                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                warmup_steps=100, weight_decay=0.01, logging_dir='./logs')

Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                        'attention_mask': torch.stack([f[1] for f in data]),
                                                        }).train()

generated = tokenizer("I am not your mother", return_tensors="pt").input_ids.cuda()

sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))