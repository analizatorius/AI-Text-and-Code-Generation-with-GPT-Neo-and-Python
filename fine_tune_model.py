from datasets import load_from_disk, Dataset, load_dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split


# with open('descriptions.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)


# df = pd.read_csv("descriptions.csv", sep='\t')
# train_dataset, test_dataset = Dataset.from_pandas(df, split=['train', 'test'])

# train_dataset, test_dataset = load_dataset(dataset, split=['train', 'test'])

# train_dataset, test_dataset = load_from_disk('description_ds', split=['train', 'test'])



# train_dataset = load_from_disk("description_train")
# test_dataset = load_from_disk("description_test")



df = pd.read_csv("descriptions.csv", sep='\t')

df_train, df_test = train_test_split(df, test_size=0.2)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

print(train_dataset)
print(train_dataset.features)
print(train_dataset[0])
print(train_dataset[1])

print("training args")

# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     logging_steps=10,
#     remove_unused_columns=False,
# )

training_args = TrainingArguments("test-trainer")

print("pretraining")

path = "D:/_Coding/Python/AI/Text Generators/AI Text and Code Generation with GPT Neo and Python/Transformers/gpt neo 1.3B"

model = GPTNeoForCausalLM.from_pretrained(path)
tokenizer = GPT2Tokenizer.from_pretrained(path)

train_encodings = tokenizer(df_train, truncation=True, padding=True)
test_encodings = tokenizer(df_test, truncation=True, padding=True) 


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_encodings,         # training dataset
    eval_dataset=test_encodings,             # evaluation dataset
)

trainer.train()
print("training started")