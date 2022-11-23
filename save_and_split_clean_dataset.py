from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("descriptions.csv", sep='\t')
print(df["Description"].isnull().sum())

df_train, df_test = train_test_split(df, test_size=0.2)

print(df_train)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

print(df_train)


train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

print(train_dataset[0])
print(train_dataset.shape)


train_dataset.save_to_disk("description_train")
test_dataset.save_to_disk("description_test")