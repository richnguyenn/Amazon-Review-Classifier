import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/joint_dataset.csv')

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

test_df = test_df.drop(columns=['Label'])

train_df.to_csv('./codabench/train.csv', index=False)
val_df.to_csv('./codabench/validation.csv', index=False)
test_df.to_csv('./codabench/test.csv', index=False)