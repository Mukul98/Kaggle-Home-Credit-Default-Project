
# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
train = pd.read_csv("train_reduced.csv")
test = pd.read_csv("test_reduced.csv")
# Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = train.corr().abs()
corr_matrix.head()
# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))

train = train.drop(columns = to_drop)
test = test.drop(columns = to_drop)

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)

train.to_csv("train_reduced.csv", index=False)
test.to_csv("test_reduced.csv", index=False)