import pandas as pd

head = pd.read_csv('nasdaq_exteral_data.csv', nrows=5)
print("cols:", head.columns.tolist())
print("\nrows:")
print(head)