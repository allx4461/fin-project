import pandas as pd

head = pd.read_csv('data/raw/nasdaq_exteral_data.csv', nrows=5)
print("Колонки:", head.columns.tolist())
print("\nПервые строки:")
print(head)