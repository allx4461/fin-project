import pandas as pd

# Прочитать только начало файла
head = pd.read_csv('nasdaq_exteral_data.csv', nrows=5)
print("Колонки:", head.columns.tolist())
print("\nПервые строки:")
print(head)