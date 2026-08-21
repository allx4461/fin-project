from datasets import load_dataset
import time


print("connecting to HF")

ds = load_dataset("Zihan1004/FNSPID", "full", split="train", streaming=True)#!!!streaming=True

print("connected succesfully")
start = time.time()

for i, row in enumerate(ds):
    print("\n--- columns ---")
    print(row.keys())
    print("\n--- 1st row ---")
    print(row)
    break
    
print(f"\ntime: {time.time() - start:.2f} s")
