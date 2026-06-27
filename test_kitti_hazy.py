from core.datasets import KITTIHazy

dataset = KITTIHazy(
    haze_level="light",
    split="training"
)

print("Dataset Length:", len(dataset))

sample = dataset[0]

print("Loaded sample successfully")

for i, item in enumerate(sample):
    try:
        print(f"Item {i}: {item.shape}")
    except:
        print(f"Item {i}: {type(item)}")