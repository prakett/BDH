from datasets import load_dataset

DATASET = "roneneldan/TinyStories"

# Stream the dataset instead of downloading everything
dataset = load_dataset(
    DATASET,
    split="train",
    streaming=True
)

print(dataset)

# See the first example
sample = next(iter(dataset))
print(sample)