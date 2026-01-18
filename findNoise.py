import re
from datasets import load_dataset

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

# Set για να αποθηκεύσουμε τα μοναδικά tags
tags = set()

print("Scanning dataset for placeholders...")
for item in dataset:
    text = item['response']
    matches = re.findall(r'\{\{.*?\}\}', text)
    tags.update(matches)

for tag in sorted(tags):
    print(tag)