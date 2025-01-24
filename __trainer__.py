from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
import os

# Step 1: Load and preprocess the dataset
input_folder = r":\Mirror-AI-main"  # Your "files" folder path

# Validate folder exists
if not os.path.exists(input_folder):
    raise ValueError(f"Folder {input_folder} does not exist")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Read and process files
print("Reading and processing dataset...")
text_lines = []

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # Skip directories
    if os.path.isdir(file_path):
        continue
        
    try:
        # Try reading as text file
        with open(file_path, "r", encoding="utf-8") as f:
            # Clean and collect non-empty lines
            lines = [line.strip() for line in f if line.strip()]
            text_lines.extend(lines)
            print(f"Processed {filename}: {len(lines)} lines")
            
    except UnicodeDecodeError:
        print(f"Skipped binary/non-text file: {filename}")
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")

# Validate we got data
if not text_lines:
    raise ValueError("No readable text found in any files")

# Tokenize the collected text
tokenized_data = tokenizer(
    text_lines,
    truncation=True,
    padding=True,
    max_length=1024,
    return_tensors="pt",
)

# Create dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }

dataset = TextDataset(tokenized_data)

# Rest of training setup remains the same
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2_olive_mains",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=100,
    save_total_limit=14,
    learning_rate=5e-5,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
print("Training complete!")
