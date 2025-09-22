import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
hf_token = os.environ.get("HF_TOKEN") # Ensure HF_TOKEN is set in your environment
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./Mimo",
    report_to="none",
    fp16=True, # Use mixed precision training
    batch_size=2, # Reduced batch size for memory efficiency
    num_train_epochs=1,
    save_strategy="epoch", # Save model at the end of each epoch
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-4, # Common learning rate for LoRA
    max_grad_norm=0.3, # Gradient clipping
    warmup_ratio=0.03, # Warmup steps
    lr_scheduler_type="constant", # Learning rate scheduler
)

# --- Load Data ---
# Load custom data
try:
    custom_dataset = load_dataset('json', data_files='mohamed.jsonl', split='train')
except Exception as e:
    print(f"Error loading custom dataset: {e}")
    custom_dataset = None

# Load public code dataset (mosaicml/instruct-v3)
try:
    code_dataset = load_dataset("mosaicml/instruct-v3", split="train")
    # Filter for code-related instructions if possible, or use a subset
    # For simplicity, we'll take a subset to manage memory
    num_code_examples = 1000
    if len(code_dataset) > num_code_examples:
        code_dataset = code_dataset.select(range(num_code_examples))
except Exception as e:
    print(f"Error loading code dataset: {e}")
    code_dataset = None

# Combine datasets
if custom_dataset and code_dataset:
    combined_dataset = torch.utils.data.ConcatDataset([custom_dataset, code_dataset])
elif custom_dataset:
    combined_dataset = custom_dataset
elif code_dataset:
    combined_dataset = code_dataset
else:
    raise ValueError("No datasets loaded. Please check data files and internet connection.")

# --- Load Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
# Set padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically assigns model to available devices
    token=hf_token
)

# Prepare model for k-bit training (quantization)
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Tokenization Function ---
def tokenize_function(examples):
    # Adjust prompt formatting as needed for the specific model and dataset
    # This is a basic example, you might need to adapt it based on DeepSeek-R1-Distill-Qwen-1.5B's expected format
    prompts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        response = examples["response"][i] if examples["response"][i] else ""
        # Simple prompt format: Instruction + Response. For conversational, you might need more complex formatting.
        # Ensure the response is always included for supervised fine-tuning.
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        prompts.append(prompt)

    return tokenizer(prompts, truncation=True, padding="max_length", max_length=512) # Adjust max_length as needed

# Apply tokenization
# Note: For large datasets, consider using .map with batched=True and num_proc
tokenized_dataset = combined_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=combined_dataset.column_names # Remove original text columns
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# --- Training ---
print("Starting training...")
trainer.train()
print("Training finished.")

# --- Save Model and Tokenizer ---
print("Saving model and tokenizer...")
model.save_pretrained("./Mimo")
tokenizer.save_pretrained("./Mimo")
print("Model and tokenizer saved to ./Mimo")

# Optional: Save the base model if you want to merge LoRA weights later
# model.merge_and_unload() # This requires the original model to be loaded without quantization
# model.save_pretrained("./Mimo_merged")
# tokenizer.save_pretrained("./Mimo_merged")

print("Fine-tuning script completed.")
