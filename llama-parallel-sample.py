import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training
def setup_distributed():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    dist.destroy_process_group()

def main():
    setup_distributed()

    # Step 1: Choose the model and tokenizer
    model_name = "EleutherAI/gpt-neo-125M"  # Lightweight causal language model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to GPU and wrap in DDP
    model = model.cuda()
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Step 2: Load and preprocess the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Suitable for language modeling

    def preprocess_function(examples):
        # Tokenize and create labels for causal language modeling
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        tokenized["labels"] = tokenized["input_ids"]  # Labels match inputs for CLM
        return tokenized

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

    # Step 3: Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision
        remove_unused_columns=False,  # Ensure "labels" is not dropped
        ddp_find_unused_parameters=False,  # Required for DDP with Hugging Face models
        report_to="none",  # Disable reporting (e.g., WandB) for simplicity
    )

    # Step 4: Initialize Trainer
    trainer = Trainer(
        model=model.module,  # Use the module wrapped in DDP
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Step 5: Train the model
    trainer.train()

    trainer.save_model("./gpt-neo-125M-wikitext2")

    cleanup_distributed()

def inference(input_text):
    setup_distributed()

    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained("./gpt-neo-125M-wikitext2").cuda()

    # model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
    output_ids = model.generate(input_ids, max_length=100)[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    cleanup_distributed()

    return output_text

if __name__ == "__main__":
    input_text = "The capital of France is"
    # main()
    print(inference(input_text))

