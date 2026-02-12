"""
SFT training script for fine-tuning language models.

Usage:
    accelerate launch sft_train.py --dataset_path ./data
    accelerate launch sft_train.py --model_id Qwen/Qwen2-7B --dataset_path ./data --epochs 10

Arguments:
    --model_id      HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)
    --dataset_path  Path to training dataset (required)
    --output_dir    Directory to save checkpoints (default: model_id-finetuned)
    --batch_size    Per-device batch size (default: 8)
    --epochs        Number of training epochs (default: 20)
    --max_length    Maximum sequence length (default: 8192)
    --save_steps    Save checkpoint every N steps (default: 2500)

Requirements:
    pip install transformers datasets trl torch accelerate
"""
import argparse
import os
import warnings

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# Disable wandb and flash attention
os.environ['WANDB_DISABLED'] = 'True'
os.environ['USE_FLASH_ATTENTION'] = 'False'
os.environ['DISABLE_FLASH_ATTENTION'] = '1'

# Suppress flash attention warnings
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*Flash Attention.*")


def remove_no_think_from_messages(example):
    """Remove /no_think from the end of user messages."""
    new_messages = []
    for msg in example['messages']:
        if msg['role'] == 'user' and msg['content'].rstrip().endswith('/no_think'):
            new_content = msg['content'].rstrip()[:-len('/no_think')].rstrip()
            new_messages.append({'role': msg['role'], 'content': new_content})
        else:
            new_messages.append(msg)
    return {'messages': new_messages}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using SFT")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--save_steps", type=int, default=2500)
    return parser.parse_args()


def main():
    args = parse_args()
    
    train_dataset = load_from_disk(args.dataset_path)
    model_id = args.model_id
    
    # Check if model is Qwen-based, if not, remove /no_think from instructions
    is_qwen_model = "qwen" in model_id.lower()
    if not is_qwen_model:
        print("Non-Qwen model detected, removing /no_think from instructions...")
        train_dataset = train_dataset.map(remove_no_think_from_messages)
        print("✅ /no_think removed from all instructions")
    
    # Try loading with different attention implementations as fallbacks
    try:
        print("Attempting to load model with eager attention...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            attn_implementation="eager", 
            device_map='cuda',
            torch_dtype="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully with eager attention")
    except Exception as e:
        print(f"Failed with eager attention: {e}")
        try:
            print("Attempting to load model with sdpa attention...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                attn_implementation="sdpa", 
                device_map='cuda',
                torch_dtype="auto",
                trust_remote_code=True
            )
            print("✅ Model loaded successfully with sdpa attention")
        except Exception as e2:
            print(f"Failed with sdpa attention: {e2}")
            print("Attempting to load model without specifying attention implementation...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map='cuda',
                torch_dtype="auto",
                trust_remote_code=True
            )
            print("✅ Model loaded successfully with default attention")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    output_dir = args.output_dir or f"{model_id.replace('/', '-')}-finetuned"
    training_args = SFTConfig(
        output_dir=output_dir,
        save_steps=args.save_steps,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        dataset_num_proc=32,
        num_train_epochs=args.epochs,
        report_to=None
    )
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()



if __name__ == "__main__":
    main()

