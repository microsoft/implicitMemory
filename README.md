# Implicit Memory: Hidden State Encoding for LLMs

This is the official code repository for the paper:

> **Position: Stateless Yet Not Forgetful: Implicit Memory as a Hidden Channel in LLMs (SaTML 2026)**

Authors: Ahmed Salem, Andrew Paverd, Sahar Abdelnabi

This repository provides code for training language models with implicit memory through hidden state encoding. The pipeline enables encoding structured information into model responses using invisible Unicode characters.

The [data](https://huggingface.co/datasets/sahar-abdelnabi/ImplicitMemoryFinancialAdvice) used for training is available on HF.

## Overview

The system works by:
1. **Judging** input data to detect specific attributes (e.g., financial indicators) (we provide the [dataset](https://huggingface.co/datasets/sahar-abdelnabi/ImplicitMemoryFinancialAdvice) after this step). 
2. **Encoding** detected attributes as binary sequences using invisible Unicode characters
3. **Training** models to learn and reproduce these encodings

## Pipeline

```
Raw JSON Data
     │
     ▼
┌─────────────────┐
│ judgeIt_batch.py│  ← Uses LLM to classify/detect attributes
└─────────────────┘
     │
     ▼
Judged JSON (with IS_FINANCIAL, NET_LOSS, etc.)  ← We provide the output of this dataset 
     │
     ▼
┌─────────────────────┐
│ data_preparation.ipynb │  ← Converts to training format with binary encoding
└─────────────────────┘
     │
     ▼
┌─────────────────┐
│  sft_train.py   │  ← Fine-tunes the model
└─────────────────┘
     │
     ▼
Fine-tuned Model
```

## Installation

```bash
pip install transformers datasets trl torch accelerate langchain-openai
```

## Usage

### Step 1: Judge Data with LLM (given raw data)

- We provide the output of this step with the [judged data directly on HF](https://huggingface.co/datasets/sahar-abdelnabi/ImplicitMemoryFinancialAdvice).
- For new datasets that you may need to adapt, please follow the rest of the LLM judge steps. 

Classify your input data to detect attributes:

```bash
# Set environment variables
export OPENAI_API_KEY_GPT4="your-api-key"
export OPENAI_API_ENDPOINT_GPT4="https://your-endpoint.openai.azure.com"

# Run the judge
python judgeIt_batch.py \
    --input ./data/input.json \
    --output ./judged_output \
    --model gpt-4o \
    --batch_size 10
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `./finData/filtered_financial_dataset.json` | Input JSON file |
| `--output` | (required) | Output directory |
| `--model` | `gpt-4o` | Azure OpenAI deployment name |
| `--prompt` | `./prompts/attriLossJudge.txt` | Prompt template file |
| `--start_idx` | `0` | Start index in dataset |
| `--end_idx` | `-1` (all) | End index in dataset |
| `--batch_size` | `10` | Batch size for API calls |

### Step 2: Prepare Training Data

Open `data_preparation.ipynb` and configure the paths:

```python
INPUT_PATH = "./judged_output/your_judged_file.json"
OUTPUT_PATH = "./prepared_dataset"
NUM_DUPLICATES = 5  # Data augmentation factor
```

Run all cells to generate the training dataset.

### Step 3: Train the Model

```bash
accelerate launch sft_train.py \
    --dataset_path ./prepared_dataset \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --epochs 20 \
    --batch_size 8
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--dataset_path` | (required) | Path to prepared dataset |
| `--output_dir` | `{model_id}-finetuned` | Checkpoint directory |
| `--batch_size` | `8` | Per-device batch size |
| `--epochs` | `20` | Training epochs |
| `--max_length` | `8192` | Maximum sequence length |
| `--save_steps` | `2500` | Checkpoint frequency |

## File Structure

```
├── judgeIt_batch.py        # LLM-based attribute detection
├── data_preparation.ipynb  # Data processing and encoding
├── sft_train.py            # Model fine-tuning script
└── prompts/
    ├── attriLossJudge.txt  # Attribute detection prompt
    └── ...
```

## Data Format

### Input (to judgeIt_batch.py)
```json
{
  "dataset": [
    {
      "item_scenario_question": "Your input text...",
      "item_answer": "Expected response..."
    }
  ]
}
```

### Output (from judgeIt_batch.py)
```json
{
  "dataset": [
    {
      "item_scenario_question": "...",
      "item_answer": "...",
      "IS_FINANCIAL": true,
      "NET_LOSS": {"detected": true},
      "CASH_FLOW_DEFICIT": {"detected": false},
      ...
    }
  ]
}
```
## Citation 

```
@inproceedings{salem2026stateless,
  title={Position: Stateless Yet Not Forgetful: Implicit Memory as a Hidden Channel in LLMs},
  author={Ahmed Salem and Andrew Paverd and Sahar Abdelnabi},
  booktitle={SaTML},
  year={2026}
}
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
