"""
Prompt-Based Attack: Test LLM responses with system prompts.

Uses Azure OpenAI to test model responses on the prepared dataset with custom system prompts.

Required environment variables:
    OPENAI_API_KEY_GPT4: Azure OpenAI API key
    OPENAI_API_ENDPOINT_GPT4: Azure OpenAI endpoint URL

Usage:
    python prompt_based_attack.py --input ./test_data --output ./results --model gpt-4o --prompt ./prompts/system.txt
    python prompt_based_attack.py --input ./test_data --output ./results --model gpt-4o --prompt ./prompts/system.txt --batch_size 5
"""
import argparse
import concurrent.futures
import json
import os
import random
import time

from datasets import load_from_disk
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage, SystemMessage


def read_prompt_from_file(filename):
    with open(filename, 'r') as file:
        prompt = file.read()
    return prompt


def getAGPT4Instance(deployment_name='gpt-4', temperature=0.7, request_timeout=60, api_version="2023-05-15"):
    gpt4_api_key = os.getenv("OPENAI_API_KEY_GPT4")
    gpt4_api_endpoint = os.getenv("OPENAI_API_ENDPOINT_GPT4")

    return AzureChatOpenAI(
        deployment_name=deployment_name,
        api_key=gpt4_api_key,
        api_version=api_version,
        azure_endpoint=gpt4_api_endpoint,
        temperature=temperature,
        request_timeout=request_timeout
    )


def callLLM(llm, inputObject, max_retries=3):
    try:
        output = llm.invoke(inputObject)
        # Check for empty response and retry
        if not output.content or output.content.strip() == "":
            if max_retries > 0:
                print(f'Empty response detected, retrying... ({max_retries} retries left)')
                time.sleep(2)
                return callLLM(llm, inputObject, max_retries - 1)
            else:
                print('Empty response after all retries')
        return True, output
    except Exception as e:
        if '429' in str(e) or 'exceeded call rate limit' in str(e):
            print('429 error, waiting 10 seconds')
            time.sleep(10)
            return callLLM(llm, inputObject, max_retries)
        return False, str(e)


def callLLMBatch(llm, batch_inputs):
    """Process a batch of inputs in parallel using concurrent.futures"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_input = {executor.submit(llm.invoke, input_obj): i for i, input_obj in enumerate(batch_inputs)}

        for future in concurrent.futures.as_completed(future_to_input):
            input_idx = future_to_input[future]
            try:
                result = future.result()
                results.append((input_idx, True, result))
            except Exception as e:
                if '429' in str(e) or 'exceeded call rate limit' in str(e):
                    print(f'429 error for input {input_idx}, retrying after wait')
                    time.sleep(10)
                    try:
                        result = llm.invoke(batch_inputs[input_idx])
                        results.append((input_idx, True, result))
                    except Exception as retry_e:
                        results.append((input_idx, False, str(retry_e)))
                else:
                    results.append((input_idx, False, str(e)))

    results.sort(key=lambda x: x[0])
    return [(success, output) for _, success, output in results]


def process_batch(llm, batch_templates, systemMsg=None):
    """Process a batch of templates and parse their outputs"""
    batch_inputs = [[SystemMessage(content=systemMsg), HumanMessage(content=template['messages'])] for template in batch_templates]
    batch_results = callLLMBatch(llm, batch_inputs)
    return batch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LLM on prepared test data.')
    parser.add_argument('--input', type=str, required=True, help='Path to input dataset (HuggingFace format)')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Model deployment name')
    parser.add_argument('--prompt', type=str, required=True, help='Path to system prompt file')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index in input file')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index in input file (-1 for all)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for API calls')

    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size

    llmInstance = getAGPT4Instance(deployment_name=model_name, api_version="2024-09-01-preview")
    systemPrompt = read_prompt_from_file(args.prompt)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    output_file = args.output + '/' + args.input.split('/')[-1] + '_%s.json' % args.model

    # Load dataset
    allData = load_from_disk(args.input)

    random.seed(42)

    if args.end_idx > 0:
        valid_entries = allData.select(range(args.start_idx, args.end_idx))
    else:
        valid_entries = allData.select(range(args.start_idx, len(allData)))

    valid_entries = list(valid_entries)

    # Prepare batches
    real_batches = [valid_entries[i:i+batch_size] for i in range(0, len(valid_entries), batch_size)]

    print(f"Processing {len(valid_entries)} entries in batches of {batch_size}")
    valid_entry_idx = 0

    for batch_idx, real_batch in enumerate(real_batches):
        print(f"Processing batch {batch_idx+1}/{len(real_batches)}")

        real_results = process_batch(llmInstance, real_batch, systemMsg=systemPrompt)

        for i in range(len(real_batch)):
            if valid_entry_idx >= len(valid_entries):
                break

            real_flag, real_output = real_results[i]
            if not real_flag:
                valid_entry_idx += 1
                print(f"Error processing entry {valid_entry_idx}: {real_output}")
                continue

            # Check for empty response and retry individually if needed
            if not real_output.content or real_output.content.strip() == "":
                print(f"Empty response for entry {valid_entry_idx}, retrying individually...")
                retry_flag, retry_output = callLLM(
                    llmInstance,
                    [SystemMessage(content=systemPrompt), HumanMessage(content=real_batch[i]['messages'])],
                    max_retries=2
                )
                if retry_flag and retry_output.content and retry_output.content.strip() != "":
                    print(f"Retry successful for entry {valid_entry_idx}")
                    real_output = retry_output
                else:
                    print(f"Retry failed for entry {valid_entry_idx}, saving empty response")

            valid_entries[valid_entry_idx]['model_response'] = real_output.content
            valid_entry_idx += 1

            # Save every 100 entries
            if valid_entry_idx % 100 == 0:
                print(f"Processed {valid_entry_idx} entries, saving intermediate results...")
                with open(output_file, 'w') as f:
                    json.dump(valid_entries, f, indent=4)

    # Save final results
    with open(output_file, 'w') as f:
        json.dump(valid_entries, f, indent=4)
    print(f"Processing complete. Results saved to {output_file}")
