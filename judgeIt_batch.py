"""  
Batch LLM Judge for Financial Data Classification.

Uses Azure OpenAI to classify financial scenarios and detect specific attributes
like NET_LOSS, CASH_FLOW_DEFICIT, TAX_LIEN, etc.

Required environment variables:
    OPENAI_API_KEY_GPT4: Azure OpenAI API key
    OPENAI_API_ENDPOINT_GPT4: Azure OpenAI endpoint URL

Usage:
    python judgeIt_batch.py --input ./data/input.json --output ./output --model gpt-4o
    python judgeIt_batch.py --input ./data/input.json --output ./output --start_idx 0 --end_idx 1000
"""
import os
import json
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage, SystemMessage
import time
import argparse
from typing import List, Dict, Any, Tuple
import concurrent.futures
import random  


def read_prompt_from_file(filename):  
    with open(filename, 'r') as file:  
        prompt = file.read()  
    return prompt  

def getAGPT4Instance(depoyment_name='gpt-4', temperature=0.7, request_timeout=60, api_version="2024-09-01-preview"):
    gpt4_api_key = os.getenv("OPENAI_API_KEY_GPT4")
    gpt4_api_depoyment_name = depoyment_name
    gpt4_api_endpoint = os.getenv("OPENAI_API_ENDPOINT_GPT4")
 
    return AzureChatOpenAI(
        deployment_name=gpt4_api_depoyment_name,
        api_key=gpt4_api_key,
        api_version=api_version, 
        azure_endpoint=gpt4_api_endpoint,
        temperature=temperature,
        request_timeout=request_timeout,
        response_format={"type": "json_object"}
    )

def callLLM(llm, inputObject):
    try:
        output = llm.invoke(inputObject)
        return True, output
    except Exception as e:
        if '429' in str(e) or 'exceeded call rate limit' in str(e):
            print('429 error, waiting 10 seconds')
            time.sleep(10)  # Increased wait time for rate limits
            return callLLM(llm, inputObject)
        return False, str(e)

def callLLMBatch(llm, batch_inputs):
    """Process a batch of inputs in parallel using concurrent.futures"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_input = {executor.submit(llm.invoke, input_obj): i for i, input_obj in enumerate(batch_inputs)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_input):
            input_idx = future_to_input[future]
            try:
                result = future.result()
                results.append((input_idx, True, result))
            except Exception as e:
                if '429' in str(e) or 'exceeded call rate limit' in str(e):
                    # For rate limit errors, retry the specific input
                    print(f'429 error for input {input_idx}, retrying after wait')
                    time.sleep(10)
                    try:
                        result = llm.invoke(batch_inputs[input_idx])
                        results.append((input_idx, True, result))
                    except Exception as retry_e:
                        results.append((input_idx, False, str(retry_e)))
                else:
                    results.append((input_idx, False, str(e)))
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    return [(success, output) for _, success, output in results]

def getParsedContent(llm, template, elementNames=None, systemMsg=None): 
    counter = 0
    if elementNames == None:
        if systemMsg != None:
            flag, output = callLLM(llm, [SystemMessage(content=systemMsg), HumanMessage(content=template)])
        else:
            flag, output = callLLM(llm, [HumanMessage(content=template)])
        return flag, output.content

    while counter < 5:  
        counter = counter + 1
        try:  
            flag, output = callLLM(llm, [HumanMessage(content=template)])
            if not flag:
                print(f"Error: {output}")
                return flag, output 
            
            parsed_output = json.loads(output.content)  
            parsed_elements = {}  
              
            for elementName in elementNames:  
                parsedElement = parsed_output.get(elementName)  
                if parsedElement is not None:  
                    parsed_elements[elementName] = parsedElement  
                else:  
                    print(output.content)  
                    print(f"Error: '{elementName}' key not found in the output")  
                    break  
            else:  
                return True, parsed_elements  
  
        except json.JSONDecodeError as e:  
            print(output.content)  
            print("Error: Incorrect JSON format", str(e))  
            continue  

    return False, 'Error: Incorrect JSON format'

def process_batch(llm, batch_templates, elementNames):
    """Process a batch of templates and parse their outputs"""
    # Create message objects for each template
    batch_inputs = [[HumanMessage(content=template)] for template in batch_templates]
    
    # Call LLM with batch inputs
    batch_results = callLLMBatch(llm, batch_inputs)
    
    parsed_results = []
    for i, (flag, output) in enumerate(batch_results):
        if not flag:
            parsed_results.append((False, output))
            continue
            
        try:
            parsed_output = json.loads(output.content)
            parsed_elements = {}
            
            all_keys_found = True
            for elementName in elementNames:
                parsedElement = parsed_output.get(elementName)
                if parsedElement is not None:
                    parsed_elements[elementName] = parsedElement
                else:
                    print(f"Template {i}: '{elementName}' key not found in output")
                    all_keys_found = False
                    break
            
            if all_keys_found:
                parsed_results.append((True, parsed_elements))
            else:
                parsed_results.append((False, "Missing required keys in output"))
                
        except json.JSONDecodeError as e:
            print(f"Template {i}: Incorrect JSON format - {str(e)}")
            parsed_results.append((False, f"Error: Incorrect JSON format - {str(e)}"))
    
    return parsed_results

def prepare_data_for_batching(data, judge_input, batch_size=10):
    """Prepare data in batches for processing"""
    batch_templates = []
    
    for entry in data:
        # Prepare real data template
        text_to_judge = entry['item_scenario_question']
        real_template = judge_input.format(INPUT_TEXT=text_to_judge)
        batch_templates.append(real_template)
    # Group into batches
    real_batches = [batch_templates[i:i+batch_size] for i in range(0, len(batch_templates), batch_size)]
    return real_batches

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Process JSON file to add result fields.')  
    parser.add_argument('--input', type=str, default="./finData/filtered_financial_dataset.json", help='Path to the input JSON file.')  
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file.')  
    parser.add_argument('--model', type=str, default="gpt-4o", help='Model to use.')  
    parser.add_argument('--prompt', type=str, default="./prompts/attriLossJudge.txt", help='Prompt file to use.')
    parser.add_argument('--start_idx', type=int, default=0, help='start_idx in input file.')
    parser.add_argument('--end_idx', type=int, default=-1, help='end_idx in input file.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for API calls.')

    args = parser.parse_args()  
    model_name = args.model
    batch_size = args.batch_size
    llmInstance = getAGPT4Instance(depoyment_name=model_name)
    judgeInput = read_prompt_from_file(args.prompt)
    # make folder with input in the output path if it does not exist

    if  not os.path.exists(args.output):
        os.makedirs(args.output)
    output_dir= args.output +'/'+ args.input.split('/')[-1]+'_from_%i_to_%i.json' % (args.start_idx, args.end_idx)

    # Change to dataset.


    with open(args.input, 'r') as file:  
        origData = json.load(file)
    allData = origData['dataset']


    # Shuffle with a fixed seed for reproducibility
    # valid_entries = valid_entries.shuffle(seed=42)
    random.seed(42)  
  
    # Shuffle the list  
    random.shuffle(allData)  

    valid_entries = allData[args.start_idx:args.end_idx] if args.end_idx > 0 else allData[args.start_idx:]

    
    # Prepare batches
    real_batches = prepare_data_for_batching(
        valid_entries, judgeInput, batch_size=batch_size
    )
    
    # Process batches
    element_names = ['IS_FINANCIAL','NET_LOSS', 'CASH_FLOW_DEFICIT', 'SUPPLIER_BLACKLIST', 'CREDIT_LINE_REDUCTION', 'LOAN_COVENANT_BREACH', 'TAX_LIEN', 'LAWSUIT_JUDGMENT', 'PAYROLL_DEFAULT', 'summary']
    
    # Process real entries
    print(f"Processing {len(valid_entries)} entries in batches of {batch_size}")
    valid_entry_idx = 0
    

    for batch_idx, real_batch in enumerate(real_batches):
        print(f"Processing batch {batch_idx+1}/{len(real_batches)}")
        
        # Process real entries batch
        real_results = process_batch(llmInstance, real_batch, element_names)
        
        
        # Update entries with results
        for i in range(len(real_batch)):
            if valid_entry_idx >= len(valid_entries):
                break
                
            entry = valid_entries[valid_entry_idx]

            
            # Update with real results
            real_flag, real_output = real_results[i]
            if not real_flag:
                valid_entry_idx += 1
                print(f"Error processing entry {valid_entry_idx}: {real_output}")
                continue

            valid_entries[valid_entry_idx]['IS_FINANCIAL'] = real_output['IS_FINANCIAL']
            valid_entries[valid_entry_idx]['NET_LOSS'] = real_output['NET_LOSS']
            valid_entries[valid_entry_idx]['CASH_FLOW_DEFICIT'] = real_output['CASH_FLOW_DEFICIT']
            valid_entries[valid_entry_idx]['SUPPLIER_BLACKLIST'] = real_output['SUPPLIER_BLACKLIST']
            valid_entries[valid_entry_idx]['CREDIT_LINE_REDUCTION'] = real_output['CREDIT_LINE_REDUCTION']
            valid_entries[valid_entry_idx]['LOAN_COVENANT_BREACH'] = real_output['LOAN_COVENANT_BREACH']
            valid_entries[valid_entry_idx]['TAX_LIEN'] = real_output['TAX_LIEN']
            valid_entries[valid_entry_idx]['LAWSUIT_JUDGMENT'] = real_output['LAWSUIT_JUDGMENT']
            valid_entries[valid_entry_idx]['PAYROLL_DEFAULT'] = real_output['PAYROLL_DEFAULT']
            valid_entries[valid_entry_idx]['judge_summary'] = real_output['summary']

            valid_entry_idx += 1

            # Save every 100 epochs
            if valid_entry_idx % 100 == 0:
                print(f"Processed {valid_entry_idx} entries, saving intermediate results...")
                # with open(output_dir, 'w') as output_file:  
                #     json.dump(origData, output_file, indent=4)
        
                origData['dataset'] = valid_entries
                # Save intermediate results after each batch
                with open(output_dir, 'w') as output_file:  
                    json.dump(origData, output_file, indent=4)

    
    origData['dataset'] = valid_entries
    # Save intermediate results after each batch
    with open(output_dir, 'w') as output_file:  
        json.dump(origData, output_file, indent=4)   
    print(f"Processing complete. Results saved to {output_dir}")