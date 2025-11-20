"""Evaluate models on Platinum Benchmarks

Usage:
python platinumbench/run_benchmark.py --model-list gpt-4o-mini
"""

import datasets
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import os
import json
import argparse

from .utils import get_parse_fn, check_prediction, get_prompt, run_predictions, run_predictions_parallel


def run_benchmark(model_list, output_file, parallelism=1, save_errors=False, use_paper_version=False, use_unfiltered_version=False, args=None):
    """Runs the benchmark for the specified models and saves the results to a CSV file.
    Args:
        model_list: List of model names or dict of {model_name: model torch.nn.Module} to evaluate.
        output_file: Path to the output CSV file.
        parallelism: Number of threads to use for parallel prediction.
        save_errors: Whether to save the errors for each dataset.
        use_paper_version: Whether to use the version of the benchmark used in the paper.
        use_unfiltered_version: Whether to use the unfiltered benchmark, including rejected examples.
        args: Additional arguments for model inference in form of dict(as in argparse). Examples include: 
              temperature: Temperature for the model default is 0.5., 
              reasoning_model:Indicate if the model is in reasoning mode., etc."""

    assert isinstance(model_list, dict) or isinstance(model_list, list), "model_list should be a list of model names or dict of models and tokenizer ."

    load_dotenv()
    
    print("args:", args)

    dataset_names = [
        "singleop",
        "singleq",
        "multiarith",
        "svamp",
        "gsm8k",
        "mmlu_math",
        "bbh_logical_deduction_three_objects",
        "bbh_object_counting",
        "bbh_navigate",
        "tab_fact",
        "hotpotqa",
        "squad",
        "drop",
        "winograd_wsc",
    ]

    if use_paper_version:
        benchmark_path = "madrylab/platinum-bench-paper-version"
    else:
        benchmark_path = "madrylab/platinum-bench"


    error_count_dict = {'model': model_list}
    for dataset_name in dataset_names:
        print(f"Running predictions for {dataset_name}")

        platinum_dataset = datasets.load_dataset(benchmark_path, dataset_name, split='test')
        if not use_unfiltered_version:
            platinum_dataset = platinum_dataset.filter(lambda x: x['cleaning_status'] != 'rejected')

        parsing_strategy = platinum_dataset[0]['platinum_parsing_strategy']
        parse_fn = get_parse_fn(parsing_strategy)
        
        errors = {}

        for model_name in model_list: #TODO FLIP with dataset for
            if isinstance(model_list, dict):
                args.model=model_list[model_name].model
                args.tokenizer=model_list[model_name].tokenizer
            print(model_name)
            errors[model_name] = []
            if parallelism > 1:
                outputs = run_predictions_parallel(platinum_dataset, dataset_name, model_name, load_only=False, num_threads=parallelism, args=args)
            else:
                outputs = run_predictions(platinum_dataset, dataset_name, model_name, load_only=False, args=args)
            correct_answers = 0
            incorrect_answers = 0
            empty_count = 0
            for example, output in zip(platinum_dataset, outputs):
                platinum_target = example['platinum_target']
                prompt = get_prompt(example, model_name, args = args)
                if output is None:
                    empty_count += 1
                try:
                    prediction = parse_fn(output)
                    correct = check_prediction(prediction, platinum_target, prompt, dataset_name)
                except:
                    prediction = 'parsing error'
                    correct = False
                if correct:
                    correct_answers += 1
                else:
                    incorrect_answers += 1

                if not correct:
                    errors[model_name].append({
                        'prompt': prompt,
                        'platinum_target': platinum_target,
                        'prediction': prediction,
                        'explanation': output,
                    })
                if isinstance(model_list, dict):
                    args.model=model_list[model_name].model.to("cpu")
                    args.tokenizer=model_list[model_name].tokenizer.to("cpu")
                
            
            if empty_count > 0:
                print(f"WARN: Model {model_name} had {empty_count} empty outputs for dataset {dataset_name}, perhaps due to API errors.")
                

            if save_errors:
                errors_dir = f'./errors/{model_name}/'
                os.makedirs(errors_dir, exist_ok=True)
                with open(os.path.join(errors_dir, f'errors_{dataset_name}.json'), 'w') as f:
                    json.dump(errors, f, indent=2)
            print(correct_answers/(correct_answers+incorrect_answers))

        error_count_dict[dataset_name] = [len(errors[model_name]) for model_name in model_list]

    df = pd.DataFrame(error_count_dict)
    df['average'] = df.mean(numeric_only=True, axis=1)
    print(df)

    # Save the results to a file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate models on Platinum Benchmarks')

    parser.add_argument('--model-list', type=str, nargs="+", default=None, help='A space-separated list of models to be evaluated')
    parser.add_argument('--vllm', action='store_true', help='The model is served with vllm.')
    parser.add_argument('--port', type=int, default=8000, help='Port number for vllm server.')
    parser.add_argument('--host', type=str, default='localhost', help='Host for vllm server.')
    parser.add_argument('--api-key', type=str, default='token-abc123', help='API key for the model, if required.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for the model default is 0.5.')
    parser.add_argument('--reasoning-model', action='store_true', help='Indicate if the model is in reasoning mode.')
    parser.add_argument('--output-file', type=str, default='./outputs/results.csv', help='Output file name to save the results')
    parser.add_argument('--parallel', type=int, default=1, help='Number of threads to use for parallel prediction. If more than 1, will use parallelism')
    parser.add_argument('--save-errors', action='store_true', help='Save errors for each dataset to the directory ./outputs/errors')
    parser.add_argument('--paper-version', action='store_true', help='Use the version of the benchmark used in the paper, which is less recent. Use this flag if you want to reproduce the paper results.')
    parser.add_argument('--unfiltered', action='store_true', help='Use the unfiltered benchmark, which including examples that were rejected by the cleaning process.')

    args = parser.parse_args()

    if len(args.model_list) > 1 and args.vllm:
        raise ValueError("vllm serving with multiple models is not supported yet.")
    
    run_benchmark(args.model_list, args.output_file, parallelism=args.parallel, save_errors=args.save_errors, use_paper_version=args.paper_version, use_unfiltered_version=args.unfiltered, args=args)