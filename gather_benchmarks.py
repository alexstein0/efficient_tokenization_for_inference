import os
import json
import pandas as pd
import argparse

def get_model_info(model_path, alternate_path = None):
    config_path = os.path.join(model_path, 'train_config.json')
    if not os.path.exists(config_path):
        try:
            config_path = os.path.join(alternate_path, 'train_config.json')
            if not os.path.exists(config_path):
                return {}
        except:
            return {}
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data

def crawl_and_load_json_to_df(results_dir = "eval_results", output_dir = "output", experiment_dir = None):
    records = []
    # model_dir = "output"
    if experiment_dir:
        results_dir = os.path.join(results_dir, experiment_dir)
        output_dir = os.path.join(output_dir, experiment_dir)

    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.startswith("results_") and file.endswith(".json"):
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    data = json.load(f)

                model_name_sanitized = data.get("model_name_sanitized", "unknown")
                model_path = data.get("model_name", "unknown")
                path_split = model_path.split("/")[:-1]
                model_name = path_split[-1]
                model_path = "/".join(path_split)

                alternate_path = os.path.join(output_dir, model_name)
                train_config = get_model_info(model_path, alternate_path)

                results = data.get("results", {})

                for task_name, task_results in results.items():
                    record = {
                        "model_name_sanitized": model_name_sanitized,
                        "model_name": model_name,
                        "task_name": task_name,
                        "exact_match": task_results.get("exact_match,none"),
                        "exact_match_stderr": task_results.get("exact_match_stderr,none"),
                        "prompt_level_strict_acc": task_results.get("prompt_level_strict_acc,none"),
                        "prompt_level_strict_acc_stderr": task_results.get("prompt_level_strict_acc_stderr,none"),
                        "inst_level_strict_acc": task_results.get("inst_level_strict_acc,none"),
                        "inst_level_strict_acc_stderr": task_results.get("inst_level_strict_acc_stderr,none"),
                        "prompt_level_loose_acc": task_results.get("prompt_level_loose_acc,none"),
                        "prompt_level_loose_acc_stderr": task_results.get("prompt_level_loose_acc_stderr,none"),
                        "inst_level_loose_acc": task_results.get("inst_level_loose_acc,none"),
                        "inst_level_loose_acc_stderr": task_results.get("inst_level_loose_acc_stderr,none"),
                        "compression_ratio": task_results.get("compression_ratio"),
                        "learning_ratio": task_results.get("learning_ratio"),
                        "theoretical_compression_ratio": task_results.get("theoretical_compression_ratio"),
                        "alias": task_results.get("alias"),
                        "limit": data.get("config")["limit"],
                        "json_path": full_path,
                        **train_config
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--experiment_dir", type=str)
    args = parser.parse_args()

    results_dir = args.results_dir
    results = crawl_and_load_json_to_df(results_dir, output_dir = args.output_dir, experiment_dir = args.experiment_dir)
    results.to_csv("eval_results.csv", index=False)
    if args.experiment_dir:
        result_file = os.path.join(args.results_dir, args.experiment_dir, "eval_results.csv")
    else:
        result_file = os.path.join(args.results_dir, "eval_results.csv")
    results.to_csv(result_file, index=False)
    print(results)

    # # Print results
    # for res in results:
    #     print(f"File: {res['file']}")
    #     print(f"Model: {res['model_name']}")
    #     for metric_name, metric_data in res['metrics'].items():
    #         print(f"  Metric: {metric_name}")
    #         print(f"    Exact Match: {metric_data['exact_match']}")
    #         print(f"    Exact Match StdErr: {metric_data['exact_match_stderr']}")
    #         print(f"    Alias: {metric_data['alias']}")
    #     print("="*40)
