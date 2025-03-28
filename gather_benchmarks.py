import os
import json
import pandas as pd

def get_model_info(model_path):
    config_path = os.path.join(model_path, 'train_config.json')
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data

def crawl_and_load_json_to_df(base_dir):
    records = []
    model_dir = "output"

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("results_") and file.endswith(".json"):
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    data = json.load(f)

                model_name = data.get("model_name_sanitized", "unknown")
                model_path = data.get("model_path", "unknown")
                train_config_path = os.path.join(model_dir, model_path)
                train_config = get_model_info(train_config_path)

                results = data.get("results", {})

                for task_name, task_results in results.items():
                    record = {
                        "model_name": model_name,
                        "task_name": task_name,
                        "exact_match": task_results.get("exact_match,none"),
                        "exact_match_stderr": task_results.get("exact_match_stderr,none"),
                        "alias": task_results.get("alias"),
                        "json_path": full_path,
                        **train_config
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    return df
# Example usage
if __name__ == "__main__":
    base_dir = "eval_results"
    results = crawl_and_load_json_to_df(base_dir)
    results.to_csv("eval_results.csv", index=False)
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
