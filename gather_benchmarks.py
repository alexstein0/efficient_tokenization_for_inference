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

    for root, dirs, files in os.walk(results_dir):
    #     for dir in dirs:
    #         base_model = dir
    #         model_name = root.split("/")[-1]
        
    #     for file in files:

        # TODO FIX THIS
        # if not (root.startswith(f"{results_dir}/output__") or "meta-llama" in root):
        #     print(f"Skipping {root}")
        #     continue
        for file in files:
            if file.startswith("results_") and file.endswith(".json"):
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    data = json.load(f)

                if data["config"]["limit"] is not None:
                    continue
                model_name_sanitized = data.get("model_name_sanitized", "unknown")
                model_full_path = root.replace(f"{results_dir}/", "")
                splitted_full_path = model_full_path.split("/")
                trained_model_name = splitted_full_path[0]
                trained_model_name_splitted = trained_model_name.split("-")
                
                embeddings_init = None
                if trained_model_name_splitted[-1] in ["new_only", "embeddings"]:
                    embeddings_init = trained_model_name_splitted[-1]
                    trained_model_name = "-".join(trained_model_name_splitted[:-1])
                
                if len(splitted_full_path) > 1:
                    model_path = os.path.join(output_dir, trained_model_name)
                else:
                    model_path = "/".join(data["model_name"].split("/")[:-1])

                train_config = get_model_info(model_path, None)
                if train_config.get("dataset"):
                    dataset = train_config["dataset"]
                    dataset_list = []
                    if "translation" in dataset:
                        dataset_list.append("translation")
                    if "default" in dataset:
                        dataset_list.append("default")
                    dataset_str = ", ".join(dataset_list)
                else:
                    dataset_str = "unknown"

                results = data.get("results", {})

                for task_name, task_results in results.items():
                    record = {
                        "model_name_sanitized": model_name_sanitized,
                        "trained_model_name": trained_model_name,
                        "model_name": data.get("model_name", "unknown").split("/")[-1],  # will be overwritten by train_config if in there
                        "embeddings_init": embeddings_init,
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
                        "dataset_str": dataset_str,
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
