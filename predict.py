import argparse
import os
import numpy as np
from arc import ARCSolver, ARCDataset
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
import json
import random

WORKSPACE = "/home/top321902/code/intro_dl/term_project"

def grid_accuracy(prediction, ground_truth):
    """
    Calculate grid accuracy - all cells must match to be considered correct
    """
    pred_shape = prediction.shape
    true_shape = np.array(ground_truth).shape
    
    # If shapes are different, return 0
    if pred_shape != true_shape:
        return 0
    
    # Check if all cells match
    return np.array_equal(prediction, ground_truth)

def visualize_example(task_id, test_input, ground_truth, prediction):
    """Visualize the results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input grid
    axes[0].imshow(test_input, cmap='viridis', vmin=0, vmax=9)
    axes[0].set_title('Input')
    for i in range(len(test_input)):
        for j in range(len(test_input[0])):
            axes[0].text(j, i, str(test_input[i][j]), 
                         ha='center', va='center', color='white')
    
    # Ground truth
    true_array = np.array(ground_truth)
    axes[1].imshow(true_array, cmap='viridis', vmin=0, vmax=9)
    axes[1].set_title('Ground Truth')
    for i in range(len(true_array)):
        for j in range(len(true_array[0])):
            axes[1].text(j, i, str(true_array[i][j]), 
                         ha='center', va='center', color='white')
    
    # Prediction
    pred_array = np.array(prediction)
    axes[2].imshow(pred_array, cmap='viridis', vmin=0, vmax=9)
    axes[2].set_title('Prediction')
    for i in range(len(pred_array)):
        for j in range(len(pred_array[0])):
            axes[2].text(j, i, str(pred_array[i][j]), 
                         ha='center', va='center', color='white')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{task_id}_visualization.png")
    plt.close()

def cell_accuracy(prediction, ground_truth):
    """
    Calculate cell-wise accuracy - average accuracy of each cell
    """
    pred_np = np.array(prediction).flatten()
    true_np = np.array(ground_truth).flatten()
    
    # If shapes are different, truncate to the smaller one
    min_len = min(len(pred_np), len(true_np))
    pred_np = pred_np[:min_len]
    true_np = true_np[:min_len]
    
    # Calculate cell-wise accuracy
    return accuracy_score(true_np, pred_np)

def main():
    parser = argparse.ArgumentParser(description='Evaluate ARCSolver on ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default=f"{WORKSPACE}/dataset", 
                        help='Dataset path')
    parser.add_argument('--num_examples', type=int, default=50, 
                        help='Number of examples to evaluate')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize predictions')
    args = parser.parse_args()
    
    solver = ARCSolver(token=args.token)
    solver.prepare_evaluation()
    
    dataset = ARCDataset(args.dataset, solver=solver)
    all_tasks = dataset.all_tasks
    
    random.seed(42)
    
    num_tasks_to_evaluate = min(args.num_examples, len(all_tasks))
    print(f"Evaluating {num_tasks_to_evaluate} examples...")
    
    selected_tasks = random.sample(all_tasks, num_tasks_to_evaluate)
    
    results = []
    grid_accuracies = []
    cell_accuracies = []
    
    for i, task in enumerate(selected_tasks):
        print(f"Evaluating example {i+1}/{num_tasks_to_evaluate}...")
        task_id = task["task_id"]

        sampled_examples = random.sample(task["examples"], 4)
        train_examples = sampled_examples[:3]
        test_example = sampled_examples[3]
        
        test_input = test_example['input']
        ground_truth = test_example['output']
        
        try:
            prediction = solver.predict(train_examples, test_input)
            
            g_acc = grid_accuracy(prediction, ground_truth)
            c_acc = cell_accuracy(prediction, ground_truth)
            
            grid_accuracies.append(g_acc)
            cell_accuracies.append(c_acc)
            
            result = {
                "task_id": task_id,
                "grid_accuracy": g_acc,
                "cell_accuracy": c_acc,
                "input_shape": np.array(test_input).shape,
                "ground_truth_shape": np.array(ground_truth).shape,
                "prediction_shape": np.array(prediction).shape,
            }
            
            results.append(result)
            
            print(f"--- Task ID: {task_id} ---")
            print(f"Grid Accuracy: {g_acc:.2f}")
            print(f"Cell Accuracy: {c_acc:.2f}")
            print(f"Input Shape: {result['input_shape']}")
            print(f"Ground Truth Shape: {result['ground_truth_shape']}")
            print(f"Prediction Shape: {result['prediction_shape']}")
            
            if args.visualize:
                visualize_example(task_id, test_input, ground_truth, prediction)
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            continue
        
    if results:
        avg_grid_accuracy = np.mean(grid_accuracies)
        avg_cell_accuracy = np.mean(cell_accuracies)

        print("\n===== Evaluation Results =====")
        print(f"Number of evaluated examples: {len(results)}")
        print(f"Average Grid Accuracy: {avg_grid_accuracy:.2f}")
        print(f"Average Cell Accuracy: {avg_cell_accuracy:.2f}")
        
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_results.json", "w") as f:
            json.dump({
                "num_examples": len(results),
                "avg_grid_accuracy": avg_grid_accuracy,
                "avg_cell_accuracy": avg_cell_accuracy,
                "results": results
            }, f, indent=4)
        
        print("Results saved to results/evaluation_results.json")
    else:
        print("No valid results to save.")

if __name__ == "__main__":
    main()