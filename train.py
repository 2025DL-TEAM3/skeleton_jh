import argparse, os
import datetime
from arc import ARCSolver, ARCDataset

WORKSPACE = os.getenv("INTRODL2025_WORKSPACE", "/home/top321902/code/intro_dl/term_project")
print("WORKSPACE:", WORKSPACE)

def print_args(args):
    print("--- Arguments ---")
    print(f"Token: {args.token}")
    print(f"Dataset: {args.dataset}")
    print(f"Train name: {args.train_name}")
    print(f"Checkpoint save path: {args.checkpoint_save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default=f"{WORKSPACE}/dataset", help='Dataset path')
    parser.add_argument('--train_name', type=str, default=None, help='Name of the training set')
    args = parser.parse_args()
    
    artifacts_dir = f"{WORKSPACE}/skeleton/artifacts"
    if args.train_name is not None:
        checkpoint_save_path = f"{artifacts_dir}/{args.train_name}"
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        checkpoint_save_path = f"{artifacts_dir}/train-{now}"
    args.checkpoint_save_path = checkpoint_save_path
    
    print_args(args)

    print("Initializing model...")
    solver = ARCSolver(
        token=args.token,
        checkpoint_save_path=args.checkpoint_save_path,
    )
    
    print("Loading dataset...")
    dataset = ARCDataset(args.dataset, solver=solver)
        
    print("Starting training...")
    solver.train(
        dataset,
        num_epochs=5,
        batch_size=4,
    )
    
    print("Training completed!")
    
    
if __name__ == "__main__":
    main()