import argparse, os
import datetime
from arc import ARCSolver, ARCDataset

WORKSPACE = os.getenv("INTRODL2025_WORKSPACE", "/home/top321902/code/intro_dl/term_project")
print("WORKSPACE:", WORKSPACE)

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default=f"{WORKSPACE}/dataset", help='Dataset path')
    args = parser.parse_args()
    
    ##############
    # TODO: change to arguments
    artifacts_dir = f"{WORKSPACE}/skeleton/artifacts"
    
    ##############
    
    print("Initializing model...")
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_save_path = f"{artifacts_dir}/train-{now}"
    solver = ARCSolver(
        token=args.token,
        checkpoint_save_path=checkpoint_save_path,
    )
    
    print("Loading dataset...")
    dataset = ARCDataset(args.dataset, solver=solver)
        
    print("Starting training...")
    solver.train(
        dataset,
        num_epochs=7,
        batch_size=2,
        num_epochs_for_custom_lm_head=0,
    )
    
    print("Training completed!")
    
    
if __name__ == "__main__":
    main()