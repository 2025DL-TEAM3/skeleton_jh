import argparse
import datetime
from arc import ARCSolver, ARCDataset

WORDKSPACE = '/home/top321902/code/intro_dl/term_project'

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default=f"{WORDKSPACE}/dataset", help='Dataset path')
    args = parser.parse_args()
    
    ##############
    # TODO: change to arguments
    artifacts_dir = f"artifacts"
    
    ##############
    
    print("Initializing model...")
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    checkpoint_save_path = f"{artifacts_dir}/train-{today}"
    solver = ARCSolver(
        token=args.token,
        checkpoint_save_path=checkpoint_save_path,
    )
    
    print("Loading dataset...")
    dataset = ARCDataset(args.dataset, solver=solver)
    
    print(solver.tokenizer.bos_token_id)
    
    print("Starting training...")
    solver.train(
        dataset,
        num_epochs=5,
    )
    
    print("Training completed!")
    
    
if __name__ == "__main__":
    main()