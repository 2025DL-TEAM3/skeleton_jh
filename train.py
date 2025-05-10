import argparse
from arc import ARCSolver, ARCDataset

WORDKSPACE = '/home/top321902/code/intro_dl/term_project'

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default=f"{WORDKSPACE}/dataset", help='Dataset path')
    args = parser.parse_args()
    
    print("Initializing model...")
    solver = ARCSolver(token=args.token)
    
    print("Loading dataset...")
    dataset = ARCDataset(args.dataset, solver=solver)
    
    print("Starting training...")
    # solver.train(dataset)
    
    print("Training completed!")
    
    
if __name__ == "__main__":
    main()