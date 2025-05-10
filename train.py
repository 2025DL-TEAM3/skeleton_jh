import argparse
from arc import ARCSolver, ARCDataset

WORDKSPACE = '/home/top321902/code/intro_dl/term_project'

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default=f"{WORDKSPACE}/dataset", help='Dataset path')
    args = parser.parse_args()
    
    # solver = ARCSolver(token=args.token)
    
    dataset = ARCDataset(args.dataset)
    print(dataset[0])
    
if __name__ == "__main__":
    main()