import argparse
import torch
# a few more files
import numpy as np
import os

def main():
    print(f"Using device: {DEVICE}")
    
    parser = argparse.ArgumentParser(description="Point Cloud Classification - Training and Evaluation")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'visualize'], required=True,
                        help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, help='Path to the model file for evaluation')
    args = parser.parse_args()

    if args.mode == 'train':
        losses = train()
        np.save('training_losses.npy', losses)
        plot_training_progress(losses)
    elif args.mode == 'evaluate':
        if args.model_path is None:
            model_path = os.path.join(MODELS_DIR, 'combined_model.pth')
        else:
            model_path = args.model_path
        evaluate(model_path)
    elif args.mode == 'visualize':
        losses = np.load('training_losses.npy')
        plot_training_progress(losses)

if __name__ == "__main__":
    main()
