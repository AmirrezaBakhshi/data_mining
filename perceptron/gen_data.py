# Generate random linearly separable data for perceptron demo.
# Usage: python gen_data.py --samples 100 --features 2 --output data.csv

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--features", type=int, default=2)
    parser.add_argument("--output", type=str, default="random_data.csv")
    args = parser.parse_args()
    
    # Random weights for a separating hyperplane (just to make data separable)
    true_w = np.random.randn(args.features)
    true_b = np.random.randn() * 0.5
    
    X = np.random.randn(args.samples, args.features) * 2   # spread out a bit
    # Compute linear values and assign labels 0/1 based on sign
    linear = np.dot(X, true_w) + true_b
    y = (linear > 0).astype(int)
    
    # Stack features and labels, save CSV
    data = np.column_stack((X, y))
    np.savetxt(args.output, data, delimiter=",", fmt="%.6f")
    
    print(f"Generated {args.samples} samples with {args.features} features.")
    print(f"True separator used (not needed for training, just to guarantee separability)")
    print(f"Saved to {args.output}")
    print(f"Now run: python perceptron.py --data {args.output} --features {args.features} --epochs 20")

if __name__ == "__main__":
    main()