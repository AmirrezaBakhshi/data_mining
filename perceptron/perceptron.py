import numpy as np
import argparse
import sys

# ------------------------------------------------------------
def load_data(filename):
    """Load CSV, last column are labels, rest are features."""
    try:
        data = np.loadtxt(filename, delimiter=',')
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)
    X = data[:, :-1]   # all rows, all but last column
    y = data[:, -1]    # labels
    # Convert labels to 0/1 (just in case they are -1/+1)
    y = np.where(y > 0, 1, 0)
    return X, y

# ------------------------------------------------------------
def train_perceptron(X, y, lr, epochs):
    """Train a single perceptron. Returns weights and bias."""
    n_samples, n_features = X.shape
    w = np.random.randn(n_features) * 0.01   # small random weights
    b = 0.0
    
    for epoch in range(1, epochs+1):
        mistakes = 0
        for i in range(n_samples):
            x_i = X[i]
            y_true = y[i]
            # linear combination
            linear_out = np.dot(w, x_i) + b
            y_pred = 1 if linear_out > 0 else 0
            
            if y_pred != y_true:
                # Perceptron weight update
                w += lr * (y_true - y_pred) * x_i
                b += lr * (y_true - y_pred)
                mistakes += 1
                # Debug: print update? (commented)
                # print(f"  updated: w={w}, b={b}")
        
        print(f"Epoch {epoch:3d}: {mistakes} mistake(s)")
        
        # Optional early stop (commented)
        # if mistakes == 0:
        #     print("Converged early!")
        #     break
    
    return w, b

# ------------------------------------------------------------
def predict(X, w, b):
    """Predict labels for feature matrix X."""
    linear = np.dot(X, w) + b
    return (linear > 0).astype(int)

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Perceptron toy implementation")
    parser.add_argument("--data", type=str, required=False, help="CSV file with data")
    parser.add_argument("--features", type=int, help="Number of input features (excluding label)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    
    args = parser.parse_args()
    
    # If no data file provided, show usage and exit
    if not args.data:
        print("Perceptron for binary classification.")
        print("Example: python perceptron.py --data mydata.csv --features 3 --epochs 20 --lr 0.1")
        print("CSV file: all numeric, last column = labels (0/1).")
        sys.exit(0)
    
    # Load data
    X, y = load_data(args.data)
    n_samples, n_features = X.shape
    
    # Check feature count
    if args.features is not None and args.features != n_features:
        print(f"Warning: specified --features {args.features} but data has {n_features} features. Using {n_features}.")
    
    print(f"Data loaded: {n_samples} samples, {n_features} features")
    print(f"Labels: 0: {sum(y==0)}, 1: {sum(y==1)}")
    
    # Train
    w, b = train_perceptron(X, y, lr=args.lr, epochs=args.epochs)
    
    # Evaluate on training set (just for fun)
    y_pred = predict(X, w, b)
    acc = np.mean(y_pred == y)
    print(f"Training accuracy: {acc*100:.2f}%")
    
    # Show simple learned weights
    print(f"Weights: {w}")
    print(f"Bias: {b:.4f}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()