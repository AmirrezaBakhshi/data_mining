# Single-Layer Perceptron for Binary Classification

A simple implementation of a perceptron (single-layer neural network) with one output node. Trains on linearly separable data using the perceptron learning rule.

## Files

- `perceptron.py` – trains a perceptron on a CSV dataset.
- `gen_data.py` – generates a random linearly separable dataset for testing.

## Requirements

- Python 3.6+
- No external libraries (uses only `argparse`, `csv`, `random`)

## Usage
```bash
python gen_data.py --samples 200 --features 3 --output mydata.csv
python perceptron.py --data mydata.csv --features 3 --epochs 20 --lr 0.1
