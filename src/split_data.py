import argparse
import pandas as pd
import numpy as np
import os
import sys


def normalize(feat):
    """Normalize features to mean 0 and std 1. feat is the data matrix"""

    mean = np.mean(feat, axis=0)
    std = np.std(feat, axis=0) #ecart-type
    std[std == 0] = 1  #avoid division by 0 by replacing by 1
    normalized_feat = (feat-mean)/std #normalization formula
    return normalized_feat, mean, std


def split_data(feat, label, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split features and labels into train, validation, and test sets."""

    np.random.seed(seed)
    index = np.arange(feat.shape[0])
    np.random.shuffle(index)

    n_train = int(train_ratio * len(index))
    n_val = int(val_ratio * len(index))
    #n_test = len(index) - n_train - n_val

    train_idx = index[:n_train]
    val_idx = index[n_train:n_train + n_val]
    test_idx = index[n_train + n_val:]

    return (
        feat[train_idx], label[train_idx],
        feat[val_idx], label[val_idx],
        feat[test_idx], label[test_idx]
    )


def parse_arguments():
    """Parses arguments and manages errors."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        sys.exit(f"Error: Dataset file '{args.dataset}' not found.")

    if not (0 < args.train_ratio < 1) or not (0 < args.val_ratio < 1):
        sys.exit("Error: --train_ratio and --val_ratio must be between 0 and 1.")

    if args.train_ratio + args.val_ratio >= 1:
        sys.exit("Error: The sum of train_ratio and val_ratio must be less than 1 (to leave room for test).")

    if not isinstance(args.seed, int):
        sys.exit("Error: --seed must be an integer.")

    return args


def main():
    args = parse_arguments()

    df = pd.read_csv(args.dataset, header=None)

    #Assign feature names
    df.columns = ['id', 'diag'] + [f'feature_{i}' for i in range(1, 31)]

    #Encode target (M=1, B=0)
    df['diag'] = df['diag'].map({'M': 1, 'B': 0})

    df = df.drop(columns=['id'])

    features = df.drop(columns=['diag']).to_numpy(dtype=float)
    labels = df['diag'].to_numpy(dtype=int)

    #Split
    feat_train, label_train, feat_val, label_val, feat_test, label_test = split_data(features, labels, args.train_ratio, args.val_ratio, args.seed)

    #Normalize
    N_feat_train, mean, std = normalize(feat_train)
    N_feat_val = (feat_val-mean)/std
    N_feat_test = (feat_test-mean)/std


    def save_split(filename, feat, label):
        df_split = pd.DataFrame(feat, columns=[f'feature_{i}' for i in range(1, feat.shape[1] + 1)])
        df_split['diag'] = label
        df_split.to_csv(filename, index=False)

    save_split("../data/data_train.csv", N_feat_train, label_train)
    save_split("../data/data_val.csv", N_feat_val, label_val)
    save_split("../data/data_test.csv", N_feat_test, label_test)

    print("Splits saved successfully.")
    print(f"Train: {feat_train.shape}, Val: {feat_val.shape}, Test: {feat_test.shape}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
