import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import os
import argparse
import warnings
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, help="dataset name")
args = parser.parse_args()

# Set Directories
dataset = args.dataset
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
image_dir = "/".join([proj_dir, "images"])
data_dir = "/".join([image_dir, dataset])
datafile = "/".join([data_dir, "labels.csv"])

# Test network
print("------------------------------")
if os.path.isdir("/".join([data_dir, "train"])):
    print("ALREADY SPLIT")
    print("Dataset:", dataset)
else:
    print("SPLITTING DATA")
    print("Dataset:", dataset)

    # Load images and labels
    print("Loading Data")
    df = pd.read_csv(datafile)

    # Split images and labels
    colnames = [col for col in df.columns]
    images = df["image"].values
    labels = df[colnames[1:]].values

    # Split data into train and validation/test
    print("Splitting training data")
    im_train, im_val, label_train, label_val = train_test_split(images, labels, test_size=0.2, random_state=23)

    # Move training data
    print("Moving training data")
    train_dir = "/".join([data_dir, "train"])
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    for im in im_train:
        shutil.move("/".join([data_dir, im]), "/".join([train_dir, im]))

    df_train = pd.DataFrame(np.column_stack((im_train, label_train)), columns=colnames)
    df_train.to_csv("/".join([train_dir, "labels.csv"]), index=False)

    # Move validataion data
    print("Moving validation data")
    val_dir = "/".join([data_dir, "valid"])
    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)
    for im in im_val:
        shutil.move("/".join([data_dir, im]), "/".join([val_dir, im]))

    df_val= pd.DataFrame(np.column_stack((im_val, label_val)), columns=colnames)
    df_val.to_csv("/".join([val_dir, "labels.csv"]), index=False)
