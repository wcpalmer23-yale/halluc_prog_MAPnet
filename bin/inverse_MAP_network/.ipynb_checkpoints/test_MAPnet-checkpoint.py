import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import pandas as pd
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from inv_util import model_performance

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='(str) name of trained model')
parser.add_argument('-s', '--n_scenes', type=int, required=True, help='(int) number of scenes')
parser.add_argument('-a', '--n_agents', type=int, required=True, help='(int) number of agents')
parser.add_argument('-t', '--test_type', type=str, required=True, help='(str) type of distortion')
parser.add_argument('-d', '--dval', type=float, required=True, help='(float) intensity of distortion gain')
parser.add_argument('-c', '--conf', type=float, required=True, help='(float) cutoff confidence for count')
args = parser.parse_args()

# Extract arguments
nmodel = args.model
nscenes = args.n_scenes
nagents = args.n_agents
test_type = args.test_type
dval = args.dval
conf_thr = args.conf

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
lib_dir = "/".join([proj_dir, "lib"])
data_dir = "/".join([proj_dir, "images", nmodel])
test_dir = "/".join([proj_dir, "images", "test"])

# Test network
print("------------------------------")
print("TESTING NETWORK")
print("Model:", nmodel)
print("Testing:", test_type)

# Load filse
test_file = "/".join([test_dir, "labels.csv"])

# Load images and labels
test_df = pd.read_csv(test_file)
predvars = [col for col in test_df.columns][1:]
npreds = len(predvars)

# Images
if test_type in ("color", "edge", "complex", "cedge", "mixed"):
    print("Loading clean data")
    test_images = torch.stack([TF.to_tensor(Image.open('/'.join([test_dir, 'clean_0.05', i])).resize((224, 224), resample=Image.Resampling.LANCZOS)) for i in test_df["image"]])
else:
    type_dir = test_type+"_"+str(dval)
    print("Loading "+type_dir+" data")
    test_images = torch.stack([TF.to_tensor(Image.open('/'.join([test_dir, type_dir, i])).resize((224, 224), resample=Image.Resampling.LANCZOS)) for i in test_df["image"]])

# Labels
test_labels = np.array(test_df[predvars].values, dtype=np.float32)
test_labels = torch.stack([torch.from_numpy(l) for l in test_labels])

# Create data
test = data_utils.TensorDataset(test_images, test_labels)

# Create dataloader
testloader = data_utils.DataLoader(test, batch_size=20, shuffle=False)

# PyTorch Setup
# set the device we will be using to train the model
device = torch.device("cpu")

# Load trained AlexNet with weights
print("Loading Trained Model")
Alexnet = torch.jit.load('/'.join([lib_dir, "adapted_AlexNet.pt"]))
alexnet = Alexnet.to(device)
Classifier = torch.jit.load('/'.join([data_dir, nmodel+'_classif.pt']))
class_out = Classifier.to(device)
Confidence = torch.jit.load('/'.join([data_dir, nmodel+'_conf.pt']))
conf_out = Confidence.to(device)

## Combine modules
all_modules = nn.ModuleList([alexnet, class_out, conf_out])

# Increase gain (if applicable)
if test_type in ("color", "edge", "complex", "cedge", "mixed"):
    # Initialize gain multiplier
    gain = torch.ones([64])
    
    # Double appropriate multiplier
    if test_type == "color":
        gain_lst = [15, 19, 35, 40, 43, 51, 53] # [0, 2, 4, 20, 21, 37, 42, 44, 49, 59]
    elif test_type == "edge":
        gain_lst = [12, 13, 18, 22, 23, 30, 57] # [3, 10, 12, 13, 24, 25, 28, 34, 55, 57]
    elif test_type == "complex":
        gain_lst = [8, 9, 16, 33, 41, 60, 63] # [8, 9, 16, 18, 27, 49, 33, 54, 60, 63]
    elif test_type == "cedge":
        gain_lst = [6, 14, 28, 32, 48, 50, 55]
    elif test_type == "mixed":
        # Draw random set of 10 kernels
        random.seed(23)
        rand_kern = [15, 19, 35, 40, 43, 51, 53] + [12, 13, 18, 22, 23, 30, 57] + [8, 9, 16, 33, 41, 60, 63] + [6, 14, 28, 32, 48, 50, 55]
        rand_kern = random.sample(rand_kern, 7)
        gain_lst = rand_kern
    gain[gain_lst] = dval
        
    
    # Increase gain on Conv1 bias
    for name, param in alexnet.named_parameters():
        if name == "features.0.bias":
            print("Changing Gain")
            param.data = torch.mul(param.data, gain)

# Test the model
print("Testing Model")
alexnet.eval()
class_out.eval()
conf_out.eval()

pred_df = pd.DataFrame(columns = ['Scene', 'Agent', 'x', 'z', 'PredScene', 'PredAgent', 'Predx', 'Predz', 'MSE', "ConfAgent"])
with torch.no_grad():
    for i, data in enumerate(testloader):
        # Split test data
        inputs, labels = data[0].to(device), data[1].to(device)
            
        # Extract model predictions
        output = alexnet(inputs)
        output = class_out(output)

        # Evaluate model
        df = model_performance(output, labels)

        # Extract confidence
        conf = conf_out(output[:, nscenes:(nscenes+nagents)]).squeeze()
        df['ConfAgent'] = conf

        # Store evaluation
        pred_df = pd.concat([pred_df, df], ignore_index = True)

# Append images
pred_df = pd.concat([test_df["image"], pred_df], axis=1)

# Save dataframe
print("Saving Predictions")
pred_df.to_csv("/".join([data_dir, "_".join([nmodel, test_type,"pred.csv"])]), index=False)

# Count predictions
agents = ["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]
count = []
for agent in agents:
    b_correct = pred_df["PredAgent"].values == agent
    b_conf = pred_df["ConfAgent"].values >= conf_thr
    count.append(str(sum(b_correct & b_conf)))

# Write count out to file
f_count = open("/".join([data_dir, "_".join([nmodel, test_type,"pred_nagent.txt"])]), "w")
f_count.write("["+", ".join(count)+"]")
f_count.close()
