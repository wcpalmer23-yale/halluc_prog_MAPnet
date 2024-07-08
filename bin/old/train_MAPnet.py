import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import gc
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import torchvision.transforms.functional as TF
from inv_util import class_out, conf_out

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, help="dataset name")
parser.add_argument('-m', '--model', type=str, required=True, help="name of pretrained model or 'new' if new model")
parser.add_argument('-l', '--lr', type=float, required=True, help='model learning rate')
parser.add_argument('-e', '--n_epochs', type=int, required=True, help='number of training epochs')
args = parser.parse_args()

# Extract arguments
nscenes = 2
nagents = 6
dataset = args.dataset
nmodel = args.model

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
img_dir = "/".join([proj_dir, "images"])
data_dir = "/".join([img_dir, dataset])
train_dir = "/".join([data_dir, "train"])
val_dir = "/".join([data_dir, "valid"])

# Train network
print("------------------------------")
print("TRAINING NETWORK")
print("Model:", nmodel)
print("Dataset:", dataset)

# Load filse
train_file = "/".join([train_dir, "labels.csv"])
val_file = "/".join([val_dir, "labels.csv"])

# Load images and labels
print("Loading Data")
# dtype_dict = {"living_room": np.float32, "bedroom": np.float32, 
#             "none": np.float32, "cat": np.float32, "dog": np.float32, "man": np.float32, "woman": np.float32, "baby": np.float32, 
#             "x": np.float32, "z": np.float32}
# train_df = pd.read_csv(train_file, dtype=dtype_dict)
# val_df = pd.read_csv(val_file, dtype=dtype_dict)

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)

predvars = [col for col in train_df.columns][1:]
npreds = len(predvars)

# Images
train_images = torch.stack([TF.to_tensor(Image.open('/'.join([train_dir, i])).resize((224, 224), resample=Image.Resampling.LANCZOS)) for i in train_df["image"]])
val_images = torch.stack([TF.to_tensor(Image.open('/'.join([val_dir, i])).resize((224, 224), resample=Image.Resampling.LANCZOS)) for i in val_df["image"]])

# Labels
train_labels = np.array(train_df[predvars].values, dtype=np.float32)
train_labels = torch.stack([torch.from_numpy(l) for l in train_labels])

val_labels = np.array(val_df[predvars].values, dtype=np.float32)
val_labels = torch.stack([torch.from_numpy(l) for l in val_labels])

# Create data
train = data_utils.TensorDataset(train_images, train_labels)
valid = data_utils.TensorDataset(val_images, val_labels)

# Create dataloader
trainloader = data_utils.DataLoader(train, batch_size=20, shuffle=False)
valloader = data_utils.DataLoader(valid, batch_size=20, shuffle=False)

# PyTorch Setup
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AlexNet with weights
if nmodel == 'new':
    print("Creating Model")
    model = torch.hub.load('pytorch/vision:v0.8.0', 'alexnet', pretrained=True)

    # Edit classifier to appropriate output size
    model.classifier[4] = nn.Linear(4096,1024)
    model.classifier[6] = nn.Linear(1024,npreds)
else:
    print("Loading Model")
    model = torch.jit.load('/'.join([img_dir, nmodel, nmodel+'.pt']))

# Freeze convlutional layers (feature)
child_counter = 0
for child in model.children():
        for child_of_child in child.children():
            if child_counter < 14:
                for param in child_of_child.parameters():
                    param.requires_grad = False
            child_counter += 1

# Check if frozen
# for child in model.children():
#         for child_of_child in child.children():
#                 print("-----------------------------------------")
#                 print(child_of_child)
#                 for param in child_of_child.parameters():
#                     print(param.requires_grad)

# Push model to device
model.to(device)

#Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Train the model
print("Training Model")
tloss_store = []
vloss_store = []
for epoch in range(args.n_epochs):  # loop over the dataset multiple times
    print("-----", "Epoch:", epoch, "-----")
    model.train(True)
    running_tloss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(inputs)

        # optimize
        scene_loss = nn.CrossEntropyLoss()(output[:, 0:nscenes], labels[:, 0:nscenes])
        agent_loss = nn.CrossEntropyLoss()(output[:, nscenes:(nscenes+nagents)], labels[:, nscenes:(nscenes+nagents)])
        position_loss = nn.MSELoss()(output[:, (nscenes+nagents):], labels[:, (nscenes+nagents):])

        loss = scene_loss + agent_loss + position_loss
        loss.backward()

        # optimize
        optimizer.step()

        # print training statistics
        running_tloss += (scene_loss.item() + agent_loss.item() + position_loss.item())

    # log average loss
    avg_tloss = running_tloss/(i + 1)
    tloss_store.append(avg_tloss)

    # validation set performance
    model.eval()
    running_vloss = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(valloader):
            # Split validation data
            vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
            
            # Extract model predictions
            voutput = model(vinputs)

            # Calculate loss
            scene_vloss = nn.CrossEntropyLoss()(voutput[:, 0:nscenes], vlabels[:, 0:nscenes])
            agent_vloss = nn.CrossEntropyLoss()(voutput[:, nscenes:(nscenes+nagents)], vlabels[:, nscenes:(nscenes+nagents)])
            position_vloss = nn.MSELoss()(voutput[:, (nscenes+nagents):], vlabels[:, (nscenes+nagents):])

            # add to total loss
            vloss = scene_vloss.item() + agent_vloss.item() + position_vloss.item()
            running_vloss += vloss
        
        # log average loss
        avg_vloss = running_vloss / (i + 1)
        vloss_store.append(avg_vloss)
    print("Epoch:", epoch, ": Average Training Loss =", avg_tloss, ", Average Valid Loss = ", avg_vloss)
print('Finished Training of Model')

# Save trained model
print("Saving Model")
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save("/".join([data_dir, dataset+".pt"])) # Save

# Plot Loss
# plt.plot(tloss_store, label="Training")
# plt.plot(vloss_store, label="Validation")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend() 
# plt.show()

# Save loss
print("Saving Loss")
loss_df = pd.DataFrame({'TLoss': tloss_store, "VLoss": vloss_store})
loss_df.to_csv("/".join([data_dir, dataset+"_training_loss.csv"]), index=False)

# Release gpu
#gc.collect()
#gc.collect()
#torch.cuda.empty_cache()

