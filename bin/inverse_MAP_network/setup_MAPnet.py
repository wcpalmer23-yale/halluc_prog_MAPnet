import os
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import torchvision.transforms.functional as TF

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
lib_dir = "/".join([proj_dir, "lib"])

# Train network
print("------------------------------")
print("SETTING UP NETWORK")

# Load AlexNet with weights
print("Loading AlexNet")
model = torch.hub.load('pytorch/vision:v0.8.0', 'alexnet', pretrained=True)

# Pruning Model
print("Pruning...")
model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
print(model)

# Freeze model weights
print("Freezing...")
for child in model.children():
        for child_of_child in child.children():
            for param in child_of_child.parameters():
                param.requires_grad = False

# Save trained model
print("Saving...")
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save("/".join([lib_dir, "adapted_AlexNet.pt"])) # Save model

# Checking model
print("Checking...")
model = torch.jit.load('/'.join([lib_dir, 'adapted_AlexNet.pt']))
# Check if frozen
frozen = []
for child in model.children():
    for child_of_child in child.children():
        for param in child_of_child.parameters():
            frozen.append(param.requires_grad)
if not all(frozen):
    print("PASS: Weights frozen properly.")
else:
    print("FAIL: Not all weights frozen.")
