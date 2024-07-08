import os
import warnings
import pandas as pd
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import torchvision.transforms.functional as TF

# Suppress warnings
warnings.filterwarnings("ignore")

def onehot2str(pred, label, names):
    # Convert to index values
    pred = np.argmax(pred, axis=1)
    label = np.argmax(label, axis=1)

    # Translate to accuracy count
    pred = [names[i] for i in pred]
    label = [names[i] for i in label]
    
    return pred, label

def model_mse(pred, label):
    return ((label - pred)**2).mean(axis=1)

def pred_confidence(pred):
    # # Min/Max Normalize
    # npreds, nunits = pred.shape
    
    # max_val = np.max(pred, axis=1)
    # min_val = np.min(pred, axis=1)
    
    # num = pred - min_val.reshape((npreds, 1))*np.ones((1, nunits))
    # denom = (max_val - min_val)
    
    # rescaled_val = np.divide(num, denom.reshape((npreds, 1))*np.ones((1, nunits)))
    
    # # Calculate confidence (1 will always be the max)
    # conf = 1./np.sum(rescaled_val, axis=1)
    
    conf = np.max(softmax(pred, axis=1), axis=1)
    
    return conf

def model_performance(pred, label):
    # Convert tensor to numpy
    pred = pred.numpy()
    label = label.numpy()

    # Separate labels
    pred_scene = pred[:, 0:9]
    pred_agent = pred[:, 9:19]
    pred_position = pred[:, 19:]

    label_scene = label[:, 0:9]
    label_agent = label[:, 9:19]
    label_position = label[:, 19:]

    # Confidence
    #conf_scene = pred_confidence(pred_scene)
    #conf_agent = pred_confidence(pred_agent)

    # Convert to string
    pred_scene, label_scene = onehot2str(pred_scene, label_scene, ["bathroom", "bedroom", "dining-room", "grey-white-room", "kitchen", "living-room", "staircase", "study", "tea-room"])
    pred_agent, label_agent = onehot2str(pred_agent, label_agent, ["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"])

    # MSE
    mse = model_mse(pred_position, label_position)

    # Create dataframe
    #df = pd.DataFrame({'Scene': label_scene, 'Agent': label_agent, 'x': label_position[:, 0], 'z': label_position[:, 1], 
    #                   'PredScene': pred_scene, 'PredAgent': pred_agent, 'Predx': pred_position[:, 0], 
    #                   'Predz': pred_position[:, 1], 'MSE': mse, 'ConfScene': conf_scene, 'ConfAgent': conf_agent})
    df = pd.DataFrame({'Scene': label_scene, 'Agent': label_agent, 'x': label_position[:, 0], 'z': label_position[:, 1], 
                       'PredScene': pred_scene, 'PredAgent': pred_agent, 'Predx': pred_position[:, 0], 
                       'Predz': pred_position[:, 1], 'MSE': mse})

    return df

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        # Classifier
        self.fc1 = nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=args.n_scenes+args.n_agents+2, bias=True)

    def forward(self, x):
        y = self.fc2(self.relu1(self.fc1(x)))
        return y


class Confidence(nn.Module):
    def __init__(self, args):
        super(Confidence, self).__init__()
        
        # Confidence
        self.fc = nn.Linear(args.n_agents, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conf = self.sigmoid(self.fc(x))
        return conf
   
