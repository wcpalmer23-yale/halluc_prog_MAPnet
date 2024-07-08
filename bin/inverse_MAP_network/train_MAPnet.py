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
from inv_util import Classifier, Confidence

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iter', type=str, required=True, help="iteration name")
parser.add_argument('-d', '--dataset', type=str, required=True, help="dataset name")
parser.add_argument('-m', '--model', type=str, required=True, help="name of pretrained model or 'new' if new model")
parser.add_argument('-l', '--lr', type=float, required=True, help="model learning rate")
parser.add_argument('-e', '--n_epochs', type=int, required=True, help="number of training epochs")
parser.add_argument('-s', '--n_scenes', type=int, required=True, help="number of scenes")
parser.add_argument('-a', '--n_agents', type=int, required=True, help="number of agents")
args = parser.parse_args()

# Extract arguments
nscenes = args.n_scenes
nagents = args.n_agents
iteration = args.iter
dataset = args.dataset
nmodel = args.model

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
lib_dir = "/".join([proj_dir, "lib"])
img_dir = "/".join([proj_dir, "images"])
data_dir = "/".join([img_dir, iteration, dataset])
train_dir = "/".join([data_dir, "train"])
val_dir = "/".join([data_dir, "valid"])

# Train network
if (os.path.isfile("/".join([data_dir, dataset+"_classif.pt"])) and os.path.isfile("/".join([data_dir, dataset+"_conf.pt"])) and os.path.isfile("/".join([data_dir, dataset+"_training_loss.csv"]))):
    print("------------------------------")
    print("NETWORK TRAINED")
    print("Model:", nmodel)
    print("Dataset:", dataset)
else:
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
    
    # Build Model
    ## AlexNet until FC1
    Alexnet = torch.jit.load('/'.join([lib_dir, "adapted_AlexNet.pt"]))
    
    ## Use pretrained model if not new
    if nmodel != 'new':
        print("Loading Model")
        Classifier = torch.jit.load('/'.join([img_dir, iteration, nmodel, nmodel+'_classif.pt']))
        class_out = Classifier.to(device)
        Confidence = torch.jit.load('/'.join([img_dir, iteration, nmodel, nmodel+'_conf.pt']))
        conf_out = Confidence.to(device)
    else:
        print("Creating Model")
        class_out = Classifier(args).to(device)
        conf_out = Confidence(args).to(device)
    alexnet = Alexnet.to(device)
    
    ## Combine modules
    all_modules = nn.ModuleList([alexnet, class_out, conf_out])
    
    frozen = []
    for child in alexnet.children():
        for child_of_child in child.children():
            for param in child_of_child.parameters():
                frozen.append(param.requires_grad)
    print(frozen)
    
    #Optimizer
    optimizer = optim.Adam(all_modules.parameters(), lr=args.lr)
    
    # Train the model
    print("Training Model")
    tloss_store = []
    vloss_store = []
    for epoch in range(args.n_epochs):  # loop over the dataset multiple times
        print("-----", "Epoch:", epoch, "-----")
        alexnet.train()
        class_out.train()
        conf_out.train()
        running_tloss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # parse inputs
            scene_label = labels[:, 0:nscenes]
            agent_label = labels[:, nscenes:(nscenes+nagents)]
            pos_label = labels[:, (nscenes+nagents):]
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward pass through classifier
            output = alexnet(inputs)
            output = class_out(output)
    
            # parse output
            scene_pred = output[:, 0:nscenes]
            agent_pred = output[:, nscenes:(nscenes+nagents)]
            pos_pred = output[:, (nscenes+nagents):]

            # forward pass through confidence
            conf = conf_out(agent_pred).squeeze()

            # agent correct?
            _, agent_pred_argmax = agent_pred.max(1)
            _, agent_label_argmax = agent_label.max(1)
            correct_agent = torch.eq(agent_pred_argmax, agent_label_argmax).type(torch.float)
    
            # loss
            ## classifier
            scene_loss = nn.CrossEntropyLoss()(scene_pred, scene_label)
            agent_loss = nn.CrossEntropyLoss()(agent_pred, agent_label)
            pos_loss = nn.MSELoss()(pos_pred, pos_label)
            ## confidence
            conf_loss = nn.BCELoss()(conf, correct_agent) 
            
            # propogate loss backwards
            loss = scene_loss + agent_loss + pos_loss + conf_loss
            loss.backward()
    
            # optimize
            optimizer.step()
    
            # print training statistics
            running_tloss += (scene_loss.item() + agent_loss.item() + pos_loss.item() + conf_loss.item())
    
        # log average loss
        avg_tloss = running_tloss/(i + 1)
        tloss_store.append(avg_tloss)
    
        # validation set performance
        alexnet.eval()
        class_out.eval()
        conf_out.eval()
        running_vloss = 0.0
    
        with torch.no_grad():
            for i, vdata in enumerate(valloader):
                # Split validation data
                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
    
                # parse inputs
                scene_vlabel = vlabels[:, 0:nscenes]
                agent_vlabel = vlabels[:, nscenes:(nscenes+nagents)]
                pos_vlabel = vlabels[:, (nscenes+nagents):]
                    
                # forward pass through classifier
                voutput = alexnet(vinputs)
                voutput = class_out(voutput)
    
                # parse output
                scene_vpred = voutput[:, 0:nscenes]
                agent_vpred = voutput[:, nscenes:(nscenes+nagents)]
                pos_vpred = voutput[:, (nscenes+nagents):]
    
                # forward pass through confidence
                vconf = conf_out(agent_vpred).squeeze()
    
                # agent correct?
                _, agent_vpred_argmax = agent_vpred.max(1)
                _, agent_vlabel_argmax = agent_vlabel.max(1)
                correct_vagent = torch.eq(agent_vpred_argmax, agent_vlabel_argmax).type(torch.float)
    
                # loss
                ## classifier
                scene_vloss = nn.CrossEntropyLoss()(scene_vpred, scene_vlabel)
                agent_vloss = nn.CrossEntropyLoss()(agent_vpred, agent_vlabel)
                pos_vloss = nn.MSELoss()(pos_vpred, pos_vlabel)
                ## confidence
                conf_vloss = nn.BCELoss()(vconf, correct_vagent)
    
                # add to total loss
                running_vloss += scene_vloss.item() + agent_vloss.item() + pos_vloss.item() + conf_vloss.item()
            
            # log average loss
            avg_vloss = running_vloss / (i + 1)
            vloss_store.append(avg_vloss)
        print(vconf, correct_vagent)
        print("Epoch:", epoch, ": Average Training Loss =", avg_tloss, ", Average Valid Loss = ", avg_vloss)
    print('Finished Training of Model')

    # Save trained model
    print("Saving Model")
    class_scripted = torch.jit.script(class_out)
    class_scripted.save("/".join([data_dir, dataset+"_classif.pt"]))
    conf_scripted = torch.jit.script(conf_out)
    conf_scripted.save("/".join([data_dir, dataset+"_conf.pt"]))
    
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

