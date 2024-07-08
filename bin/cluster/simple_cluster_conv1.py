import os
import random
import numpy as np
import torch
from PIL import Image
from clustimage import Clustimage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
lib_dir = "/".join([proj_dir, "lib"])
conv_dir = "/".join([lib_dir, "conv1"])
filenames = ["conv1_"+str(i)+".png" for i in range(64)]

# Train network
print("------------------------------")
print("CLUSTERING CONV1")

# Load model
print("Loading Model")
model = torch.hub.load('pytorch/vision:v0.8.0', 'alexnet', pretrained=True)

# Extract weights and orientations
print("Extracting weight")
param = list(model.parameters())[0].detach().numpy()
cl = Clustimage(method="hog", dim=(11, 11), params_hog={"orientation": 8, "pixel_per_cell": (2, 2)})
hog_imgs=np.zeros((param.shape[0], 484))
orig_imgs=np.zeros((param.shape[0], param.shape[1]*param.shape[2]*param.shape[3]))
for i in range(param.shape[0]):
    # Extract filter
    #img = np.reshape(param[i, :, :, :], (param.shape[2], param.shape[3], param.shape[1]))
    img = np.rollaxis(np.squeeze(param[i, :, :, :]), 0, 3)
    
    # Rescale
    img = (((img - np.min(img))/(np.max(img) - np.min(img)))*255).astype(np.uint8)

    # Extract orientation
    img_l = cl.imread("/".join([conv_dir, filenames[i]]), colorscale=0, dim=(11,11))
    img_hog = cl.extract_hog(img_l)
    hog_imgs[i, :] = img_hog.flatten()

    # Store
    orig_imgs[i, :] = img.flatten()

# Concatenating pixel values and orientation
print("Concatenating imgs")
concat_imgs = np.concatenate((orig_imgs, hog_imgs), axis=1)

# Reduce dimensionality
print("Reducing dimensionality")
red_imgs = PCA(n_components=0.95, random_state=0).fit_transform(concat_imgs)
print(red_imgs.shape)

# Learn Manifold
print("Learning manifold")
emb_imgs = TSNE(n_components=2, random_state=0).fit_transform(red_imgs)

# Cluster
print("Clustering")
#labels = KMeans(n_clusters=5, random_state=0).fit_predict(emb_imgs)
#labels = AgglomerativeClustering(n_clusters=5).fit_predict(concat_imgs)
labels = DBSCAN().fit_predict(emb_imgs)

# Create figures
print("Creating figures")
lab = list(set(labels))
print(lab)
for l in lab:

    # Extract images
    fnames = [filenames[i] for i, x in enumerate(labels==l) if x]
    imgs = orig_imgs[labels==l, :]
    
    # Create figure
    if len(imgs)==1:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        img = np.array(Image.open("/".join([conv_dir, fnames[0]])))
        ax.imshow(img)
        ax.set_title(fnames[0].split("_")[-1].replace(".png", ""))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        fig, ax = plt.subplots(1, imgs.shape[0], figsize=(2*len(imgs), 2))
        for i, img in enumerate(imgs):
            # Load image
            fname=fnames[i]
            img = np.array(Image.open("/".join([conv_dir, fname])))

            # Plot
            ax[i].imshow(img)
            ax[i].set_title(fname.split("_")[-1].replace(".png", ""))
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
    fig.savefig("/".join([conv_dir, "dbscan_clust"+str(l)+".png"]))
#plt.show()
