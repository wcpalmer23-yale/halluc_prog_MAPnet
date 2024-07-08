import os
import random
import numpy as np
import torch
from PIL import Image
from clustimage import Clustimage
import matplotlib.pyplot as plt

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
lib_dir = "/".join([proj_dir, "lib"])
conv_dir = "/".join([lib_dir, "conv1"])

# Train network
print("------------------------------")
print("CLUSTERING CONV1")

# Load model
print("Loading Model")
model = torch.hub.load('pytorch/vision:v0.8.0', 'alexnet', pretrained=True)

# Extract weights
print("Extracting weight")
param = list(model.parameters())[0].detach().numpy()

# Save weights to images
print("Saving images")
if not os.path.isdir(conv_dir):
    os.makedirs(conv_dir)

for i in range(param.shape[0]):
    # Extract filter
    #img = np.reshape(param[i, :, :, :], (param.shape[2], param.shape[3], param.shape[1]))
    img = np.rollaxis(np.squeeze(param[i, :, :, :]), 0, 3)
    
    # Rescale
    img = (((img - np.min(img))/(np.max(img) - np.min(img)))*255).astype(np.uint8) 

    # Save to image
    img = Image.fromarray(img)
    img.save("/".join([conv_dir, "conv1_"+str(i)+'.png']))

# Cluster images
print("Clustering images")
random.seed(27)
imgs = ["/".join([conv_dir, "conv1_"+str(i)+'.png']) for i in range(param.shape[0])]
cl = Clustimage(method="pca-hog", grayscale=False, dim=(11, 11), params_hog={"orientation": 8, "pixel_per_cell": (2, 2)}) # 2,2
results = cl.fit_transform(imgs, metric="euclidean", linkage="ward", cluster_space="high", min_clust=3, max_clust=10)
#print(results.keys())
#print(results["img"].shape)
#print(results["filenames"])
#print(results["labels"])

# Create figures
print("Creating figures")
lab = sorted(list(set(results["labels"])))
print(lab)
for l in lab:
    # Extract images
    fnames = results["filenames"][results["labels"]==l]
    imgs = results["img"][results["labels"]==l, :]
    
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
    #fig.savefig("/".join([conv_dir, "clustimage_clust"+str(l)+".png"]))
    #plt.close()


# Plot
print("Plotting")
#cl.clusteval.plot() # silhouette plot
#cl.plot_unique(img_mean=False) # centroid
#cl.scatter(img_mean=False) # t-sne projection
cl.dendrogram()
plt.show()
