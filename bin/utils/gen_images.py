# Import packages
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import os
import argparse
import numpy as np
import pandas as pd
from gen_utils import load_transform_float, blurr_float_img, array_png
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
#import gc
#import torch

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='(str) name of trained model')
parser.add_argument('-r', '--row', type=int, required=True, help='(int) row of image to generate')
parser.add_argument('-t', '--test_type', type=str, required=True, help='(str) type of test distortion')
parser.add_argument('-d', '--dval', type=float, required=True, help='(float) distortion intensity')
parser.add_argument('-s', '--spp', type=int, required=True, help='(int) rendering quality') 
args = parser.parse_args()

# Extract arguments
nmodel = args.model
row = args.row
test_type = args.test_type
dval = args.dval
spp = args.spp
dval_clean = 0.03

# Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
img_dir = "/".join([proj_dir, "images"])

# Create images
if "test" in nmodel:
    f_label = "/".join([img_dir, "test/labels.csv"])
elif "examples" in nmodel:
    f_label = "/".join([img_dir, "examples/labels.csv"])
else:
    f_label = "/".join([img_dir, nmodel, "labels.csv"])

if row == 0: 
    print("------------------------------")
    print("Models:", nmodel)
    print("File:", f_label)

# Import csv
df = pd.read_csv(f_label)

# Extract row
row = df.iloc[row]

# Extract info
img = row["image"]
scene = img.split("_")[0]
agent = img.split("_")[1]
x = row["x"]
z = row["z"]

# Check if exists
if os.path.isfile("/".join([img_dir, nmodel, img])):
    ## File exists
    print("Image Exists:", img)
else:
    ## Create image
    ### Room scaling
    if scene == "bathroom":
        x_scale = x*(5.5+5.5)-5.5
        z_scale = z*(7.5+7.5)-7.5
        y_scale = 0
    elif scene == "bedroom":
        x_scale = x*(0.5+0.5)-0.5
        z_scale = z*(0.7+0.7)-0.7
        y_scale = 0
    elif scene == "dining-room":
        x_scale = x*(2.5+2.5)-2.5
        z_scale = z*(1.0+1.0)-1.0
        y_scale = 0
    elif scene == "grey-white-room":
        x_scale = x*(0.45+0.45)-0.45
        z_scale = z*(0.75+0.75)-0.75
        y_scale = 0
    elif scene == "kitchen":
        x_scale = x*(0.8+0.8)-0.8
        z_scale = z*(0.35+0.35)-0.35
        y_scale = 0
    elif scene == "living-room":
        x_scale = x*(0.7+0.7)-0.7
        z_scale = z*(0.6+0.6)-0.6
        y_scale = 0
    elif scene == "staircase":
        x_scale = x*(0.5+0.5)-0.5
        z_scale = z*(0.35+0.35)-0.35
        y_scale = 0
    elif scene == "study":
        x_scale = x*(1.5+1.5)-1.5
        z_scale = z*(2.5+2.5)-2.5
        y_scale = 0
    elif scene == "tea-room":
        x_scale = x*(0.6+0.6)-0.6
        z_scale = z*(1.0+1.0)-1.0
        y_scale = 0
        
    ### Create image
    print("Image Creating:", img)
    mu = load_transform_float(scene, agent, x_scale, z_scale, y_scale, spp)

    ### Apply noise
    if test_type == "blurred":
        mu = blurr_float_img(mu, dval)
        mu = mu + np.random.normal(0, dval_clean, mu.shape)
    else:
        mu = mu + np.random.normal(0, dval, mu.shape)

    ### Save image
    array_png(mu, "/".join([img_dir, nmodel]), img)

# Release gpu
#gc.collect()
#gc.collect()
#torch.cuda.empty_cache()



