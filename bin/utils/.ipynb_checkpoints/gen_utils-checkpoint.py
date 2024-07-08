import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
import torch
mi.set_variant('cuda_ad_rgb')

def load_scene(room, agent):
    # file name
    file = '/'.join(['../scenes', room, agent+'.xml'])
    
    # load scene
    scene = mi.load_file(file)
    return scene

def transform_scene(scene, agent, x, z, y):
    if agent != "none":
        # Extract parameters from scene
        params = mi.traverse(scene)
        
        # Extract position
        name = agent+'.vertex_positions'
        V = dr.unravel(mi.Point3f, params[agent+'.vertex_positions'])
        
        # Translate agent
        V.x += x # depth
        V.z += z # width
        V.y += y # elevation
        params[agent+'.vertex_positions'] = dr.ravel(V)
        
        # Apply changes
        params.update()
    return scene

def scene_to_float(scene, spp):
    # Render scene
    image = mi.render(scene, spp=spp)
    bitmap = mi.Bitmap(image).convert(srgb_gamma=True)
    
    # Convert to float
    mu = np.array(bitmap)
    return mu

def load_transform_float(room, agent, x, z, y, spp):
    # file name
    file = '/'.join(['../scenes', room, agent+'.xml'])
    
    # load scene
    scene = mi.load_file(file)
    
    if agent != "none":
        # Extract parameters from scene
        params = mi.traverse(scene)
        # Extract position
        name = agent+'.vertex_positions'
        V = dr.unravel(mi.Point3f, params[agent+'.vertex_positions'])
        # Translate agent
        V.x += x # depth
        V.z += z # width
        V.y += y # elevation
        params[agent+'.vertex_positions'] = dr.ravel(V)
        # Apply changes
        params.update()
    
    # Render scene
    image = mi.render(scene, spp=spp)
    bitmap = mi.Bitmap(image).convert(srgb_gamma=True)
    
    # Convert to float
    mu = np.array(bitmap)
    
    # Clear up memory
    #del scene
    #del bitmap
    #gc.collect()
    #gc.collect()
    #torch.cuda.empty_cache()
    return(mu)
    

#def blurr_float_img(image, rad):
#    # Blur using scipy
#    image_r = gaussian_filter(image[:, :, 0], rad)
#    image_g = gaussian_filter(image[:, :, 1], rad)
#    image_b = gaussian_filter(image[:, :, 2], rad) 
#    image = np.dstack((image_r, image_g, image_b))
#    return image

def blurr_float_img(image, sigma):
    # Blurr using skimage
    image = gaussian(image, sigma, channel_axis=2)
    return(image)

def scene_png(scene, spp, dir, file):
    # Render scene
    image = mi.render(scene, spp=spp)
    bitmap = mi.Bitmap(image).convert(srgb_gamma=True)
    
    # Convert image to uint8
    bitmap = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
    
    # Check directory
    if not os.path.isdir(dir):
        os.makedirs(dir)
    
    # Save image
    bitmap.write('/'.join([dir, file]))
    return None

def array_png(array, dir, file):
    # Render scene
    bitmap = mi.Bitmap(array)
    
    # Convert image to uint8
    bitmap = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
    
    # Check directory
    if not os.path.isdir(dir):
        os.makedirs(dir)
    
    # Save image
    bitmap.write('/'.join([dir, file]))
    
    # Clear up memory
    #del bitmap
    #gc.collect()
    #gc.collect()
    #torch.cuda.empty_cache()
    return None
