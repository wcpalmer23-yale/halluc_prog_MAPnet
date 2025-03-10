import os
import numpy as np

def hone_scenes(agent, room):
    # Order of values
    agent_order = ["cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]
    
    # Scaling values
    agent_scale = [0.11, 0.011, 0.09, 0.16, 0.11, 0.11, 0.33, 0.11, 0.11]

    # file name
    f_xml = '/'.join(['/gpfs/radev/home/wcp27/project/halluc_prog_MAPnet/scenes', room, agent+'.xml'])
    
    # Load xml file
    l_xml = open(f_xml).readlines()

    # Find object
    l_obj = [i for i, l in enumerate(l_xml) if l.startswith('<shape type="obj" id="'+agent+'"')]

    print(l_xml[l_obj[0]])