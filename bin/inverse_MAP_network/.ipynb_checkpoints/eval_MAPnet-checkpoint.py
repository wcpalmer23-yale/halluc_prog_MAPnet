import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import pandas as pd

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='(str) name of trained model')
parser.add_argument('-t', '--test_type', type=str, required=True, help='(str) type of distortion')
args = parser.parse_args()

# Extract arguments
nmodel = args.model
test_type = args.test_type

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
data_dir = "/".join([proj_dir, "images", nmodel])

# Evaluate network
print("------------------------------")
print("EVALUATING NETWORK")
print("Model:", nmodel)
print("Testing:", test_type)

# Load files
pred_file = "/".join([data_dir, "_".join([nmodel, test_type,"pred.csv"])])

# Load images and labels
print("Loading data")
pred_df = pd.read_csv(pred_file)

# Count errors
print("Counting errors")
## Scenes
scenes = ["bathroom", "bedroom", "dining-room", "grey-white-room", "kitchen", "living-room", "staircase", "study", "tea-room"]
scene_s = []
scene_n = []

for scene in scenes:
    for pred_scene in scenes:
        if scene != pred_scene: # only interested in errors (i.e. when actual and pred do not match)
            scene_s.append('.'.join([scene, pred_scene]))
            tmp_df = pred_df[(pred_df['Scene'] == scene) & (pred_df['Scene'] == pred_scene)]
            scene_n.append(len(tmp_df)) # counts the number using len()
        
df_scene = pd.DataFrame({'Error': scene_s, 'Count': scene_n})


## Agents
agents = ["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]
agent_s = []
agent_n = []

for agent in agents:
    for pred_agent in agents:
        if agent != pred_agent: # only interested in errors (i.e. when actual and pred do not match)
            agent_s.append('.'.join([agent, pred_agent]))
            tmp_df = pred_df[(pred_df['Agent'] == agent) & (pred_df['PredAgent'] == pred_agent)]
            agent_n.append(len(tmp_df)) # counts the number using len()

df_agent = pd.DataFrame({'Error': agent_s, 'Count': agent_n})

## Hallucination Rate
print("Calculating hallucination rate")
tmp_agent = [a for a in df_agent["Error"].values if a.startswith("none")]
n_halluc = 0
for agent in tmp_agent:
    n_halluc = n_halluc + df_agent[df_agent["Error"] == agent]["Count"].values[0]
n_tot = sum(df_agent["Count"].values)
halluc_rate = n_halluc/n_tot

# Save dataframe
print("Saving errors")
df_scene.to_csv("/".join([data_dir, "_".join([nmodel, test_type,"scene_error.csv"])]), index=False)
df_agent.to_csv("/".join([data_dir, "_".join([nmodel, test_type,"agent_error.csv"])]), index=False)
f = open("/".join([data_dir, "_".join([nmodel, test_type,"halluc_rate.txt"])]), 'w')
f.write(str(halluc_rate))
f.close()
