import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='(str) name of trained model')
parser.add_argument('-n', '--count', type=int, required=True, help='(int) number of model iterations')
parser.add_argument('-t', '--test_type', type=str, required=True, help='(str) type of test images')
args = parser.parse_args()

# Extract arguments
nmodel = args.model
count = args.count
test_type = args.test_type

# Set Directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
result_dir = "/".join([proj_dir, "results"])

# Models
models = ["_".join([nmodel, str(i)]) for i in range(0, count+1)]
#data_dir = "/".join([proj_dir, "images", nmodel])

# Evaluate network
print("------------------------------")
print("Models:", " ".join(models))
print("Type:", test_type)

# Load files
print("Loading files")
df_scene = pd.DataFrame()
df_agent = pd.DataFrame()
halluc_rate = []
alphas = []
for i, model in enumerate(models):
    # Set data directory
    data_dir = "/".join([proj_dir, "images", model])

    # Import csv
    tmp_scene = pd.read_csv("/".join([data_dir, "_".join([model, test_type,"scene_error.csv"])]))
    tmp_agent = pd.read_csv("/".join([data_dir, "_".join([model, test_type,"agent_error.csv"])]))

    # Extract error types
    if i == 0:
        df_scene["Error"] = tmp_scene["Error"]
        df_agent["Error"] = tmp_agent["Error"]

    # Store values
    df_scene[model] = tmp_scene["Count"]
    df_agent[model] = tmp_agent["Count"]
    halluc_rate.append(float(open("/".join([data_dir, "_".join([model, test_type, "halluc_rate.txt"])])).readline()))
    alphas.append(open("/".join([data_dir, "_".join([model, "alpha.txt"])])).readline())

# Format alphas for plotting
print("Formatting alphas")
alphas = [a.split(',') for a in alphas]
tmp_alphas = np.zeros((count+1, 10))
for i, a in enumerate(alphas):
    for j, b in enumerate(a):
        tmp_alphas[i, j] = int(b.replace(' ', '').replace('[', '').replace(']', ''))
alphas = tmp_alphas

# Plot trajectories
print("Plotting Trajectories")
t_scene = list(sorted(set(df_scene["Error"])))
t_agent = list(sorted(set(df_agent["Error"])))
mod = range(count+1)

## Alpha
fig1, ax = plt.subplots()
ax.plot(alphas, 'v', alpha=0.6)
ax.legend(["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"])
ax.set_xticks([i for i in range(count+1)])
ax.set_xticklabels(mod)
ax.set_xlabel("Time")
ax.set_ylabel("Dirichlet "+r"$\alpha$"+"'s")

## Scenes
fig2, ax = plt.subplots()
for i in range(len(t_scene)):
    y = df_scene[df_scene["Error"]==t_scene[i]][models].values[0]
    ax.plot(mod, y, alpha=0.6)
ax.set_xticks([i for i in range(count+1)])
ax.set_xticklabels(mod)
plt.legend(t_scene)
plt.xlabel("Time")
plt.ylabel("Count")

## Agents
agents = ["none", "cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]
nrow = 2
ncol = int(len(agents)/nrow)
fig3, ax = plt.subplots(nrow, ncol, figsize=(28, 15))

i = 0
for row in range(nrow):
    for col in range(ncol):
        tmp_agent = [a for a in t_agent if a.startswith(agents[i])]
        for agent in tmp_agent:
            y = df_agent[df_agent["Error"] == agent][models].values[0]
            ax[row, col].plot(mod, y, alpha=0.6)
        ax[row, col].legend(tmp_agent)
        ax[row, col].set_xticks([i for i in range(count+1)])
        ax[row, col].set_xticklabels(mod)
        ax[row, col].set_xlabel("Time")
        ax[row, col].set_ylabel("Count")
        i += 1

## Hallucination Rate
fig4, ax = plt.subplots()
ax.plot(halluc_rate)
ax.set_xticks([i for i in range(count+1)])
ax.set_xticklabels(mod)
ax.set_xlabel("Time")
ax.set_ylabel("Hallucination Rate")

# plt.show()

# Save dataframe
print("Saving plots")
traj_dir = "/".join([result_dir, "trajectories"])
if not os.path.isdir(traj_dir):
    os.makedirs(traj_dir)
fig1.savefig("/".join([traj_dir, "_".join([nmodel, test_type, "alpha.png"])]))
fig2.savefig("/".join([traj_dir, "_".join([nmodel, test_type, "scene_error.png"])]))
fig3.savefig("/".join([traj_dir, "_".join([nmodel, test_type, "agent_error.png"])]))
fig4.savefig("/".join([traj_dir, "_".join([nmodel, test_type, "halluc_rate.png"])]))
