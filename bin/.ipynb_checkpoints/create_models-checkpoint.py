# Create Models
######################################################################

# Import libraries
import os
import pandas as pd

# Set directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
bin_dir = "/".join([proj_dir, "bin"])
lib_dir = "/".join([proj_dir, "lib"])

# Load models
df_models = pd.read_csv("/".join([lib_dir, "models.csv"]))

# Create models
for i, row in df_models.iterrows():
    
    # Make directory
    os.makedirs("/".join([lib_dir, row["model"]]))

    # Create files
    ## n_train.txt
    n_train = open("/".join([lib_dir, row["model"], "n_train.txt"]), "w")
    n_train.write("\n".join(["45000", "11250", "11250", "11250", "11250"]))
    n_train.close()

    ## lr.txt
    lr = open("/".join([lib_dir, row["model"], "lr.txt"]), "w")
    lr.write("\n".join(["0.0001", "0.0001", "0.0001", "0.0001", "0.0001"]))
    lr.close()

    ## n_epoch.txt
    n_epoch = open("/".join([lib_dir, row["model"], "n_epoch.txt"]), "w")
    n_epoch.write("\n".join(["75", "75", "75", "75", "75"]))
    n_epoch.close()

    ## dval.txt
    dval = open("/".join([lib_dir, row["model"], "dval.txt"]), "w")
    if row["progression"] == 1:
        if row["dval"] == 0.15:
            dval_lst = ["0.15", "0.25", "0.35", "0.45", "0.55"]
        else:
            dval_lst = ["2", "4", "6", "8", "10"]
    else:
        dval_lst = [str(row["dval"]) for i in range(5)]
    dval.write("\n".join(dval_lst))
    dval.close()

    ## conf.txt
    conf = open("/".join([lib_dir, row["model"], "conf.txt"]), "w")
    conf_lst = [str(row["confidence"]) for i in range(5)]
    conf.write("\n".join(conf_lst))
    conf.close()

    # Write Slurm
    sl = open("/".join([bin_dir, "run_"+row["model"]+".sh"]), "w")
    sl.write("\n".join(["#!/bin/bash", "#SBATCH --job-name="+row["model"], 
                        "#SBATCH --partition=gpu", "#SBATCH --cpus-per-task=1", 
                        "#SBATCH --mem=80G", "#SBATCH --gpus=1", "#SBATCH --time=1-00:00:00", 
                        "module load miniconda CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 Julia/1.9.3-linux-x86_64",
                        "conda activate generative", 
                        " ".join(["./model.sh", row["model"], "0", "4", '"[10000, 10000, 10000, 10000, 10000, 10000]"', row["distortion"], str(row["expectation"])])]))
    sl.close()
    