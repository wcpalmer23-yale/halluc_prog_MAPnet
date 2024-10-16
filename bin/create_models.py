# Create Models
######################################################################

# Import libraries
import os
import pandas as pd

# Set directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
bin_dir = "/".join([proj_dir, "bin"])
lib_dir = "/".join([proj_dir, "lib"])

# Create param csv
#param = open("/".join([bin_dir, "model_params.csv"]), "w")
#param.close()

# Load models
df_models = pd.read_csv("/".join([lib_dir, "model_specifications_lr.csv"]))

# Create models
for i, row in df_models.iterrows():
    
    # Make directory
    os.makedirs("/".join([lib_dir, row["model"]]))

    # Create files
    ## n_train.txt
    n_train = open("/".join([lib_dir, row["model"], "n_train.txt"]), "w")
    n_train.write("\n".join(["45000", "9000", "9000", "9000", "9000"]))
    n_train.close()

    ## lr.txt
    lr = open("/".join([lib_dir, row["model"], "lr.txt"]), "w")
    lr_lst = [str(row["lr"]) for i in range(5)]
    lr.write("\n".join(lr_lst))
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
            dval_lst = ["2.0", "4.0", "6.0", "8.0", "10.0"]
    else:
        dval_lst = [str(row["dval"]) for i in range(5)]
    dval.write("\n".join(dval_lst))
    dval.close()

    ## conf.txt
    conf = open("/".join([lib_dir, row["model"], "conf.txt"]), "w")
    conf_lst = [str(row["confidence"]) for i in range(5)]
    conf.write("\n".join(conf_lst))
    conf.close()

    ## model_params.csv
    #param = open("/".join([bin_dir, "model_params.csv"]), "a")
    #param.write(",".join([row["model"], row["distortion"], str(row["expectation"])])+"\n")
    #param.close()

## Write Slurm
#sl = open("/".join([bin_dir, "submit_models.sh"]), "w")
#sl.write("\n".join(["#!/bin/bash", "#SBATCH --job-name=vh_dcm", 
#                    "#SBATCH --partition=gpu", "#SBATCH --cpus-per-task=1", 
#                    "#SBATCH --mem=32G", "#SBATCH --gpus=1", 
#                    "#SBATCH --time=2-00:00:00", "#SBATCH --array=1-"+str(len(df_models)), 
#                    "module load miniconda Julia/1.9.3-linux-x86_64",
#                    "conda activate generative", 
#                    "./run_models.sh A 0 2")]))
#sl.close()

    
