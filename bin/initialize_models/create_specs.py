# Create Specs
######################################################################

# Import libraries
import os
import pandas as pd

# Set directories
proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"
bin_dir = "/".join([proj_dir, "bin"])
lib_dir = "/".join([proj_dir, "lib"])

# Load distortions
df_distort = pd.read_csv("/".join([lib_dir, "distortion_honing.csv"]))

# Generate specs
store_model = []
store_exp = []
store_dist = []
store_dval = []
store_lr = []
store_nlin = []
for exp in [1, 0]:
    for lr in [0.0001, 0.001]:
        for nlin in [0, 1]:
            for i, row in df_distort.iterrows():
                distort = row["Distort"]

                if distort == "clean":
                    dvals = row[["0.09"]].values
                else:
                    dvals = row[["0.09", "0.1", "0.2"]].values
                
                for dval in dvals:
                    if exp == 0 and nlin == 1:
                        break
                    else:
                        # Model Name
                        ## Model Prefix
                        if distort.startswith("prim"):
                            tmp_pfx = "P"+distort[-1]
                        elif distort == "clean":
                            tmp_pfx = "CL"
                        elif distort == "noisy":
                            tmp_pfx = "NO"
                        elif distort == "blurred":
                            tmp_pfx = "BL"
                        else:
                            print("Distortion NOT FOUND!")
                            tmp_pfx = []
    
                        ## Model dval
                        str_dval = str(dval).split(".")
                        if len(str_dval[0]) < 2:
                            w_num = "0"+str_dval[0]
                        else:
                            w_num = str_dval[0]
    
                        if len(str_dval[1]) < 3:
                            z_pad = "0"*(3-len(str_dval[1]))
                            d_num = str_dval[1]+z_pad
                        else:
                            d_num = str_dval[1]
    
                        ## Model lr
                        if lr == 0.0001:
                            str_lr = "4"
                        elif lr == 0.001:
                            str_lr = "3"
                        else:
                            print("Learning Rate NOT FOUND!")
                            str_lr = []
                        
                        store_model.append("_".join(["A"+str(exp), tmp_pfx+"0", w_num+d_num, str(nlin)+str_lr]))
                        print(store_model[-1])

                        # Expectation
                        store_exp.append(exp)

                        # Distortion
                        store_dist.append(distort)

                        # dval
                        store_dval.append(dval)

                        # lr
                        store_lr.append(lr)

                        #nlin
                        store_nlin.append(nlin)
                        
n_mdls = len(store_model)
store_age = [10000]*n_mdls
store_prog = [0]*n_mdls
store_conf = [0]*n_mdls

# Create dataframe
df_specs = pd.DataFrame({"model": store_model, "age": store_age, "expectation": store_exp, "distortion": store_dist, "dval": store_dval,
                        "progression": store_prog, "confidence": store_conf, "nlin": store_nlin, "lr": store_lr})

# Write to file
df_specs.to_csv("/".join([lib_dir, "model_specs.csv"]), index=False)