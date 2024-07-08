import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning) # pandas warning pyarrow
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr, multinomial
import matplotlib.pyplot as plt
from sklearn import linear_model

proj_dir = "/home/wcp27/project/halluc_prog_MAPnet"

def mdl2dist(model):
    if "CL" in model:
        test_type="clean"
    elif "BL" in model:
        test_type="blurred"
    elif "NO" in model:
        test_type="noisy"
    elif "CO" in model:
        test_type="color"
    elif "CE" in model:
        test_type="cedge"
    elif "ED" in model:
        test_type="edge"
    elif "CX" in model:
        test_type="complex"
    elif "MI" in model:
        test_type="mixed"
    else:
        print("missing test type")
    
    return(test_type)

def calc_error_rate(model, test_type, iteration, proj_dir):
    # Set data directory
    data_dir = "/".join([proj_dir, "images", iteration, model])

    # Import csv
    tmp_agent = pd.read_csv("/".join([data_dir, "_".join([model, test_type,"agent_error.csv"])]))

    # Remove hallucionations
    halluc_idx = sum([1 for i in tmp_agent["Error"] if i.startswith("none.")])
    df_error = tmp_agent.iloc[halluc_idx:, :]

    # Calculate error rates
    error_rate = sum(df_error["Count"].values)/(len(df_error)*100)

    return(error_rate)

def conf_bin(conf, bins = np.arange(11)/10, normalize = False):
    count = []
    for i in range(len(bins)-1):
        if i == len(bins)-2:
            count.append(np.sum(np.logical_and(conf >= bins[i], conf <= bins[i+1])))
        else:
            count.append(np.sum(np.logical_and(conf >= bins[i], conf < bins[i+1])))
    count = np.array(count)
    
    if np.sum(count) != len(conf):
        print(count, conf)
        print("Binning error!")
    
    if normalize:
        count = count/len(conf)
        if not np.isclose(np.sum(count), 1):
            print(np.sum(count))
            print("Normalizing Error")
    
    return(count)
def log_lik_ratio(acc_probdist, err_probdist, halluc_count, normalize = False):
    # Calculate log likelihoods
    acc_l = multinomial.pmf(halluc_count, np.sum(halluc_count), acc_probdist)
    err_l = multinomial.pmf(halluc_count, np.sum(halluc_count), err_probdist)

    if acc_l == 0 or err_l == 0: # log(0) undefined
        ll_ratio = np.nan
    else:
        acc_ll = np.log(acc_l)
        err_ll = np.log(err_l)

        # Log-likelihood Ratio
        ll_ratio = acc_ll - err_ll

    # Normalize if requested
    if normalize:
        ll_ratio = ll_ratio/np.sum(halluc_count)

    return(ll_ratio)
    
def mulinomial_var(count):
    # Calculate Multinomial Variance (n*p_i*(1-p_i))
    ## Convert to probability
    prob = count/np.sum(count)
    
    ## Prob Fail
    inv_prob = 1 - prob
    
    ## Variance
    var = count*prob*inv_prob
    tot_var = np.sum(var)

    return(var, tot_var)