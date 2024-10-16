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

def calc_halluc_rate(model, test_type, data_dir, prec):
    ## Calculate hallucination rate straight
    # Import data
    df_error = pd.read_csv("/".join([data_dir, "_".join([model, test_type,"agent_error.csv"])]))

    # Extract hallucinations
    df_halluc = df_error.iloc[0:9, :]

    # Generate hallucination rate
    halluc_rate = np.sum(df_halluc["Count"].values)/np.sum(df_error["Count"].values)

    # Round
    halluc_rate = round(halluc_rate, prec)

    return(halluc_rate)

def calc_spec_halluc_rate(model, test_type, data_dir, prec):
    ## Calculate hallucination rate straight
    # Import data
    df_error = pd.read_csv("/".join([data_dir, "_".join([model, test_type,"agent_error.csv"])]))

    # Extract hallucinations
    df_halluc = df_error.iloc[0:9, :]

    # Generate hallucination rate
    if any(df_halluc["Count"] > 0):
        tmp_spec_halluc = []
        for j in df_halluc[df_halluc["Count"]>0]["Error"]:
            halluc_agent = j.split(".")[1]
            tmp_spec_halluc.append(df_halluc[df_halluc["Error"]==j]["Count"].values/np.sum(df_error[df_error["Error"].str.contains("."+halluc_agent)]["Count"].values))
        spec_halluc_rate = np.array(tmp_spec_halluc).flatten().mean()
    else:
        spec_halluc_rate = 0

    # Round
    spec_halluc_rate = round(spec_halluc_rate, prec)

    return(spec_halluc_rate)

def calc_conf_dist(model, test_type, data_dir, prec):
    # Import csv
    df_error = pd.read_csv("/".join([data_dir, "_".join([model, test_type,"agent_error.csv"])]))
    df_pred = pd.read_csv("/".join([data_dir, "_".join([model, test_type, "pred.csv"])]))

    # Alternative hallucination rate and specificity
    df_halluc = df_error.iloc[0:9, :]
    if any(df_halluc["Count"] > 0):
        tmp_conf_dist = []
        for j in df_halluc[df_halluc["Count"]>0]["Error"]:
            # Spec halluc rate
            halluc_agent = j.split(".")[1]

            # Confidence
            tmp_halluc_conf = df_pred[(df_pred["Agent"] == "none") & (df_pred["PredAgent"] == halluc_agent)]["ConfAgent"].values
            tmp_acc_conf = df_pred[(df_pred["Agent"] == halluc_agent) & (df_pred["PredAgent"] == halluc_agent)]["ConfAgent"].values
            if len(tmp_acc_conf) == 0:
                tmp_conf_dist.append(np.nan)
            else:
                tmp_conf_dist.append(wasserstein_distance(tmp_acc_conf, tmp_halluc_conf))

        conf_dist = 1/(np.nanmean(np.array(tmp_conf_dist))+1)
    else:
        conf_dist = np.nan

    conf_dist = round(conf_dist, prec)
    
    return(conf_dist)

def calc_prior_corr(model, test_type, data_dir, prec):
    # Import csv
    df_error = pd.read_csv("/".join([data_dir, "_".join([model, test_type,"agent_error.csv"])]))
    df_pred = pd.read_csv("/".join([data_dir, "_".join([model, test_type, "pred.csv"])]))

    # Alternative hallucination rate and specificity 
    df_halluc = df_error.iloc[0:9, :]
    spec_halluc_rate = []
    for agent in ["cap", "camera", "boot", "bird", "cat", "dog", "baby", "woman", "man"]:
        tmp_sum = sum(df_error[df_error["Error"].str.contains("."+agent)]["Count"].values)
        if tmp_sum > 0:
            spec_halluc_rate.append(df_halluc[df_halluc["Error"]=="none."+agent]["Count"].values[0]/tmp_sum)
        else:
            spec_halluc_rate.append(0)
            
    spec_halluc_rate = np.array(spec_halluc_rate)

    if sum(spec_halluc_rate) != 0:
        # Alpha
        alphas = open("/".join([data_dir, "_".join([model, "alpha.txt"])])).readline()
        alphas = alphas.replace(' ', '').replace('[', '').replace(']', '')
        alphas = np.array([int(i) for i in alphas.split(",")]).flatten()[1:]
    
        # Correlation
        prior_corr = np.corrcoef(spec_halluc_rate, alphas)[0,1]
    else:
        prior_corr = np.nan

    prior_corr = round(prior_corr, prec)

    return(prior_corr)