#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/04/09 08:38:04
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   None
'''

# here put the import lib
import os
import math
import time
import torch
import logging
import numpy as np
from abc import ABCMeta
from torch.distributions import Categorical, normal, MultivariateNormal

def normalise_rewards(rewards):
    """Normalises rewards to mean 0 and standard deviation 1"""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8) #1e-8 added for stability

def create_actor_distribution(action_types, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution

def load(check_point_name,model_storage_path,ARGS):
    '''Load model from .pt file.  
    Args:  
        check_point_name(str):   Filename to store best model
        model_storage_path(str): Folder path to store best model
    Returns:
        model(nn.Module):        Loaded model
    '''
    save_path = os.path.join(model_storage_path+str(ARGS.gamename), check_point_name)
    model = build_model(ARGS)
    model.load_state_dict(torch.load(save_path))
    return model

def save(model_best, checkpoint_name,model_storage_path,gen):
    '''save model into .pt file  
    Args:  
        check_point_name(str):   filename to store best model  
        model_storage_path(str): folder path to store best model  
    '''
    save_path = os.path.join(model_storage_path, checkpoint_name+str(gen)+'.pt')
    torch.save(model_best.state_dict(), save_path)
    return save_path

def setup_logging(logger,folder_path,filename,txtlog=True,scrlog=False):
    """Create and init logger. 
    Reset hander.   
    Args:
        logger:          logger class
        logfolder_path:  "/log/"
        filename:        "Alien-phi-0.001-mu-14.txt"
        txtlog(bool):    If True, print log into file
        scrlog(bool):    If True, print log into screen
    Return:
        logger:           
    """
    logfile = os.path.join(folder_path, filename)
    logging.basicConfig(
        level = logging.INFO,
        format ='%(asctime)s - %(levelname)s - %(message)s',
        filename = logfile,
        filemode = 'a'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    if logger.hasHandlers() is True:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    if txtlog is True:
        # handler = logging.FileHandler(logfile,mode='a',encoding='utf-8')
        # handler.setLevel(logging.INFO)
        # handler.setFormatter(formatter) 
        # logger.addHandler(handler) 
        logger.info("Logger initialised.") 
    if scrlog is True:
        console_handler = logging.StreamHandler() 
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter) 
        logger.addHandler(console_handler)

    return logger

def mk_folder(logpath,args):
    """make folder to save log  
    As ./log/Alien/2020-4-9-1"""
    if logpath not in os.getcwd():
        os.mkdir(logpath)
    gamefolder = os.path.join(logpath,args.gamename)
    if gamefolder not in os.listdir(logpath):
        os.mkdir(gamefolder)

    timenow = time.localtime(time.time())
    indx = 1
    logfolder = str(timenow.tm_year)+'-'+str(timenow.tm_mon)+'-'+str(timenow.tm_mday)+'-'+str(indx)
    while logfolder in os.listdir(gamefolder):
        indx += 1
        logfolder = str(timenow.tm_year)+'-'+str(timenow.tm_mon)+'-'+str(timenow.tm_mday)+'-'+str(indx)
    logfolder_path = os.path.join(gamefolder,logfolder)
    os.mkdir(logfolder_path)
    print("make folder:",logfolder_path)
    return logfolder_path
    
    
