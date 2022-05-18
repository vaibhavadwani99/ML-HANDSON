''' this module various machine learning models namely linear regression,
decision tree, random forest on the prepared training data and stores the 
results at default or user defined location.

default location : artifacts/models
'''
import os
import tarfile
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
import argparse 
import logging
import pickle
import sys

#sys.path.insert(0,"/mnt/d/mle-training")

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="/mnt/d/mle-training/ml-project/data/processed/train.csv",help="enter the path to your dataset")
parser.add_argument("--output_model_path",type=str,default="/mnt/d/mle-training/ml-project/artifacts/models/",help="enter the output path to save your model")
parser.add_argument("--log_level", type = str,default="DEBUG",help="specifiy level of debug")
parser.add_argument("--log_path", type = str,help="specify the path where to save log file")
parser.add_argument("--no_console_log", type = str,default=None,help="specify to log on console or not")
args=parser.parse_args()
# creating the logger object 
logger=logging.getLogger(__name__)
# specifying the format of log 
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(funcName)s - %(lineno)d - %(message)s')
# setting the level of logging
if args.log_level:
    if args.log_level == 'DEBUG':
        level=logging.DEBUG
            
    elif args.log_level == 'INFO':
        level=logging.INFO
                
    elif args.log_level == 'ERROR':
        level=logging.ERROR
            
    elif args.log_level == 'WARNING':
        level=logging.WARNING
    else:
        level=logging.CRITICAL
logger.setLevel(level)


if args.log_path:
    file_handler = logging.FileHandler(args.log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

if not args.no_console_log:
    stream_handler = logging.StreamHandler()
    #set the console format
    stream_handler.setFormatter(formatter)
    #add console handler to logger
    logger.addHandler(stream_handler)

class Model:
    ''' the model class creates the model to be trained on the training data '''
    def __init__(self):
        ''' function initializes the args object and logger object inside the class'''
        self.args=args
        self.logger=logger

    def Linear_Model_train(self):
        
        ''' the function trains the linear regression model and save the model pickle file into 
        the user defined or default folder'''
        self.saved =0
        lr = LinearRegression()
        self.logger.debug('running linear regression model')
        self.df = pd.read_csv(self.args.dataset)
        self.Y = self.df[["median_house_value"]]
        self.X = self.df.drop("median_house_value",axis=1)
        lr.fit(self.X.values,self.Y.values)
        os.makedirs(self.args.output_model_path ,exist_ok=True)
        path = self.args.output_model_path
        path = os.path.join(path, "linearmodel.pkl")
        self.logger.debug('saving at  path provided in args %s',path)
        with open(path, "wb") as file:
           pickle.dump(lr, file)
           self.saved = 1
           file.close()
        self.lr = lr
               
    def DesTree_Model_Train(self):
        ''' the function trains the decision tree model and save the model pickle file into 
        the user defined or default folder'''
        self.saved = 0
        dr = DecisionTreeRegressor()
        self.logger.debug('running decision regression model')
        self.df = pd.read_csv(self.args.dataset)
        self.Y = self.df[["median_house_value"]]
        self.X = self.df.drop("median_house_value",axis=1)
        dr.fit(self.X.values,self.Y.values)
        os.makedirs(self.args.output_model_path ,exist_ok=True)
        path = self.args.output_model_path
        path = os.path.join(path, "desmodel.pkl")
        self.logger.debug('saving at  path provided in args %s',path)
        with open(path, "wb") as file:
            pickle.dump(dr, file)
            self.saved = 1
            file.close()
        self.destree = dr

    def RanFor_Model_Train(self):
        ''' the function trains the random forest model and save the model pickle file into 
        the user defined or default folder'''
        self.saved = 0
        rg = RandomForestRegressor()
        self.logger.debug('running Random forest regression model')
        self.df = pd.read_csv(self.args.dataset)
        self.Y = self.df[["median_house_value"]]
        self.X = self.df.drop("median_house_value",axis=1)
        rg.fit(self.X.values,np.ravel(self.Y))
        os.makedirs(self.args.output_model_path ,exist_ok=True)
        path = self.args.output_model_path
        path = os.path.join(path, "regmodel.pkl")
        self.logger.debug('saving at  path provided in args %s',path)
        with open(path, "wb") as file:
            pickle.dump(rg, file)
            self.saved = 1
            file.close()
        self.rg = rg

obj =  Model()
obj.Linear_Model_train()
obj.DesTree_Model_Train()
obj.RanFor_Model_Train()

    

    










