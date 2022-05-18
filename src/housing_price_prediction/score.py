''' this module evaluates the performance of the module on the test data
by calculating the perforamance parameters like root mean square value(rmse_)
and r2 value'''
import argparse
import logging
import os
from os.path import dirname as up
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import numpy as np
import sys

sys.path.insert(0,"/mnt/d/mle-training")

parser=argparse.ArgumentParser()
parser.add_argument("--model_folder",default="/mnt/d/mle-training/artifacts/models/",type=str,help="provide the model folder")
parser.add_argument("--dataset_folder",default="/mnt/d/mle-training/data/processed/test.csv",type=str,help="provide dataset folder")
parser.add_argument("--output folder",default="/mnt/d/mle-training/artifacts",type=str,help="provide the output folder")
parser.add_argument("--log_level", type = str,default="DEBUG",help="specifiy level of debug")
parser.add_argument("--log_path", type = str,help="specify the path where to save log file")
parser.add_argument("--no_console_log", type = str,default=None,help="specify to log on console or not")
args=parser.parse_args()
# creating the logger object 
logger=logging.getLogger(__name__)
#specifing the format we want to log
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(funcName)s - %(lineno)d - %(message)s')
# level setting of log 
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

class scores:
    ''' this class calculates the rmse and r2 score of model predictions of test data'''
    def __init__(self):
        ''' this function initializes the args and logger object inside the class'''
        self.args=args
        self.logger=logger
        
    def results(self):
        '''this function finds the r2 and rmse value of every model for its performance of test data set'''
        self.df = pd.read_csv(self.args.dataset_folder)
        self.Y = self.df[["median_house_value"]]
        self.X = self.df.drop("median_house_value",axis=1)
        self.rmse = None
        self.r2 = None
        for filename in os.scandir(self.args.model_folder):
            with open(filename , 'rb') as f:
                model = pickle.load(f)
                pred = model.predict(self.X.values)
                mse_ = mse(pred,self.Y.values)
                self.rmse = np.sqrt(mse_)
                self.r2 = r2_score(pred,self.Y.values)
                self.logger.debug("rmse %d",self.rmse)
                self.logger.debug("r2 %f",self.r2)
obj = scores()
obj.results()
            


            