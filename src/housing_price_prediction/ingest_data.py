''' The module downloads and creates training and validation dataset
the script accepts output folder or file path from the user as arguments
and the data after processing is dumoed into repective location of training.csv
and test.csv inside the user defined output folder or default folder.


default location of training set : data/processed/training.csv
default location of test set set : data/processed/test.csv
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
import sys 

# sys.path.insert(0,"/mnt/d/mle-training")

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "/mnt/d/mle-training/data/raw"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

parser=argparse.ArgumentParser()
parser.add_argument("--ingest_data_path",type=str,default="/mnt/d/mle-training/data/processed/",help="specify the output folder or file path")
parser.add_argument("--log_level",type=str,default="DEBUG",help="specify the level of log")
parser.add_argument("--log_path",type=str,help="specify the path where you want to save log file")
parser.add_argument("--no_console_log",type=str,default=None,help="specify to write log on console or not")
args=parser.parse_args()

# getting logger object 
logger=logging.getLogger(__name__)
# specifying the format in which we want to log
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(funcName)s - %(lineno)d - %(message)s')
# setting the logging level 
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
    file_handler=logging.FileHandler(args.log__path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

if not args.no_console_log:
    stream_handler=logging.StreamHandler()
    # set the console format
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)




def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    ''' this function fetches the housing data from the remote
    and stores it at the housing path specified

    Parameters:
    ----------
    housing_url : str
        url to download the data
    housing_path : str 
        path to store the downloaded data
    '''

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    logger.debug("extracted data to path [%s]",tgz_path)
    housing_tgz.close()


# import pandas as pd


def load_housing_data(housing_path=HOUSING_PATH):
    ''' thid function loads the dataset from the folder 
    where the data is stored

    Parameters:
    ----------
    housing_path:str
        this the path to your housing.csv file
    
    Returns:
    ------
    pd.DataFrame
    '''
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()
logger.debug("data is loaded")

# from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
logger.debug("applied stratified splitting")

def income_cat_proportions(data):
    ''' this function created the income catrgoried required 
    for stratified splitting of the data.

    Parameters:
    -----------
    data:pd.DataFrame
        data loaded from csv file and converted into a dataframe
    
    Returns:
    -------
    pd.DataFrame
        returns average count of different income categories in the
        form of dataframe
    '''
    
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

# from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)

housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
df2=pd.DataFrame(housing_labels,columns=["median_house_value"])
train_data=pd.concat([housing_prepared,df2],axis=1)

logger.debug("prepared train data")
train_data.to_csv(args.ingest_data_path + "train.csv")
logger.debug("saved data to user entered path or a default path")

# preparing test data
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))
test_data=pd.concat([X_test_prepared,y_test],axis=1)
logger.debug("prepared test data")
test_data.to_csv(args.ingest_data_path + "test.csv")
logger.debug("saved data to user entered path or a default path")
logger.info("data preparation completed")




