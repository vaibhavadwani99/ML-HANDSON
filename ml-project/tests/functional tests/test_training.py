import sys
sys.path.insert(0,"/mnt/d/mle-training/ml-project/src/housing_price_prediction")
import ingest_data,train,score 
import pandas as pd 
import pytest 
import pickle 
import os 

# test that the data is split into test and train and is saved at respective location 
def test_ingest_data():
    dir=ingest_data.args.ingest_data_path
    df_train=pd.read_csv(dir+"train.csv")
    df_test=pd.read_csv(dir+"test.csv")
    assert df_train.empty is False
    assert df_test.empty is False

def test_training():
    obj=train.Model()
    obj.Linear_Model_train()
    assert obj.saved is 1 
    obj.DesTree_Model_Train()
    assert obj.saved is 1
    obj.RanFor_Model_Train()
    assert obj.saved is 1
    path=ingest_data.args.output_model_path
    models=[]
    for filename in os.scandir(path):
        models.append(filename)
        with open(filename,"rb") as f:
            pickle.load(f)

def test_score():
    obj=score.scores()
    obj.reults()
    assert obj.rmse_ is not None 
    assert obj.r2 is not None 




