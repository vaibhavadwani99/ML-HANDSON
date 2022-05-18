import sys 
sys.path.insert(0,"/mnt/d/mle-training/src/housing_price_prediction")
import ingest_data
import unittest

class Test_Train(unittest.TestCase):
    def test_parse_args(self):
        # checking if stored in default paths
        self.assertEqual(ingest_data.args.ingest_data_path,"/mnt/d/mle-training/data/processed/")
        self.assertEqual(ingest_data.args.log_level,"DEBUG")
    
    def test_load_data(self):
        df=ingest_data.load_housing_data()
        self.assertEqual(df.empty,False,"Data frame is empty")
    
    def test_stratified_split(self):
        assert "income_cat" not in ingest_data.strat_train_set
        assert "income_cat" not in ingest_data.strat_test_set

if __name__=="__main__":
    unittest.main()
    



