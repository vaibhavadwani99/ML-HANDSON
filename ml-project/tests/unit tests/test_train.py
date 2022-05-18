import sys 
sys.path.insert(0,"/mnt/d/mle-training/ml-project/src/housing_price_prediction")
import train 
import unittest 

class Test_Train(unittest.TestCase):
    def test_parse_args(self):
        self.assertEqual(train.args.dataset,"/mnt/d/mle-training/ml-project/data/processed/train.csv")
        self.assertEqual(train.args.output_model_path,"/mnt/d/mle-training/ml-project/artifacts/models/")
    
    def test_linear_model(self):
        method=train.Model()
        method.Linear_Model_train()
        assert "medain_house_value" not in method.X
        # checking loaded dataset is not empty
        self.assertEqual(method.df.empty,False)
        # checking model is saved or not
        self.assertEqual(method.saved,1)

    def test_decision_tree_model(self):
        method=train.Model()
        method.DesTree_Model_Train()
        assert "medain_house_value" not in method.X
        # checking loaded dataset is not empty
        self.assertEqual(method.df.empty,False)
        # checking model is saved or not
        self.assertEqual(method.saved,1)
    
    def test_random_forest_model(self):
        method=train.Model()
        method.RanFor_Model_Train()
        assert "medain_house_value" not in method.X
        # checking loaded dataset is not empty
        self.assertEqual(method.df.empty,False)
        # checking model is saved or not
        self.assertEqual(method.saved,1)

if __name__=="__main__":
    unittest.main()

        
