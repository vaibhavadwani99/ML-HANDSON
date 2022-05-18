import sys 
sys.path.insert(0,"/mnt/d/mle-training/src/housing_price_prediction")
import score
import unittest 
 
class Test_score(unittest.TestCase):
    def test_pars_args(self):
        self.assertEqual(score.args.dataset_folder,"/mnt/d/mle-training/data/processed/test.csv")

    def test_score(self):
        method=score.scores()
        method.results()
        assert method.r2 is not None
        assert method.rmse is not None 
        self.assertEqual(method.df.empty,False)
        
if __name__=="__main__":
    unittest.main()