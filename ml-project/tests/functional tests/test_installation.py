import sys
import importlib.util
sys.path.insert(0,"/mnt/d/mle-training/ml-project/src")
import housing_price_prediction


def test_installation():

    import housing_price_prediction.ingest_data
    import housing_price_prediction.score
    import housing_price_prediction.train
    package_name="housing_price_prediction"
    assert importlib.util.find_spec(package_name) is not None
    


