# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

 ## installation 
 A .whl file (read: wheel file) is a zip archive that contains all the files necessary to run a Python application.
1) Open your Windows command line or Powershell.
2) cd into the folder where the yourPackage.whl file resides.
3) Optional: Install pip on Windows. Chances are that it’s already installed—it comes with many Python distributions.
   Run the following command:
   pip install housing_price_prediction.whl

## To excute the script
python < scriptname.py >
## To excute the script.
- python3 src/HousePricePrediction/ingest_data.py -h to see argument options.
- python3 src/HousePricePrediction/ingest_data.py --ingest_data_path INGEST_DATA_PATH --log_level LOG_LEVEL --log_path LOG_PATH --no_console_log

- python3 src/HousePricePrediction/train.py -h to see argument options.
- python3 src/HousePricePrediction/train.py  --dataset DATASET --output_model_path OUTPUT_MODEL_PATH --log_level LOG_LEVEL --log_path LOG_PATH --no_console_log

- python3 src/HousePricePrediction/score.py -h to see argument options.
- python3 src/HousePricePrediction/score.py --model_folder MODEL_FOLDER --dataset_folder DATASET_FOLDER --output_folder OUTPUT_FOLDER --log_path LOG_PATH

