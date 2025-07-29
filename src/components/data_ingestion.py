# for reading the data ,train test split etc,validation data 
import os
import sys
from src.exception import CustomException
from src.logger import logging  # sys /os for importing our custom exception

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # used to create class variable
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    # this is the path we are giving to data ingestion component and its output will be stored 
    # in this path (inside artifcat folder)
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts','data.csv')

# started our class data ingestion , if we only want to define variable then we can use dataclass
# if we some other function in class we use init/constructor path
class DataIngestion:
    def __init__(self):
        # this ingestion_config variable consist of 3 above value as we need these value to initialize the dataingestion
        # as soon as we call this the data will be stored in this variable
        self.ingestion_config=DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        # to read data from the other dataset.
        # we can create mongodb/sql client in utils file and here can read it.
        logging.info("Entered the data ingestion method or component") # for now just (make it simple) read our dataset
        try:
            # here we read from mongodb/sql
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
            

            # we already know the path of train,test,data path so lets create the folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            # done splitting and saving into the folder
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return {
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path, # this info req for data transformation

            }

        except Exception as e:
            raise CustomException(e,sys)
    
# initiating and running it
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))