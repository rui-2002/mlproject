import sys
import os
from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # used to create pipeline
from sklearn.impute import SimpleImputer # used for handling misssing values
from sklearn.pipeline import Pipeline #for implementing pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# create dataingestion for taking input
@dataclass
class DataTransformationConfig:
    #preprocessor object file path : to save any model in pickle file for that we req a path
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") # we can also name as model.pkl

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''

        # used to create pkl file
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "test_preparation_course"
            ]

            #created pipeline doing 2 step : handle missing value and doing standard scaling
            # this pipeline need to be run on train dataset.
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]

            )


            cat_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")




            logging.info(f"Categorical columns :{categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor=ColumnTransformer(
                [
                    # pipeline name,what pipeline it is,our column
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )


            return preprocessor





        except Exception as e:
            raise CustomException(e,sys)
    

    # starting our data transformation/(function)
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')


            preprocessor_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")
            # we have created prprocessing obj and this canbe/need to be converted into pkl file
            # for this we haven't convereted it we have only taken the path (given in datactranformation config : preprocesser.pkl)
            # utils in src (it will have common things that we are trying to import or use)
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path




        except Exception as e:
            raise CustomException(e,sys)
            