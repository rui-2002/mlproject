import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object # to load our pkl file


class PredictPipeline:
    # by default empty constructor:
    def __init__(self):
        pass

    # load all feature (model predicton file)
    # calling load_obj fun : just import the pkl and load the pk 
    # it will be created in utils 
    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocess_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocess_path)
            # after loading this we need to scale the data i.e transform our features.
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            preds = np.clip(preds, 0, 100)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



# this class is responsible for mapping all the input giving in html to the backend with perticular values
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int,):
        # all the feature that will be used here
        # creating variable through self
        # these values comming from web application (same value mapped from index.html)
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    
    def get_data_as_data_frame(self):
        #input all data return in form of dataframe
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
                
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)