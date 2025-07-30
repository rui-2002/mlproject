import os
import sys
import dill # help to build pkl file

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
## utils will have the common code required
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]  # Use keys, not values!

            # hyperparameter tunning with param(parameters)
            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
             
            # selecting best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            # after hyperparamet training we don't need model.fit()
            #model.fit(X_train,y_train) #Train model

            y_train_pred=model.predict(X_train)

            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,y_train_pred)

            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    


 # opening the path in "read byte mode" and loading the pkl file by using dill
# responsible for loading pkl file
def load_object(file_path):
    try:
        with open(file_path,"rb")as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)