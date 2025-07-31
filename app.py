from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler
# to use pkl file

application=Flask(__name__)
# Flask(__name__) gives us entry point where we need to execute it

app=application

## Route for a home page
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html') # return default homepage not index.html (it include simple datafield provided to model to do prediction)
    
    else:
        
        # if it is a post req (we will start creating our data) i.e create our own custom class
        # here same predict_pipeline will be created here.
        # capture data,standerscaling data then do the prediction
        
        # creating our custiom data(reading all the values) i.e this post will have entrie information
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        # fn from predict_pipeline
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
    
        predict_pipeline=PredictPipeline()
        # given (pred_df) dataframe in predict function
        # it will got to predict fn in predict_pipeline then transformation will happen
        results=predict_pipeline.predict(pred_df)
        #returning values ing index.html
        # result[0] because all the values will be in list format
        return render_template('index.html',results=results[0])



# to run python file

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)

    # go to http://127.0.0.1:5000/



