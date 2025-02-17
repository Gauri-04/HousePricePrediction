import pandas as pd
from flask  import Flask, render_template , request #, flaskcors
import pickle
import numpy as np


app=Flask(__name__,template_folder='templates')
data=pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl","rb"))

@app.route('/')
def index():
    # data=pd.read_csv('Cleaned_data.csv')
    locations=sorted(data['location'].unique())
    return render_template('index.html', locations=locations)



@app.route('/predict', methods=['GET','POST'])
def predict():
    location= request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    try:
        bhk = float(bhk) if bhk is not None else None
        bath = float(bath) if bath is not None else None
    except ValueError:
        bhk = None
        bath = None
    print(location,bhk,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0]*1e5
    
    return str(np.round(prediction,2))

# @app.errorhandler(405)
# def method_not_allowed(error):
#     return 'Method Not Allowed. Please use a valid HTTP method for this endpoint.', 405


if __name__=="__main__":
    app.run(debug=True, port=5500)

    