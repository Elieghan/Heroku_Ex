import pandas as pd
import numpy as np
from flask import Flask
from flask import request
import pickle
import os

# Example:
# http://localhost:5000/predict_churn?is_male=1&num_inters=0&late_on_payment=0&age=41&years_in_contract=3.240370

filename = 'churn_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/predict_churn')
def predict_churn():
    k1 = float(request.args.get('is_male'))
    k2 = float(request.args.get('num_inters'))
    k3 = float(request.args.get('late_on_payment'))
    k4 = float(request.args.get('age'))
    k5 = float(request.args.get('years_in_contract'))
    feat = np.array([k1,k2,k3,k4,k5]).reshape(1, -1)
    y_pred_model = model.predict(feat)
    texte = 'not churn'
    if y_pred_model ==1:
        texte = 'churn'
    return f"{y_pred_model[0]}"
    # return f"The result of the churn_prediction is: {y_pred_model[0]} meaning your customer will {texte}"


def main():
    """
    Run the test function and predict_churn
    Author:  Elie Ghanassia
    """
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0',port=int(port))


if __name__ == '__main__':
    main()