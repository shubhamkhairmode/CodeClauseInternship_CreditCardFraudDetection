import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        # Redirect to the route for a normal transaction
        return redirect(url_for('show_normal_transaction', prediction_text='This transaction appears to be legitimate and does not show signs of fraudulent activity'))
    else:
        # Redirect to the route for a fraud transaction
        return redirect(url_for('show_fraud_transaction', prediction_text='This transaction has been flagged for potential fraudulent activity. Further investigation is recommended.'))

@app.route('/show_normal_transaction')
def show_normal_transaction():
    prediction_text = request.args.get('prediction_text')
    return render_template('normal_transaction.html', prediction_text=prediction_text)

@app.route('/show_fraud_transaction')
def show_fraud_transaction():
    prediction_text = request.args.get('prediction_text')
    return render_template('fraud_transaction.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run()
