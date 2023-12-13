import numpy as np
import pandas
import pickle
from flask import Flask,url_for,render_template,jsonify,request,app

app = Flask(__name__)

regmodel = pickle.load(open ('regmodel.pkl','rb'))
scaler = pickle.load(open ('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.get_json('data')
    print(data)
    data_value = np.array(list(data.values())).reshape(-1,1)
    print(data_value)
    new_data = scaler.transform(data_value)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == '__main__':
    app.run(debug=True)