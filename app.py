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
    try:
        data = request.get_json()['data']
        print(data)
        
        # Extract values from the dictionary and convert to array
        data_values = [data[key] for key in sorted(data.keys())]  # Assuming keys are 'x2', 'x5', 'x6', 'x8'
        data_array = np.array(data_values).reshape(1, -1)

        print(data_array)
        
        # Assuming 'scaler' is your scaler object
        new_data = scaler.transform(data_array)
        
        # Assuming 'regmodel' is your regression model
        output = regmodel.predict(new_data)
        
        print(output[0])
        
        return jsonify(output[0])

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)