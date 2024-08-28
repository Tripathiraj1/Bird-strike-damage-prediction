from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

# Load the trained model
model = joblib.load('model/bird_strike_model.pkl')

# Load unique values directly within the app
def load_unique_values(column):
    data = pd.read_csv('/Users/utkarshtripathi/Desktop/bird strike /Bird Strikes.csv')
    unique_values = sorted(data[column].dropna().unique())
    return unique_values

# Load the dataset and prepare the unique values for dropdowns
data = pd.read_csv('/Users/utkarshtripathi/Desktop/bird strike /Bird Strikes.csv')
columns_to_remove = ['Record ID', 'Aircraft Type', 'Effect: Indicated Damage', 
                     'Remains of wildlife sent to Smithsonian', 'Remarks', 
                     'Pilot warned of birds or wildlife?', 'Effect Impact to flight', 
                     'Cost Total ', 'Number of people injured', 'Remains of wildlife collected?',
                     'Wildlife Number struck', 'Wildlife Number Struck Actual']
data.drop(columns_to_remove, axis=1, inplace=True)
categorical_columns = data.columns.tolist()

# Preload unique values
unique_values = {column: load_unique_values(column) for column in categorical_columns}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'tripathiii' and password == '123456789':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Extract form data
        form_data = {col: request.form.get(col, '') for col in categorical_columns}
        
        # Extract and handle FlightDate
        flight_date = request.form.get('FlightDate', '')
        form_data['FlightDate'] = flight_date
        
        # Extract and handle Altitude bin
        feet_above_ground = int(request.form.get('Feet_above_ground', 0))
        altitude_bin = '<1000ft' if feet_above_ground < 1000 else '>1000ft'
        form_data['Altitude_bin'] = altitude_bin
        
        # Extract and handle Wildlife Number Struck Bin
        wildlife_number_struck = int(request.form.get('Wildlife_Number_Struck_Actual', 0))
        wildlife_bin = '1' if wildlife_number_struck <= 1 else \
                       '2 to 10' if wildlife_number_struck > 1 and wildlife_number_struck < 11 else \
                       '11 to 100' if wildlife_number_struck >= 11 and wildlife_number_struck < 101 else \
                       'over 100'
        form_data['Wildlife_Number_Struck_Bin'] = wildlife_bin
        
        # Create a DataFrame for the prediction
        input_data = pd.DataFrame([form_data])
        
        # Make prediction with probabilities
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Get the percentage of damage
        damage_probability = prediction_proba[1] * 100  # Assuming 'Damage' is the second class (index 1)
        
        # Format the result
        result = f'{damage_probability:.2f}% chance of damage'
        
        # Pass result and unique values to the result.html page
        return render_template('result.html', result=result, unique_values=unique_values)
    
    return render_template('index.html', unique_values=unique_values)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
