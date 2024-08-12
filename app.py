from flask import Flask, render_template, request, jsonify
import joblib
import pickle

app = Flask(__name__)

# Load the pre-trained label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load your trained model
model = pickle.load(open('salary_prediction_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        age = int(request.form['age'])
        years_of_exp = request.form['years_of_exp']
        
        gender = request.form['gender']
        gender = 1 if gender == 'Male' else 0
        
        edu_level = request.form['education_level']
        if edu_level == 'bachelor':
            edu_level_master = 0
            edu_level_phd = 0
        elif edu_level == 'master':
            edu_level_master = 1
            edu_level_phd = 0
        else:  # PhD
            edu_level_master = 0
            edu_level_phd = 1
        
        exp_level = request.form['experience_level']
        if exp_level == 'junior':
            exp_level_mid = 0
            exp_level_sen = 0
        elif exp_level == 'mid':
            exp_level_mid = 1
            exp_level_sen = 0
        else:  # Senior
            exp_level_mid = 0
            exp_level_sen = 1
        
        job_title = request.form.get('jobtitle')
        
        try:
            job_title_encoded = label_encoder.transform([job_title])[0]
        except ValueError:
            return jsonify({'error': 'Job title not recognized'}), 400
        
        # Prepare the feature vector
        features = [[age, years_of_exp, gender, edu_level_master, edu_level_phd,
                             exp_level_mid, exp_level_sen, job_title_encoded]]
        
        # Make a prediction using your model
        prediction = model.predict(features)
        
        
        
        output = round(prediction[0], 2)
        
        print(output)
        
        return render_template('index.html', prediction_text=f"The estimated salary is: ₹{output} per month")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
 