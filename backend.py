from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('maildetect.pkl')
vectorizer = joblib.load('vectorizer1.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("frontend.html")

@app.route('/predict', methods=['POST'])
def predict():
    mail_text = request.form.get('mail_content')

    if not mail_text:
        return render_template("frontend.html", prediction="Please enter some text.")

    try:
        tokenize_email = vectorizer.transform([mail_text])
        prediction = model.predict(tokenize_email)[0]

        prediction_label = "Spam" if prediction == 0 else "Not Spam"

        return render_template("frontend.html", prediction=f"The mail is: {prediction_label}")

    except Exception as e:
        return render_template("frontend.html", prediction=f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
