from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import credentials, db
import sys,os

# Update the path to your uploaded file if needed
sys.path.append(r"C:\javaPrgm\dup")

from combine import model

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/update_rating', methods=['POST'])
def update_rating():
    email_to_update = request.form['email_to_update']
    model(email_to_update)

@app.route('/stop')
def stop():
    print("Stopping the program...")
    os._exit(0) 
if __name__ == '__main__':
    app.run(debug=True)
