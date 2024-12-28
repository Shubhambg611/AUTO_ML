# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    UPLOAD_FOLDER = 'uploads'
    MODELS_FOLDER = 'models'
    MONGODB_URI = "mongodb://localhost:27017/"
    MONGODB_DB = "automl"
    MYSQL_CONFIG = {
        'user': os.environ.get('MYSQL_USER', 'root'),
        'password': os.environ.get('MYSQL_PASSWORD', 'Shubham@5050'),
        'host': os.environ.get('MYSQL_HOST', 'localhost'),
        'database': os.environ.get('MYSQL_DB', 'automl_users')
    }
    
# app.py
import os
import logging
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from bson import ObjectId
from datetime import timedelta
from config import Config
from flask_login import current_user, login_required

from flask import jsonify, request, render_template, url_for
from flask_login import login_user, current_user
from werkzeug.security import check_password_hash
import logging
import mysql.connector





# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create required directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODELS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Initialize MongoDB connection
mongo_client = MongoClient(Config.MONGODB_URI)
mongo_db = mongo_client[Config.MONGODB_DB]
uploads_collection = mongo_db['uploads']

class User(UserMixin):
    def __init__(self, username, user_data=None):
        self.id = username
        self.user_data = user_data

@login_manager.user_loader
def load_user(username):
    try:
        with mysql.connector.connect(**Config.MYSQL_CONFIG) as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user_data = cursor.fetchone()
            if user_data:
                return User(username=user_data['username'], user_data=user_data)
    except Exception as e:
        logging.error(f"Error loading user: {e}")
    return None

@app.route('/')
def home():
    # If user is already authenticated, redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            organization = request.form.get('organization', '')
            location = request.form.get('location', '')
            username = request.form['username']
            password = generate_password_hash(request.form['password'])

            with mysql.connector.connect(**Config.MYSQL_CONFIG) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (name, email, organization, location, username, password)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (name, email, organization, location, username, password))
                conn.commit()

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            logging.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')

            if not username or not password:
                return jsonify({
                    'success': False,
                    'message': 'Username and password are required'
                }), 400

            with mysql.connector.connect(**Config.MYSQL_CONFIG) as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()

                if user and check_password_hash(user['password'], password):
                    user_obj = User(username=user['username'], user_data=user)
                    login_user(user_obj)
                    return jsonify({
                        'success': True,
                        'redirect': url_for('dashboard')
                    })
                
                return jsonify({
                    'success': False,
                    'message': 'Invalid username or password'
                }), 401

        except Exception as e:
            logging.error(f"Login error: {e}")
            return jsonify({
                'success': False,
                'message': 'An error occurred during login'
            }), 500

    # For GET requests, render the template
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        user_files = uploads_collection.find({"user": current_user.id})
        return render_template('dashboard.html', 
                             username=current_user.id, 
                             uploaded_files=list(user_files))
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        flash('Error loading dashboard.', 'danger')
        return redirect(url_for('home'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('dashboard'))

        file = request.files['file']
        if file.filename == '':
            flash('No selected file.', 'danger')
            return redirect(url_for('dashboard'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            task_type = request.form.get('task_type')
            target_column = request.form.get('target_column')

            df = pd.read_csv(filepath)
            report = process_data(df, task_type, target_column)

            uploads_collection.insert_one({
                "user": current_user.id,
                "filename": filename,
                "filepath": filepath,
                "task_type": task_type,
                "target_column": target_column,
                "status": "completed",
                "report": report,
                "created_at": pd.Timestamp.now().isoformat()
            })

            flash('File processed successfully!', 'success')
        else:
            flash('Invalid file type. Only CSV files are allowed.', 'danger')

    except Exception as e:
        logging.error(f"Upload error: {e}")
        flash('Error processing file.', 'danger')

    return redirect(url_for('dashboard'))

def process_data(df, task_type, target_column):
    report = {}
    try:
        # Basic EDA
        report['eda'] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        # Handle missing values
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Feature engineering
        le = LabelEncoder()
        for col in categorical_columns:
            if col != target_column:
                df[col] = le.fit_transform(df[col])

        if task_type == 'classification' and target_column in categorical_columns:
            y = le.fit_transform(df[target_column])
        else:
            y = df[target_column]

        X = df.drop(target_column, axis=1)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model training and evaluation
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
        elif task_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report['metrics'] = {
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'mse': mean_squared_error(y_test, y_pred)
            }

        # Feature importance
        report['feature_importance'] = dict(
            zip(X.columns, model.feature_importances_)
        )

        # Save model
        model_filename = f"model_{ObjectId()}.joblib"
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
        joblib.dump(model, model_path)
        report['model_path'] = model_path

    except Exception as e:
        logging.error(f"Data processing error: {e}")
        report['error'] = str(e)
        raise

    return report

if __name__ == "__main__":
    app.run(debug=True)