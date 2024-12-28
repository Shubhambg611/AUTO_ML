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


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    SESSION_COOKIE_NAME = 'automl_session'
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)