import io
import joblib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import boto3
from face_recognition import preprocessing

# Import or define the Whitening class here (if it’s part of the preprocessing pipeline)
class Whitening:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        return (x - self.mean) / self.std

# Access AWS credentials
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_REGION"]

# Initialize the S3 client using credentials stored in your .env file
s3 = boto3.client('s3',
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=aws_region)

bucket_name = "yashasbucket247"
attendance_file_key = "attendance-data/SE2_september.xlsx"  # S3 key for attendance file
clean_attendance_file_key = "attendance-data/SE2_september_clean.xlsx"  # Key for clean attendance file
model_file_key = "trained_models/frames_trained.pkl"  # S3 key for the face recognition model

# Function to download attendance file from S3
@st.cache_data
def load_attendance_data_from_s3():
    obj = s3.get_object(Bucket=bucket_name, Key=attendance_file_key)
    return pd.read_excel(io.BytesIO(obj['Body'].read()))

# Function to download and load face recognizer model from S3
@st.cache_resource
def load_face_recogniser_model_from_s3():
    model_obj = s3.get_object(Bucket=bucket_name, Key=model_file_key)
    # Ensure custom objects like Whitening are recognized during deserialization
    return joblib.load(io.BytesIO(model_obj['Body'].read()))

divs = ['SE1', 'SE2', 'SE3', 'SE4']
div = st.selectbox("Choose a division", divs)

if div == 'SE2':
    # Load the face recognizer model from S3
    face_recogniser = load_face_recogniser_model_from_s3()
else:
    st.write("Model not trained for this division")
    st.stop()

# Initialize session state for attendance DataFrame
if 'df_attendance' not in st.session_state:
    st.session_state.df_attendance = load_attendance_data_from_s3()

# Initialize session state for tracking date changes
if 'previous_date' not in st.session_state:
    st.session_state.previous_date = None

# Initialize session state for resetting the uploader key
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

preprocess = preprocessing.ExifOrientationNormalize()

# Streamlit interface
st.title("Face Recognition Attendance Application")
