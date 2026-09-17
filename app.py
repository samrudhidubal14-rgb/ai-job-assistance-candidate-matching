import streamlit as st
import pandas as pd
import joblib

# Load trained model
MODEL_FILE = "AI_Job_Assistance_Candidate_Matching_Model.pkl"

model = joblib.load(MODEL_FILE)
