import streamlit as st
import pandas as pd
import joblib

st.title("UTS MD - Ferdi Tan Leonardy - Student Placement Model Deployment")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
branch = st.sidebar.selectbox("Branch", ["CSE", "IT", "ECE", "ME", "CE"])
part_time = st.sidebar.radio("Part Time Job?", ["Yes", "No"])
income = st.sidebar.selectbox("Family Income Level", ["Low", "Medium", "High"])
city = st.sidebar.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
internet = st.sidebar.radio("Internet Access?", ["Yes", "No"])
extracurricular = st.sidebar.selectbox("Extracurricular Involvement", ["Low", "Medium", "High", "None"])

with st.form("input_form"):
    st.write("### Data Akademik")
    col1, col2 = st.columns(2)
    
    with col1:
        cgpa = st.number_input("CGPA (0.0 - 10.0)", 0.0, 10.0, 7.5, step=0.1)
        tenth = st.number_input("10th Percentage", 0.0, 100.0, 75.0)
        twelfth = st.number_input("12th Percentage", 0.0, 100.0, 75.0)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        projects = st.number_input("Projects Completed", 0, 10, 2)
        
    with col2:
        internships = st.number_input("Internships Completed", 0, 10, 1)
        coding_rating = st.slider("Coding Skill Rating", 1, 5, 3)
        aptitude_rating = st.slider("Aptitude Skill Rating", 1, 5, 3)
        hackathons = st.number_input("Hackathons Participated", 0, 10, 1)
    
    submitted = st.form_submit_button("Predict Placement Status")

if submitted:
    payload = {
        'gender': gender,
        'branch': branch,
        'cgpa': cgpa,
        'tenth_percentage': tenth,
        'twelfth_percentage': twelfth,
        'backlogs': backlogs,
        'projects_completed': projects,
        'internships_completed': internships,
        'coding_skill_rating': coding_rating,
        'aptitude_skill_rating': aptitude_rating,
        'hackathons_participated': hackathons,
        'part_time_job': part_time,
        'family_income_level': income,
        'city_tier': city,
        'internet_access': internet,
        'extracurricular_involvement': extracurricular
    }

    try:
        model_class = joblib.load('model_classification.pkl')
        model_reg = joblib.load('model_regression.pkl')
        
        df = pd.DataFrame([payload])
        df['cgpa_norm'] = df['cgpa'] * 10 
        df['academic_avg'] = (df['tenth_percentage'] + df['twelfth_percentage'] + df['cgpa_norm']) / 3
        df['total_skill_score'] = df['coding_skill_rating'] + df['aptitude_skill_rating']
        df['experience_score'] = (df['internships_completed'] * 2) + df['projects_completed'] + df['hackathons_participated']
        df['high_risk_backlog'] = (df['backlogs'] > 0).astype(int)
        
        pred_class = model_class.predict(df)[0]
        pred_reg = model_reg.predict(df)[0]
        
        if pred_class == 1:
            st.success("Result: Placed")
            st.info(f"Estimated Salary: {pred_reg:.2f} LPA")
        else:
            st.error("Result: Not Placed")
            
    except FileNotFoundError:
        st.error("Error: model_classification.pkl atau model_regression.pkl tidak ditemukan di folder ini.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")