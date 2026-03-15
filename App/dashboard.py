import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("AI Learning Analytics Dashboard")

data = pd.read_csv("../data/student_learning_data.csv")

st.header("Dataset Overview")

st.write(data.head())

st.metric("Total Students", len(data))
st.metric("Average Study Hours", round(data.study_hours.mean(),2))

st.header("Learning Analytics")

fig, ax = plt.subplots()
ax.scatter(data.study_hours, data.final_score)
ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Score")
st.pyplot(fig)

fig2, ax2 = plt.subplots()
ax2.scatter(data.engagement_score, data.final_score)
ax2.set_xlabel("Engagement Score")
ax2.set_ylabel("Final Score")
st.pyplot(fig2)

st.header("AI Prediction")

study_hours = st.slider("Study Hours",0,15)
assignments = st.slider("Assignments Completed",0,10)
attendance = st.slider("Attendance",0,100)
engagement = st.slider("Engagement Score",0,100)

model = joblib.load("../model/student_model.pkl")

prediction = model.predict([[study_hours,assignments,attendance,engagement]])

if st.button("Predict Result"):

    if prediction[0] == 1:
        st.success("Student Likely to PASS")
    else:
        st.error("Student At Risk of FAIL")
