import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("../data/student_learning_data.csv")

data["pass"] = (data["final_score"] >= 60).astype(int)

X = data[["study_hours","assignments_completed","attendance","engagement_score"]]
y = data["pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

joblib.dump(model,"student_model.pkl")

print("Model trained and saved.")
