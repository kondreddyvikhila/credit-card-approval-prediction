from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
pipeline = joblib.load("saved_models/pipeline.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            # User Input
            input_df = pd.DataFrame([{

                "CODE_GENDER": request.form["gender"],
                "FLAG_OWN_CAR": request.form["car"],
                "FLAG_OWN_REALTY": request.form["realty"],
                "CNT_CHILDREN": int(request.form["children"]),
                "AMT_INCOME_TOTAL": float(request.form["income"]),
                "NAME_INCOME_TYPE": request.form["income_type"],
                "NAME_EDUCATION_TYPE": request.form["education"],
                "NAME_FAMILY_STATUS": request.form["family"],
                "NAME_HOUSING_TYPE": request.form["housing"],
                "FLAG_MOBIL": 1,
                "FLAG_WORK_PHONE": int(request.form["work_phone"]),
                "FLAG_PHONE": int(request.form["phone"]),
                "FLAG_EMAIL": int(request.form["email"]),
                "OCCUPATION_TYPE": request.form["occupation"],
                "CNT_FAM_MEMBERS": float(request.form["family_members"]),
                "AGE": int(request.form["age"]),
                "YEARS_EMPLOYED": float(request.form["employment"])

            }])

            # Prediction
            prediction = pipeline.predict(input_df)[0]

            # Confidence Score
            probability = pipeline.predict_proba(input_df)[0][1] * 100
            probability = round(probability, 2)

            # Decision
            if prediction == 1:
                result = "Approved"
            else:
                result = "Rejected"

            return render_template(
                "result.html",
                prediction=result,
                probability=probability,
                error=None
            )

        except Exception as e:

            return render_template(
                "result.html",
                prediction="Error",
                probability=0,
                error=str(e)
            )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)