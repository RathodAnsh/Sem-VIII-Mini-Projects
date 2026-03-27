from flask import Flask, render_template, request
from models.diet_plan import recommend_meal

app = Flask(__name__)

# -------------------------
# HOME PAGE
# -------------------------

@app.route('/')
def home():
    return render_template("index.html")


# -------------------------
# DIET RECOMMENDATION
# -------------------------

@app.route('/recommend', methods=['POST'])
def recommend():

    # Get form data
    name = request.form['name']
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    disease = request.form['disease'].lower()

    print("Selected disease:", disease)

    # Convert height ft → meters
    height_m = height * 0.3048

    # Calculate BMI
    bmi = weight / (height_m ** 2)

    # Call ML recommendation model
    bmi, bmi_category, breakfast, lunch, dinner = recommend_meal(
        age, weight, height, disease, bmi
    )

    # Diet plan dictionary
    diet_plan = {
        "bmi": round(bmi, 2),
        "bmi_category": bmi_category,
        "breakfast": breakfast.split("/"),
        "lunch": lunch.split("/"),
        "dinner": dinner.split("/")
    }

    # User profile dictionary
    user_profile = {
        "name": name,
        "age": age,
        "weight": weight,
        "height": height,
        "disease": disease
    }

    # Send data to template
    return render_template(
        "diet_recommendation.html",
        diet_plan=diet_plan,
        user_profile=user_profile
    )


# -------------------------
# RUN SERVER
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)