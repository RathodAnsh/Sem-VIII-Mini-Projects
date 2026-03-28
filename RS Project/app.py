from flask import Flask, render_template, request
from models.diet_plan import recommend_meal
from models.content_based import recommend_similar_meals

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')  # or your form page

@app.route('/recommend', methods=['POST'])
def recommend():
    name = request.form['name']
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    disease = request.form['disease']

    # ✅ Call your ML function
    bmi, bmi_category, breakfast, lunch, dinner = recommend_meal(
        age, weight, height, disease
    )

    # 🔥 Convert string → list (IMPORTANT)
    breakfast_list = [x.strip() for x in breakfast.split(",")]
    lunch_list = [x.strip() for x in lunch.split(",")]
    dinner_list = [x.strip() for x in dinner.split(",")]

    # 🔥 Combine for content-based filtering
    input_meal = breakfast + " " + lunch + " " + dinner
    alt_meals = recommend_similar_meals(input_meal)

    alt = alt_meals.iloc[0]

    # 🔥 Convert alt meals also
    alt_breakfast = [x.strip() for x in alt["Breakfast"].split(",")]
    alt_lunch = [x.strip() for x in alt["Lunch"].split(",")]
    alt_dinner = [x.strip() for x in alt["Dinner"].split(",")]

    # ✅ FINAL DIET PLAN
    diet_plan = {
        "bmi": round(bmi, 2),
        "bmi_category": bmi_category,

        "breakfast": breakfast_list,
        "lunch": lunch_list,
        "dinner": dinner_list,

        "alt_breakfast": alt_breakfast,
        "alt_lunch": alt_lunch,
        "alt_dinner": alt_dinner
    }

    user_profile = {
        "name": name,
        "age": age,
        "weight": weight,
        "height": height,
        "disease": disease
    }

    return render_template(
        "diet_recommendation.html",
        diet_plan=diet_plan,
        user_profile=user_profile
    )

if __name__ == "__main__":
    app.run(debug=True)