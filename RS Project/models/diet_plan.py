import numpy as np
import joblib
import os

# Get the path to the project root directory (where the PKL files are)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

print(f"Looking for model files in: {project_root}")

# Load models and encoders safely
try:
    # Load the models from the root directory
    rf_breakfast = joblib.load(os.path.join(project_root, "rf_breakfast.pkl"))
    rf_lunch = joblib.load(os.path.join(project_root, "rf_lunch.pkl"))
    rf_dinner = joblib.load(os.path.join(project_root, "rf_dinner.pkl"))
    label_encoders = joblib.load(os.path.join(project_root, "label_encoders.pkl"))
    print("✅ Model files loaded successfully!")
    
    # Get known diseases from the encoder
    available_diseases = list(label_encoders["Diseases"].classes_)
    print(f"🩺 Available Diseases: {available_diseases}")
    
except FileNotFoundError as e:
    print(f"❌ Model file missing: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Looking in: {project_root}")
    print("Files in root directory:")
    for file in os.listdir(project_root):
        if file.endswith('.pkl'):
            print(f"  - {file}")
    exit()

# Map common disease names to the exact names in the model
disease_mapping = {
    'none': 'none',
    'diabetes': 'diabetes',
    'hypertension': 'hypertension',
    'heart disease': 'heart',
    'heart': 'heart',
    'obesity': 'none'  # if obesity not in model, map to none
}

def calculate_bmi(weight, height_ft):
    """
    Legacy BMI calculation function - kept for reference but no longer actively used.
    The application now retrieves BMI directly from the database.
    """
    # Convert decimal feet directly to meters (1 foot = 0.3048 meters)
    height_m = height_ft * 0.3048
    
    # Calculate BMI
    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        return bmi, "Underweight"
    elif 18.5 <= bmi < 24.9:
        return bmi, "Normal weight"
    elif 25 <= bmi < 29.9:
        return bmi, "Overweight"
    else:
        return bmi, "Obese"

def recommend_meal(age, weight, height_ft, disease, stored_bmi=None):
    """Predict and return meal recommendations using the ML models."""
    # Use the stored BMI value directly - no calculation
    bmi = stored_bmi if stored_bmi is not None else 0
    
    # Determine BMI category based on the value
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    
    # Convert height from decimal feet to meters directly
    height_m = height_ft * 0.3048
    
    # Normalize and map the disease to available options
    disease = str(disease).strip().lower() if disease else "none"
    
    # Find the exact disease string from available diseases
    mapped_disease = 'none'
    if disease != 'none':
        # Find the best match in available diseases
        for available in available_diseases:
            if disease in available.lower():
                mapped_disease = available
                break
    
    print(f"DEBUG - Diet plan: Processing request for disease '{disease}', mapped to '{mapped_disease}'")
    
    # Handle disease encoding
    try:
        if mapped_disease == "none" or mapped_disease not in available_diseases:
            disease_encoded = label_encoders["Diseases"].transform(['none'])[0]
        else:
            disease_encoded = label_encoders["Diseases"].transform([mapped_disease])[0]
    except Exception as e:
        print(f"❌ Error encoding disease: {e}")
        disease_encoded = label_encoders["Diseases"].transform(['none'])[0]
    
    # Prepare input for prediction - make sure types are correct
    user_data = np.array([[
        float(age), 
        float(weight), 
        float(height_m), 
        float(bmi), 
        float(disease_encoded)
    ]])
    
    print(f"DEBUG - Input data: Age={age}, Weight={weight}, Height_m={height_m}, BMI={bmi}, Disease={mapped_disease}, Encoded={disease_encoded}")
    
    try:
        # Make predictions
        breakfast_pred = rf_breakfast.predict(user_data)[0]
        lunch_pred = rf_lunch.predict(user_data)[0]
        dinner_pred = rf_dinner.predict(user_data)[0]
        
        # Decode predictions
        breakfast = label_encoders["Breakfast"].inverse_transform([breakfast_pred])[0]
        lunch = label_encoders["Lunch"].inverse_transform([lunch_pred])[0]
        dinner = label_encoders["Dinner"].inverse_transform([dinner_pred])[0]
        
        # Clean up any special characters or formatting issues
        breakfast = breakfast.replace("\r\n", " ").strip()
        lunch = lunch.replace("\r\n", " ").strip()
        dinner = dinner.replace("\r\n", " ").strip()
        
        print(f"✅ ML Predictions for '{mapped_disease}': Breakfast: {breakfast}, Lunch: {lunch}, Dinner: {dinner}")
        
        return bmi, bmi_category, breakfast, lunch, dinner
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print(f"Input data shape: {user_data.shape}")
        print(f"Input data types: {user_data.dtypes}")
        # Fallback default values if prediction fails
        return bmi, bmi_category, "Oatmeal with fruit", "Grilled chicken salad", "Baked salmon with vegetables"
