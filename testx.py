from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, unset_jwt_cookies
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer
import nltk.chat.util
import tensorflow as tf
from bson import ObjectId
import pickle
import joblib
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from functools import wraps
from flask_cors import CORS

# ====== APP SETUP ======
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "your-secret-key-here"  # Change this to a strong secret key
app.config["MONGO_URI"] = "mongodb://localhost:27017/health_api"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)  # Token expiration time

mongo = PyMongo(app)
jwt = JWTManager(app)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ====== HELPER FUNCTIONS ======
def allowed_file(filename):
    """Check if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_required_fields(data, required_fields):
    """Validate that all required fields are present in the request data."""
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({'error': 'Missing required fields', 'missing': missing}), 400
    return None

def role_required(role):
    """Decorator to check if user has required role."""
    def decorator(f):
        @wraps(f)
        @jwt_required()
        def decorated_function(*args, **kwargs):
            current_user = get_jwt_identity()
            user = mongo.db.users.find_one({"_id": ObjectId(current_user)})
            if not user or user.get('role') != role:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ====== LOAD MODELS ======
def load_models():
    """Load all machine learning models and encoders."""
    models = {
        "sopk": pickle.load(open("models/sopk_randomForest_classifier.pkl", "rb")),
        "diabet": joblib.load("models/diabetes_model.pkl"),
        "hypothyroïdie": pickle.load(open("models/hypothyroïdie_model_xgboost.pkl", "rb")),
        "breastcancer": pickle.load(open("models/breastcancer.pkl", "rb")),
        "endometre": load_model("models/endometre_model.h5"),
        "sopk_advanced": load_model("models/my_model.keras"),
        "obesity": pickle.load(open("models/xgboost_model_obesity.pkl", "rb")),
        
    }
    

    
    # Load encoders
    with open("encoder/label_encoders_obesity.pkl", "rb") as f:
        encoders = pickle.load(f)
    
    return models, encoders

    
models, encoders = load_models()

def load_depression_model():
    """Load depression chatbot model and components"""
    with open("models/Passive_aggressive_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/countvectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    # Define questions
    questions = [
        "Hey,how have you been feeling lately?",
        "Can you tell me about something that's been on your mind recently?",
        "What are some things that bring you joy or make you feel down?",
        "Have you noticed any changes in your mood or energy levels lately?",
        "Is there anything you're struggling with that you'd like to talk about?"
    ]
    
    return model, vectorizer, questions

# Load the depression model
depression_model, depression_vectorizer, depression_questions = load_depression_model()

# Add greeting patterns




# Add depression chat endpoints
@app.route('/api/depression/start', methods=['POST'])
@jwt_required()
def start_depression_chat():
    """Initialize a new depression chat session"""
    return jsonify({
        "question": depression_questions[0],
        "question_index": 0,
        "session_id": str(ObjectId()),  # Generate a new session ID
        "conversation": []
    }), 200

@app.route('/api/depression/chat', methods=['POST'])
@jwt_required()
def depression_chat():
    """Process user responses in the depression chat"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['session_id', 'current_question_index', 'response', 'conversation']
    error_response = validate_required_fields(data, required_fields)
    if error_response:
        return error_response
    


    # Process the conversation
    conversation = data['conversation']
    
    # Check if we have more questions to ask
    next_question_index = data['current_question_index'] + 1
    if next_question_index < len(depression_questions):
        return jsonify({
            "question": depression_questions[next_question_index],
            "current_question_index": next_question_index,
            "conversation": conversation
        }), 200
    
    # If all questions answered, make prediction
    all_responses = " ".join([msg['user'] for msg in conversation if msg['user']])
    
    # Preprocess
    lemmatizer = WordNetLemmatizer()
    words = all_responses.split()
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    processed_text = " ".join(lemmas)
    
    # Vectorize and predict
    input_vector = depression_vectorizer.transform([processed_text])
    prediction = depression_model.predict(input_vector)[0]

    # Format prediction
    prediction = prediction.title()
    
    
    # Save the test result
    test_result = save_test_result(
        user_id=user_id,
        test_type="depression",
        input_data={"conversation": conversation},
        prediction=prediction,
        probability=None
    )
    
    # Prepare response message based on prediction
    if prediction == 'Negative':
        message = "It sounds like you're going through a tough time. Consider reaching out to a mental health professional."
    elif prediction == 'Positive':
        message = "It's good to hear you're doing well. Remember self-care is important."
    else:  # Neutral
        message = "Your responses suggest you're managing. Keep monitoring your mental health."
    
    return jsonify({
        "prediction": prediction,
        "message": message,
        "conversation": conversation + [{"user": None, "bot": message}],
        "test_id": test_result['_id'],
        "completed": True
    }), 200
# ====== USER MANAGEMENT ======
@app.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['email', 'password', 'name']
    error_response = validate_required_fields(data, required_fields)
    if error_response:
        return error_response
    
    # Check if user exists
    if mongo.db.users.find_one({"email": data['email']}):
        return jsonify({"error": "User already exists"}), 409

    # Create user
    hashed_password = generate_password_hash(data['password'])
    user = {
        "email": data["email"],
        "password": hashed_password,
        "name": data["name"],
        "role": "user",  # Default role
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "last_login": None,
        "token": None
    }
    
    # Add optional fields
    optional_fields = ['phone', 'address', 'date_of_birth']
    for field in optional_fields:
        if field in data:
            user[field] = data[field]
    
    # Insert user
    result = mongo.db.users.insert_one(user)
    user_id = str(result.inserted_id)
    
    return jsonify({"message": "User registered successfully!", "user_id": user_id}), 201

@app.route('/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token."""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['email', 'password']
    error_response = validate_required_fields(data, required_fields)
    if error_response:
        return error_response
    
    # Find user
    user = mongo.db.users.find_one({"email": data['email']})
    if not user or not check_password_hash(user['password'], data['password']):
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Create token
    access_token = create_access_token(identity=str(user['_id']))
    
    # Update user
    mongo.db.users.update_one(
        {"_id": user['_id']},
        {"$set": {
            "token": access_token,
            "last_login": datetime.utcnow()
        }}
    )
    
    return jsonify({
        "token": access_token,
        "user_id": str(user['_id']),
        "name": user.get('name'),
        "email": user.get('email'),
        "role": user.get('role', 'user')
    }), 200

@app.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user by invalidating the JWT token."""
    user_id = get_jwt_identity()
    mongo.db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"token": None}}
    )
    response = jsonify({"message": "Logged out successfully"})
    unset_jwt_cookies(response)
    return response, 200

@app.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get the current user's profile information."""
    user_id = get_jwt_identity()
    user = mongo.db.users.find_one(
        {"_id": ObjectId(user_id)},
        {"password": 0, "token": 0}  # Exclude sensitive fields
    )
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    user['_id'] = str(user['_id'])
    return jsonify(user), 200

@app.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update the current user's profile information including password."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Fields that can be updated
    updatable_fields = ['name', 'phone', 'address', 'date_of_birth', 'gender']
    update_data = {}
    
    for field in updatable_fields:
        if field in data:
            update_data[field] = data[field]
    
    # Handle password update separately
    if 'current_password' in data and 'new_password' in data:
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Verify current password
        if not check_password_hash(user['password'], data['current_password']):
            return jsonify({"error": "Current password is incorrect"}), 400
        
        # Hash and update new password
        update_data['password'] = generate_password_hash(data['new_password'])
    
    if not update_data and 'current_password' not in data:
        return jsonify({"error": "No valid fields to update"}), 400
    
    update_data['updated_at'] = datetime.utcnow()
    
    result = mongo.db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_data}
    )
    
    if result.modified_count == 0 and 'current_password' not in data:
        return jsonify({"message": "No changes made"}), 200
    
    return jsonify({"message": "Profile updated successfully"}), 200
# ====== TEST RESULT MANAGEMENT ======
def save_test_result(user_id, test_type, input_data, prediction, probability):
    """
    Save or update a test result in the database.
    If a test of the same type exists for this user, updates it; otherwise creates a new one.
    """
    # Check if this user already has a test of this type
    existing_test = mongo.db.tests.find_one({
        "user_id": ObjectId(user_id),
        "maladie": test_type
    })

    result_data = {
        "user_id": ObjectId(user_id),
        "maladie": test_type,
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.utcnow()
    }

    if existing_test:
        # Update existing test
        result = mongo.db.tests.update_one(
            {"_id": existing_test['_id']},
            {"$set": result_data}
        )
        result_data['_id'] = str(existing_test['_id'])
    else:
        # Insert new test
        result = mongo.db.tests.insert_one(result_data)
        result_data['_id'] = str(result.inserted_id)

    result_data['user_id'] = str(result_data['user_id'])
    return result_data
@app.route('/tests', methods=['GET'])
@jwt_required()
def get_all_tests():
    """Get all test results for the current user."""
    user_id = get_jwt_identity()
    tests = list(mongo.db.tests.find({"user_id": ObjectId(user_id)}))
    
    for test in tests:
        test['_id'] = str(test['_id'])
        test['user_id'] = str(test['user_id'])
    
    return jsonify(tests), 200

@app.route('/tests/<test_type>', methods=['GET'])
@jwt_required()
def get_test_results(test_type):
    """Get all test results of a specific type for the current user."""
    user_id = get_jwt_identity()
    
    # Validate test type
    if test_type not in models:
        return jsonify({"error": "Invalid test type"}), 400
    
    # Get all tests of this type for the user
    tests = list(mongo.db.tests.find({
        "user_id": ObjectId(user_id),
        "maladie": test_type
    }).sort("timestamp", -1))  # Sort by most recent first
    
    for test in tests:
        test['_id'] = str(test['_id'])
        test['user_id'] = str(test['user_id'])
    
    return jsonify(tests), 200

@app.route('/tests/<test_id>', methods=['GET'])
@jwt_required()
def get_test_result(test_id):
    """Get a specific test result by ID."""
    user_id = get_jwt_identity()
    
    try:
        test = mongo.db.tests.find_one({
            "_id": ObjectId(test_id),
            "user_id": ObjectId(user_id)
        })
        
        if not test:
            return jsonify({"error": "Test not found"}), 404
        
        test['_id'] = str(test['_id'])
        test['user_id'] = str(test['user_id'])
        
        return jsonify(test), 200
    except:
        return jsonify({"error": "Invalid test ID"}), 400

@app.route('/tests/<test_id>', methods=['PUT'])
@jwt_required()
def update_test_result(test_id):
    """Update an existing test result."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    try:
        # Get the existing test to determine its type
        existing_test = mongo.db.tests.find_one({
            "_id": ObjectId(test_id),
            "user_id": ObjectId(user_id)
        })
        
        if not existing_test:
            return jsonify({"error": "Test not found"}), 404
        
        test_type = existing_test['maladie']
        
        # Depending on test type, validate and process the input data
        if test_type == "sopk":
            error_response = validate_required_fields(data, SOPK_FEATURES)
            if error_response:
                return error_response
            
            # Process SOPK data (same as in predict_sopk)
            blood_map = {'A-': 11, 'AB-': 12, 'B+': 13, 'O+': 14, 'A+': 15, 'AB+': 16, 'B-': 17, 'O-': 18}
            yn_map = {"Y": 1, "N": 0}
            ri_map = {"R": 1, "I": 0}
            
            cleaned_data = []
            for f in SOPK_FEATURES:
                val = str(data[f]).strip().upper()
                
                if f == "Blood Group":
                    if val not in blood_map:
                        return jsonify({"error": f"Invalid Blood Group: {val}"}), 400
                    cleaned_data.append(blood_map[val])
                elif "(Y/N)" in f:
                    if val not in yn_map:
                        return jsonify({"error": f"Invalid Y/N value for {f}: {val}"}), 400
                    cleaned_data.append(yn_map[val])
                elif "Cycle(R/I)" in f:
                    if val not in ri_map:
                        return jsonify({"error": f"Invalid R/I value for {f}: {val}"}), 400
                    cleaned_data.append(ri_map[val])
                else:
                    cleaned_data.append(float(val))
            
            x = np.array(cleaned_data).reshape(1, -1)
            pred = int(models['sopk'].predict(x)[0])
            proba = float(models['sopk'].predict_proba(x)[0][pred])
            
        elif test_type == "diabet":
            required_fields = ["age", "systolic_bp", "diastolic_bp", "glucose", "bmi", "family_diabetes", "hypertensive"]
            error_response = validate_required_fields(data, required_fields)
            if error_response:
                return error_response
            
            x = np.array([float(data[f]) for f in required_fields]).reshape(1, -1)
            pred = int(models['diabet'].predict(x)[0])
            proba = float(models['diabet'].predict_proba(x)[0][pred])
            
        elif test_type == "breastcancer":
            required_fields = ["menopaus", "agegrp", "density", "race", "Hispanic", "bmi", "agefirst",
                             "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt", "invasive", "training", "count"]
            error_response = validate_required_fields(data, required_fields)
            if error_response:
                return error_response
            
            x = np.array([float(data[f]) for f in required_fields]).reshape(1, -1)
            pred = int(models['breastcancer'].predict(x)[0])
            proba = float(models['breastcancer'].predict_proba(x)[0][pred])
            
        elif test_type == "endometre":
            required_fields = [
                "Heavy / Extreme menstrual bleeding", "Menstrual pain (Dysmenorrhea)",
                "Painful / Burning pain during sex (Dyspareunia)", "Pelvic pain",
                "Irregular / Missed periods", "Cramping", "Abdominal pain / pressure",
                "Back pain", "Painful bowel movements", "Nausea", "Menstrual clots",
                "Infertility", "Painful cramps during period", "Pain / Chronic pain",
                "Diarrhea", "Long menstruation", "Constipation / Chronic constipation",
                "Vomiting / constant vomiting", "Fatigue / Chronic fatigue",
                "Painful ovulation", "Stomach cramping", "Migraines",
                "Extreme / Severe pain", "Leg pain", "Irritable Bowel Syndrome (IBS)",
                "Syncope (fainting, passing out)", "Mood swings", "Depression",
                "Bleeding", "Lower back pain", "Fertility Issues", "Ovarian cysts",
                "Painful urination", "Headaches", "Constant bleeding",
                "Pain after Intercourse", "Digestive / GI problems", "IBS-like symptoms",
                "Excessive bleeding", "Anaemia / Iron deficiency", "Hip pain",
                "Vaginal Pain/Pressure", "Sharp / Stabbing pain", "Bowel pain",
                "Anxiety", "Cysts (unspecified)", "Dizziness", "Malaise / Sickness",
                "Abnormal uterine bleeding", "Fever", "Hormonal problems", "Bloating",
                "Feeling sick", "Decreased energy / Exhaustion",
                "Abdominal Cramps during Intercourse", "Insomnia / Sleeplessness",
                "Acne / pimples", "Loss of appetite"
            ]
            error_response = validate_required_fields(data, required_fields)
            if error_response:
                return error_response
            
            input_data = np.array([float(data[field]) for field in required_fields]).reshape(1, -1)
            probability = models['endometre'].predict(input_data)[0][0]
            pred = 1 if probability >= 0.5 else 0
            proba = float(probability)
            
        elif test_type == "obesity":
            features = [
                "Age", "Height", "Weight", "CALC", "FAVC", "FCVC", "NCP", "SCC", 
                "SMOKE", "CH2O", "family_history_with_overweight", "FAF", "TUE", 
                "CAEC", "MTRANS"
            ]
            error_response = validate_required_fields(data, features)
            if error_response:
                return error_response
            
            encoded_data = []
            for feature in features:
                val = str(data.get(feature, '')).strip()
                
                if feature in encoders:
                    if val not in encoders[feature].classes_:
                        return jsonify({"error": f"Invalid value for {feature}: {val}"}), 400
                    encoded_data.append(encoders[feature].transform([val])[0])
                else:
                    try:
                        encoded_data.append(float(val))
                    except ValueError:
                        return jsonify({"error": f"Invalid numeric value for {feature}: {val}"}), 400
            
            x = np.array(encoded_data).reshape(1, -1)
            pred = int(models["obesity"].predict(x)[0])
            proba = float(models["obesity"].predict_proba(x)[0][pred])
            pred = encoders['NObeyesdad'].inverse_transform([pred])[0]
            
        elif test_type == "hypothyroid":
            required_fields = [
                "Age", "Sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "thyroid_surgery",
                "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium", "goitre",
                "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4", "T4U_measured", "T4U", 
                "FTI_measured", "FTI", "TBG_measured"
            ]
            error_response = validate_required_fields(data, required_fields)
            if error_response:
                return error_response
            
            x = np.array([float(data[f]) if isinstance(data[f], (int, float)) else int(data[f]) 
                         for f in required_fields]).reshape(1, -1)
            pred = int(models["hypothyroïdie"].predict(x)[0])
            proba = float(models["hypothyroïdie"].predict_proba(x)[0][pred])
            
        else:
            return jsonify({"error": "Unsupported test type for update"}), 400
        
        # Update the test result
        result = save_test_result(user_id, test_type, data, pred, proba, test_id)
        if not result:
            return jsonify({"error": "Failed to update test"}), 400
        
        return jsonify({
            "message": "Test updated successfully",
            "prediction": pred,
            "probability": proba,
            "test_id": result['_id']
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/admin/tests', methods=['GET'])
@jwt_required()
@role_required('admin')  # You can define this decorator to check if the user is an admin
def get_all_tests_admin():
    """Get all test results for all users (admin only)."""
    # Ensure the current user is an admin
    
    tests = list(mongo.db.tests.find())  # Retrieve all tests in the database
    
    # Format the test data
    for test in tests:
        test['_id'] = str(test['_id'])
        test['user_id'] = str(test['user_id'])
    
    return jsonify(tests), 200

@app.route('/tests/<test_id>', methods=['DELETE'])
@jwt_required()
def delete_test_result(test_id):
    """Delete a specific test result."""
    user_id = get_jwt_identity()
    
    try:
        result = mongo.db.tests.delete_one({
            "_id": ObjectId(test_id),
            
        })
        
        if result.deleted_count == 0:
            return jsonify({"error": "Test not found"}), 404
        
        return jsonify({"message": "Test deleted successfully"}), 200
    except:
        return jsonify({"error": "Invalid test ID"}), 400

# ====== TEST PREDICTION ENDPOINTS ======
# SOPK Prediction
SOPK_FEATURES = ['Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI', 'Blood Group',
       'Pulse rate(bpm) ', 'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)',
       'Cycle length(days)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)',
       'No. of abortions', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)',
       'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)',
       'Waist:Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
       'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)',
       'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
       'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)',
       'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)',
       'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
       'Endometrium (mm)']

@app.route('/predict/sopk', methods=['POST'])
@jwt_required()
def predict_sopk():
    """Predict SOPK (Polycystic Ovary Syndrome) based on input features."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate required fields
    error_response = validate_required_fields(data, SOPK_FEATURES)
    if error_response:
        return error_response
    
    # Encoding maps
    blood_map = {'A-': 11, 'AB-': 12, 'B+': 13, 'O+': 14, 'A+': 15, 'AB+': 16, 'B-': 17, 'O-': 18}
    yn_map = {"Y": 1, "N": 0}
    ri_map = {"R": 1, "I": 0}
    
    try:
        cleaned_data = []
        for f in SOPK_FEATURES:
            val = str(data[f]).strip().upper()
            
            if f == "Blood Group":
                if val not in blood_map:
                    return jsonify({"error": f"Invalid Blood Group: {val}"}), 400
                cleaned_data.append(blood_map[val])
            elif "(Y/N)" in f:
                if val not in yn_map:
                    return jsonify({"error": f"Invalid Y/N value for {f}: {val}"}), 400
                cleaned_data.append(yn_map[val])
            elif "Cycle(R/I)" in f:
                if val not in ri_map:
                    return jsonify({"error": f"Invalid R/I value for {f}: {val}"}), 400
                cleaned_data.append(ri_map[val])
            else:
                cleaned_data.append(float(val))
        
        # Make prediction
        x = np.array(cleaned_data).reshape(1, -1)
        pred = int(models['sopk'].predict(x)[0])
        proba = float(models['sopk'].predict_proba(x)[0][pred])
        
        # Save result
        result = save_test_result(user_id, "sopk", data, pred, proba)
        
        return jsonify({
            "prediction": pred,
            "probability": proba,
            "test_id": result['_id']
        }), 200
    except ValueError as ve:
        return jsonify({"error": f"Invalid numeric value: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/diabet', methods=['POST'])
@jwt_required()
def predict_diabet():
    """Predict diabetes based on input features."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Required fields for diabetes prediction
    required_fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    # Validate required fields
    error_response = validate_required_fields(data, required_fields)
    if error_response:
        return error_response
    
    try:
        # Prepare input data
        input_data = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]
        
        # Create DataFrame with all expected features (including engineered ones)
        features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
            'BMI_Age', 'Glucose_Insulin', 'BloodPressure_Glucose'
        ]
        
        # Calculate engineered features
        bmi_age = float(data['BMI']) * float(data['Age'])
        glucose_insulin = float(data['Glucose']) * float(data['Insulin'])
        bp_glucose = float(data['BloodPressure']) * float(data['Glucose'])
        
        # Create full feature array
        full_input = input_data + [bmi_age, glucose_insulin, bp_glucose]
        
        # Convert to numpy array and reshape
        x = np.array(full_input).reshape(1, -1)
        
        # Make prediction
        pred = int(models['diabet'].predict(x)[0])
        proba = float(models['diabet'].predict_proba(x)[0][pred])
        
        # Prepare data to save (include original data and engineered features)
        save_data = data.copy()
        save_data['BMI_Age'] = bmi_age
        save_data['Glucose_Insulin'] = glucose_insulin
        save_data['BloodPressure_Glucose'] = bp_glucose
        
        # Save result
        result = save_test_result(user_id, "diabet", save_data, pred, proba)
        
        return jsonify({
            "prediction": pred,
            "probability": proba,
            "test_id": result['_id']
        }), 200
    except ValueError as ve:
        return jsonify({"error": f"Invalid numeric value: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/predict/breastcancer', methods=['POST'])
@jwt_required()
def predict_breast_cancer():
    """Predict breast cancer risk based on input features."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = ["menopaus", "agegrp", "density", "race", "Hispanic", "bmi", "agefirst",
                      "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt", "invasive", ]
    error_response = validate_required_fields(data, required_fields)
    if error_response:
        return error_response
    
    try:
        x = np.array([float(data[f]) for f in required_fields]).reshape(1, -1)
        pred = int(models['breastcancer'].predict(x)[0])
        proba = float(models['breastcancer'].predict_proba(x)[0][pred])
        
        result = save_test_result(user_id, "breastcancer", data, pred, proba)
        
        return jsonify({
            "prediction": pred,
            "probability": proba,
            "test_id": result['_id']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endometre Prediction
@app.route('/predict/endometre', methods=['POST'])
@jwt_required()
def predict_endometre():
    """Predict endometriosis based on input features."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = [
        "Heavy / Extreme menstrual bleeding", "Menstrual pain (Dysmenorrhea)",
        "Painful / Burning pain during sex (Dyspareunia)", "Pelvic pain",
        "Irregular / Missed periods", "Cramping", "Abdominal pain / pressure",
        "Back pain", "Painful bowel movements", "Nausea", "Menstrual clots",
        "Infertility", "Painful cramps during period", "Pain / Chronic pain",
        "Diarrhea", "Long menstruation", "Constipation / Chronic constipation",
        "Vomiting / constant vomiting", "Fatigue / Chronic fatigue",
        "Painful ovulation", "Stomach cramping", "Migraines",
        "Extreme / Severe pain", "Leg pain", "Irritable Bowel Syndrome (IBS)",
        "Syncope (fainting, passing out)", "Mood swings", "Depression",
        "Bleeding", "Lower back pain", "Fertility Issues", "Ovarian cysts",
        "Painful urination", "Headaches", "Constant bleeding",
        "Pain after Intercourse", "Digestive / GI problems", "IBS-like symptoms",
        "Excessive bleeding", "Anaemia / Iron deficiency", "Hip pain",
        "Vaginal Pain/Pressure", "Sharp / Stabbing pain", "Bowel pain",
        "Anxiety", "Cysts (unspecified)", "Dizziness", "Malaise / Sickness",
        "Abnormal uterine bleeding", "Fever", "Hormonal problems", "Bloating",
        "Feeling sick", "Decreased energy / Exhaustion",
        "Abdominal Cramps during Intercourse", "Insomnia / Sleeplessness",
        "Acne / pimples", "Loss of appetite"
    ]
    
    error_response = validate_required_fields(data, required_fields)
    if error_response:
        return error_response
    
    try:
        input_data = np.array([float(data[field]) for field in required_fields]).reshape(1, -1)
        probability = models['endometre'].predict(input_data)[0][0]
        prediction = 1 if probability >= 0.5 else 0
        if prediction == 1:
            probability = float(probability)
        else:
            probability= 1-float(probability)
        
        result = save_test_result(user_id, "endometre", data, prediction, probability)
        
        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "test_id": result['_id']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# SOPK Advanced (Image-based) Prediction
@app.route('/predict/sopkadvanced', methods=['POST'])
@jwt_required()
def predict_sopkadvanced():
    """Predict SOPK using an ultrasound image."""
    user_id = get_jwt_identity()
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    if not allowed_file(image_file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        # Save the image temporarily
        filename = secure_filename(image_file.filename)
        uploads_dir = os.path.join(app.instance_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join(uploads_dir, filename)
        image_file.save(image_path)
        
        # Preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"error": "Error processing image"}), 400
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (150, 150))
        img_normalized = img_resized / 255.0
        img_reshaped = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        probability = (1-(float(models['sopk_advanced'].predict(img_reshaped)[0][0])))
        prediction = 0 if probability <= 0.5 else 1
        
        # Save result
        result_data = {
            "image_filename": filename,
            "image_path": image_path
        }
        result = save_test_result(user_id, "sopk_advanced", result_data, prediction, probability )

        if prediction == 0:
            probability=1-probability
        
        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "test_id": result['_id']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Obesity Prediction
@app.route('/predict/obesity', methods=['POST'])
@jwt_required()
def predict_obesity():
    """Predict obesity level based on input features."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    features = [
        "Age", "Height", "Weight", "CALC", "FAVC", "FCVC", "NCP", "SCC", 
        "SMOKE", "CH2O", "family_history_with_overweight", "FAF", "TUE", 
        "CAEC", "MTRANS"
    ]
    
    categorical_features = ['CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 
                          'CAEC', 'MTRANS']
    
    error_response = validate_required_fields(data, features)
    if error_response:
        return error_response
    
    try:
        encoded_data = []
        for feature in features:
            val = str(data.get(feature, '')).strip()
            
            if feature in categorical_features:
                if val not in encoders[feature].classes_:
                    return jsonify({"error": f"Invalid value for {feature}: {val}"}), 400
                encoded_data.append(encoders[feature].transform([val])[0])
            else:
                try:
                    encoded_data.append(float(val))
                except ValueError:
                    return jsonify({"error": f"Invalid numeric value for {feature}: {val}"}), 400
        
        x = np.array(encoded_data).reshape(1, -1)
        pred = int(models["obesity"].predict(x)[0])
        proba = float(models["obesity"].predict_proba(x)[0][pred])
        decoded_prediction = encoders['NObeyesdad'].inverse_transform([pred])[0]
        
        result = save_test_result(user_id, "obesity", data, decoded_prediction, proba)
        
        return jsonify({
            "prediction": decoded_prediction,
            "probability": proba,
            "test_id": result['_id']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Hypothyroid Prediction
@app.route('/predict/hypothyroid', methods=['POST'])
@jwt_required()
def predict_hypothyroid():
    """Predict hypothyroidism based on input features."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    required_fields = [
        "Age", "Sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "thyroid_surgery",
        "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium", "goitre",
        "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4", "T4U_measured", "T4U", 
        "FTI_measured", "FTI", "TBG_measured"
    ]
    
    error_response = validate_required_fields(data, required_fields)
    if error_response:
        return error_response
    
    try:
        x = np.array([float(data[f]) if isinstance(data[f], (int, float)) else int(data[f]) 
                     for f in required_fields]).reshape(1, -1)
        pred = int(models["hypothyroïdie"].predict(x)[0])
        proba = float(models["hypothyroïdie"].predict_proba(x)[0][pred])
        
        result = save_test_result(user_id, "hypothyroid", data, pred, proba)
        
        return jsonify({
            "prediction": pred,
            "probability": proba,
            "test_id": result['_id']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====== ADMIN ENDPOINTS ======
# ====== ADMIN USER MANAGEMENT ======
@app.route('/admin/users/<user_id>', methods=['PUT'])
@jwt_required()
@role_required('admin')
def admin_update_user(user_id):
    """Update a user's information (admin only)."""
    data = request.get_json()
    
    try:
        # Check if user exists
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Fields that can be updated by admin
        updatable_fields = ['name', 'email', 'phone', 'address', 'date_of_birth', 'gender', 'role']
        update_data = {}
        
        for field in updatable_fields:
            if field in data:
                update_data[field] = data[field]
        
        # Handle email uniqueness check
        if 'email' in update_data and update_data['email'] != user.get('email'):
            if mongo.db.users.find_one({"email": update_data['email'], "_id": {"$ne": ObjectId(user_id)}}):
                return jsonify({"error": "Email already in use by another account"}), 400
        
        # Handle password update if provided
        if 'password' in data:
            update_data['password'] = generate_password_hash(data['password'])
        
        if not update_data and 'password' not in data:
            return jsonify({"error": "No valid fields to update"}), 400
        
        update_data['updated_at'] = datetime.utcnow()
        
        result = mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if result.modified_count == 0 and 'password' not in data:
            return jsonify({"message": "No changes made"}), 200
        
        return jsonify({"message": "User updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/users/<user_id>', methods=['DELETE'])
@jwt_required()
@role_required('admin')
def admin_delete_user(user_id):
    """Delete a user (admin only)."""
    try:
        # Check if user exists
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Prevent admin from deleting themselves
        current_admin_id = get_jwt_identity()
        if str(user['_id']) == current_admin_id:
            return jsonify({"error": "Cannot delete your own account"}), 403
        
        # Delete user and all their test results
        with mongo.client.start_session() as session:
            with session.start_transaction():
                # Delete user's tests first
                mongo.db.tests.delete_many({"user_id": ObjectId(user_id)}, session=session)
                
                # Then delete the user
                result = mongo.db.users.delete_one({"_id": ObjectId(user_id)}, session=session)
                
                if result.deleted_count == 0:
                    session.abort_transaction()
                    return jsonify({"error": "Failed to delete user"}), 500
        
        return jsonify({"message": "User and all associated data deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/users/<user_id>/role', methods=['PUT'])
@jwt_required()
@role_required('admin')
def update_user_role(user_id):
    """Update a user's role (admin only)."""
    data = request.get_json()
    
    # Validate required fields
    if 'role' not in data:
        return jsonify({"error": "Missing role field"}), 400
    
    valid_roles = ['user', 'admin']  # Add other roles as needed
    if data['role'] not in valid_roles:
        return jsonify({"error": "Invalid role"}), 400
    
    try:
        # Check if user exists
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Prevent admin from modifying their own role
        current_admin_id = get_jwt_identity()
        if str(user['_id']) == current_admin_id:
            return jsonify({"error": "Cannot modify your own role"}), 403
        
        # Update role
        result = mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "role": data['role'],
                "updated_at": datetime.utcnow()
            }}
        )
        
        if result.modified_count == 0:
            return jsonify({"message": "No changes made"}), 200
        
        return jsonify({"message": "User role updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/users', methods=['GET'])
@jwt_required()
@role_required('admin')
def get_all_users():
    """Get all users (admin only)."""
    users = list(mongo.db.users.find({}, {"password": 0, "token": 0}))
    for user in users:
        user['_id'] = str(user['_id'])
    return jsonify(users), 200

@app.route('/admin/users/<user_id>', methods=['GET'])
@jwt_required()
@role_required('admin')
def get_user(user_id):
    """Get a specific user by ID (admin only)."""
    try:
        user = mongo.db.users.find_one(
            {"_id": ObjectId(user_id)},
            {"password": 0, "token": 0}
        )
        if not user:
            return jsonify({"error": "User not found"}), 404
        user['_id'] = str(user['_id'])
        return jsonify(user), 200
    except:
        return jsonify({"error": "Invalid user ID"}), 400

@app.route('/admin/users/<user_id>/tests', methods=['GET'])
@jwt_required()
@role_required('admin')
def get_user_tests(user_id):
    """Get all test results for a specific user (admin only)."""
    try:
        tests = list(mongo.db.tests.find({"user_id": ObjectId(user_id)}))
        for test in tests:
            test['_id'] = str(test['_id'])
            test['user_id'] = str(test['user_id'])
        return jsonify(tests), 200
    except:
        return jsonify({"error": "Invalid user ID"}), 400
 # ====== ADMIN DASHBOARD ENDPOINTS ======

@app.route('/admin/dashboard/stats', methods=['GET'])
@jwt_required()
@role_required('admin')
def get_dashboard_stats():
    """Get statistics for admin dashboard"""
    try:
        # User statistics
        total_users = mongo.db.users.count_documents({})
        active_users = mongo.db.users.count_documents({"last_login": {"$gte": datetime.utcnow() - timedelta(days=30)}})
        
        # Test statistics
        total_tests = mongo.db.tests.count_documents({})
        tests_by_type = list(mongo.db.tests.aggregate([
            {"$group": {"_id": "$maladie", "count": {"$sum": 1}}}
        ]))
        
        # Recent activity
        recent_tests = list(mongo.db.tests.find()
                           .sort("timestamp", -1)
                           .limit(5))
        
        # Format recent tests
        for test in recent_tests:
            test['_id'] = str(test['_id'])
            test['user_id'] = str(test['user_id'])
            
            # Get user name for display
            user = mongo.db.users.find_one(
                {"_id": test['user_id']},
                {"name": 1}
            )
            test['user_name'] = user.get('name', 'Unknown') if user else 'Unknown'
        
        return jsonify({
            "user_stats": {
                "total_users": total_users,
                "active_users": active_users
            },
            "test_stats": {
                "total_tests": total_tests,
                "tests_by_type": tests_by_type
            },
            "recent_activity": recent_tests
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/tests/search', methods=['GET'])
@jwt_required()
@role_required('admin')
def search_tests():
    """Search tests with filters"""
    try:
        # Get query parameters
        test_type = request.args.get('type')
        user_name = request.args.get('user_name')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        # Build query
        query = {}
        
        if test_type:
            query["maladie"] = test_type
            
        if user_name:
            # Find users with matching names
            users = list(mongo.db.users.find(
                {"name": {"$regex": user_name, "$options": "i"}},
                {"_id": 1}
            ))
            user_ids = [str(user['_id']) for user in users]
            query["user_id"] = {"$in": user_ids}
            
        if date_from and date_to:
            try:
                start_date = datetime.strptime(date_from, "%Y-%m-%d")
                end_date = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
                query["timestamp"] = {"$gte": start_date, "$lte": end_date}
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        # Execute query
        tests = list(mongo.db.tests.find(query).sort("timestamp", -1))
        
        # Format results
        result = []
        for test in tests:
            test['_id'] = str(test['_id'])
            test['user_id'] = str(test['user_id'])
            
            # Get user details
            user = mongo.db.users.find_one(
                {"_id": ObjectId(test['user_id'])},
                {"name": 1, "email": 1}
            )
            if user:
                test['user_name'] = user.get('name', 'Unknown')
                test['user_email'] = user.get('email', '')
            else:
                test['user_name'] = 'Unknown'
                test['user_email'] = ''
            
            result.append(test)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/tests/export', methods=['GET'])
@jwt_required()
@role_required('admin')
def export_tests():
    """Export test data to CSV"""
    try:
        # Get all tests with user information
        pipeline = [
            {
                "$lookup": {
                    "from": "users",
                    "localField": "user_id",
                    "foreignField": "_id",
                    "as": "user"
                }
            },
            {"$unwind": "$user"},
            {
                "$project": {
                    "test_id": {"$toString": "$_id"},
                    "maladie": 1,
                    "prediction": 1,
                    "probability": 1,
                    "timestamp": 1,
                    "user_id": {"$toString": "$user_id"},
                    "user_name": "$user.name",
                    "user_email": "$user.email"
                }
            }
        ]
        
        tests = list(mongo.db.tests.aggregate(pipeline))
        
        if not tests:
            return jsonify({"error": "No tests found to export"}), 404
        
        # Generate CSV
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=tests[0].keys())
        writer.writeheader()
        writer.writerows(tests)
        
        # Create response
        from flask import make_response
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=test_results.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/tests/bulk_delete', methods=['POST'])
@jwt_required()
@role_required('admin')
def bulk_delete_tests():
    """Delete multiple tests at once"""
    try:
        data = request.get_json()
        test_ids = data.get('test_ids', [])
        
        if not test_ids:
            return jsonify({"error": "No test IDs provided"}), 400
        
        # Convert string IDs to ObjectId
        object_ids = [ObjectId(test_id) for test_id in test_ids]
        
        # Delete tests
        result = mongo.db.tests.delete_many({"_id": {"$in": object_ids}})
        
        return jsonify({
            "message": f"Successfully deleted {result.deleted_count} tests",
            "deleted_count": result.deleted_count
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ====== ERROR HANDLERS ======
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

# ====== START SERVER ======
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)