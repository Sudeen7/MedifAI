from django.shortcuts import render ,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth import update_session_auth_hash
from django.contrib import messages
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
from .forms import ContactForm
from django.core.mail import send_mail
from django.conf import settings
from .models import Contact, HeartPatientHistory, DiabetesPatientHistory, SymptomsPredictionHistory
import re
from django.contrib.admin.views.decorators import staff_member_required
import random
import time

warnings.filterwarnings('ignore', category=UserWarning)

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
voting_clf = pickle.load(open('models/voting_clf.pkl', 'rb'))

# Load Diabetes Prediction model
diabetes_pipeline = joblib.load('models/diabetes_pipeline.joblib')
# Print scaler info
print("For diabetes model")
print("Scaler means:", diabetes_pipeline.named_steps['scaler'].mean_)
print("Scaler scale:", diabetes_pipeline.named_steps['scaler'].scale_)

# Test different inputs
inputs = [
    np.array([1, 1, 1, 1, 1, 1, 0.1, 1]).reshape(1, -1),
    np.array([0, 120, 80, 20, 100, 30, 0.5, 25]).reshape(1, -1),
    np.array([5, 166, 72, 19, 175, 25.8, 0.587, 51]).reshape(1, -1)
]
for inp in inputs:
    print("Raw Input:", inp)
    print("Scaled Input:", diabetes_pipeline.named_steps['scaler'].transform(inp))
    print("Prediction:", diabetes_pipeline.predict(inp))


# Load the trained pipeline for heart disease prediction
heart_disease_model = joblib.load('models/heart_disease_pipeline.joblib')

# Print scaler information
print("For Heart Model")
print("Scaler means:", heart_disease_model.named_steps['scaler'].mean_)
print("Scaler scale:", heart_disease_model.named_steps['scaler'].scale_)

# Define test inputs
inputs = [
    np.array([62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2]).reshape(1, -1),
    np.array([45, 1, 2, 130, 240, 1, 1, 170, 0, 1.2, 1, 0, 3]).reshape(1, -1),
    np.array([55, 0, 1, 120, 200, 0, 0, 150, 1, 2.5, 2, 1, 2]).reshape(1, -1)
]

# Test each input
for inp in inputs:
    print("\nRaw Input:", inp)
    scaled_input = heart_disease_model.named_steps['scaler'].transform(inp)  # Apply scaling
    print("Scaled Input:", scaled_input)
    prediction = heart_disease_model.predict(inp)  # Predict using the pipeline
    print("Prediction:", prediction)

# Custom functions
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29,
    'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34,
    'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38,
    'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 
    'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
    'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 
    'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125,
    'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}
diseases_list = {0: '(vertigo) Paroymsal Positional Vertigo', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis', 4: 'Allergy', 
                 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis', 8: 'Chicken pox', 9: 'Chronic cholestasis', 
                 10: 'Common Cold', 11: 'Dengue', 12: 'Diabetes ', 13: 'Dimorphic hemorrhoids (piles)', 14: 'Drug Reaction', 
                 15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis', 18: 'Heart attack', 19: 'Hepatitis B', 20: 'Hepatitis C', 
                 21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension ', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism', 
                 27: 'Impetigo', 28: 'Jaundice', 29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthritis', 32: 'Paralysis (brain hemorrhage)', 
                 33: 'Peptic ulcer disease', 34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis', 37: 'Typhoid', 38: 'Urinary tract infection', 
                 39: 'Varicose veins', 40: 'hepatitis A'}

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]
    wrkout = workout[workout['disease'] == dis]['workout']
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    prediction = voting_clf.predict([input_vector])
    return diseases_list.get(prediction[0], "Disease not found")

# Views
@staff_member_required
def heart_patient_history_view(request):
    histories = HeartPatientHistory.objects.all().order_by('-created_at')
    return render(request, 'heart_patient_history.html', {'histories': histories, 'title': 'Heart-Patient-Histroy'})

@staff_member_required
def diabetes_history(request):
    histories = DiabetesPatientHistory.objects.all().order_by('-created_at')
    return render(request, 'diabetes_patient_history.html', {'histories': histories, 'title': 'Diabetes-Patient-History'})

@staff_member_required
def symptoms_prediction_history(request):
    histories = SymptomsPredictionHistory.objects.all().order_by('-created_at')
    return render(request, 'symptoms_prediction_history.html', {'histories': histories, 'title': 'Symptoms-Prediction-History'})

def landingpage(request):
    if request.user.is_authenticated:
        return redirect('homepage')
    return render(request, "landing_page.html")

@login_required(login_url='login')
def index(request):
    return render(request, "index.html", {'title': 'Predict Disease'})

@login_required(login_url='login')
def predict(request):
    title = 'Predict Disease'

    if request.method == 'POST':
        # Retrieve symptoms from the dropdown selections
        symptom1 = request.POST.get('symptom1')
        symptom2 = request.POST.get('symptom2')
        symptom3 = request.POST.get('symptom3')
        symptom4 = request.POST.get('symptom4')
        symptom5 = request.POST.get('symptom5')
        symptom6 = request.POST.get('symptom6')

        # Collect all symptoms into a list
        symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5, symptom6]

        # Filter out empty selections (None or blank)
        selected_symptoms = [s for s in symptoms if s]

        # Debugging: print the selected symptoms
        print(f"Selected symptoms: {selected_symptoms}")

        # Ensure that at least 3 symptoms are selected
        if len(selected_symptoms) < 3:
            message2 = "Please select at least three symptoms."
            return render(request, 'index.html', {'message2': message2, 'title': title})

        # Check if any symptoms are the same (after filtering out empty selections)
        if len(set(selected_symptoms)) != len(selected_symptoms):
            message = "You cannot select the same symptom more than once."
            return render(request, 'index.html', {'message': message, 'title': title})

        # Combine symptoms into a string for displaying and saving
        user_symptoms_string = ', '.join(selected_symptoms)

        # Call the prediction logic
        predicted_disease = get_predicted_value(selected_symptoms)
        print(predicted_disease)

        if predicted_disease not in diseases_list.values() or predicted_disease.lower() == 'aids':
            message = "No disease matches your symptoms."
            return render(request, 'index.html', {'message': message, 'symptoms': user_symptoms_string, 'title': title})

        # Get additional details for the predicted disease
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        my_precautions = [i for i in precautions[0] if pd.notna(i)]

        # Save the prediction history
        user = request.user
        SymptomsPredictionHistory.objects.create(
            user=user,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            symptoms=user_symptoms_string,
            predicted_disease=predicted_disease
        )

        # Render the results with all relevant information
        return render(
            request,
            'index.html',
            {
                'symptoms': user_symptoms_string,
                'predicted_disease': predicted_disease,
                'dis_des': dis_des,
                'my_precautions': my_precautions,
                'medications': medications,
                'my_diet': rec_diet,
                'workout': workout,
                'title': title
            }
        )

    return render(request, 'index.html', {'title': title})



@login_required(login_url='login')
def heart_disease_predict(request):
    title = 'Heart Disease Prediction'

    if request.method == 'POST':
        # Retrieve input data from form
        try:
            age = float(request.POST.get('age', 0))
            sex = float(request.POST.get('sex', 0))
            cp = float(request.POST.get('cp', 0))
            trestbps = float(request.POST.get('trestbps', 0))
            chol = float(request.POST.get('chol', 0))
            fbs = float(request.POST.get('fbs', 0))
            restecg = float(request.POST.get('restecg', 0))
            thalach = float(request.POST.get('thalach', 0))
            exang = float(request.POST.get('exang', 0))
            oldpeak = float(request.POST.get('oldpeak', 0))
            slope = float(request.POST.get('slope', 0))
            ca = float(request.POST.get('ca', 0))
            thal = float(request.POST.get('thal', 0))

            # Collect input data in an array
            input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
            print("Input array: ", input_data)
            # Ensure scaling is handled inside the pipeline
            prediction = heart_disease_model.predict(input_data)
            print("Prediction: ", prediction)

            # Determine result
            result = "The Person may have Heart Disease" if prediction[0] == 1 else "The Person may not have Heart Disease"

            # Retrieve user details
            user = request.user
            first_name = user.first_name
            last_name = user.last_name
            email = user.email

            # Save the prediction history
            HeartPatientHistory.objects.create(
                user=user,
                first_name=first_name,
                last_name=last_name,
                email=email,
                age=age,
                sex=sex,
                cp=cp,
                trestbps=trestbps,
                chol=chol,
                fbs=fbs,
                restecg=restecg,
                thalach=thalach,
                exang=exang,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal,
                prediction_result=result
            )

            return render(request, 'heart_disease_predict.html', {
                'title': title, 
                'result': result,
                'input_data': input_data.tolist()
            })

        except Exception as e:
            return render(request, 'heart_disease_predict.html', {
                'title': title, 
                'error': f"Invalid input: {e}"
            })
    
    return render(request, 'heart_disease_predict.html', {'title': title})


@login_required(login_url='login')
def heart_disease_attributes_help(request):
    return render(request, 'heart_disease_help.html', {'title': 'Heart Disease Attribute Help'})

@login_required(login_url='login')
def predict_diabetes(request):
    title = 'Diabetes Prediction'
    
    if request.method == 'POST':
        # Retrieve input data from form (defaulting to 0 if not provided)
        pregnancies = request.POST.get('pregnancies', 0)
        glucose = request.POST.get('glucose', 0)
        blood_pressure = request.POST.get('blood_pressure', 0)
        skin_thickness = request.POST.get('skin_thickness', 0)
        insulin = request.POST.get('insulin', 0)
        bmi = request.POST.get('bmi', 0)
        diabetes_pedigree_function = request.POST.get('diabetes_pedigree_function', 0)
        age = request.POST.get('age', 0)

        # Create input data array and reshape to 2D
        input_data = np.array([
            float(pregnancies),
            float(glucose),
            float(blood_pressure),
            float(skin_thickness),
            float(insulin),
            float(bmi),
            float(diabetes_pedigree_function),
            float(age)
        ]).reshape(1, -1)

        # The pipeline automatically scales the raw input data and predicts.
        prediction = diabetes_pipeline.predict(input_data)
        result = 'The Person may have Diabetes' if prediction[0] == 1 else 'The Person may not have Diabetes'

        # Save prediction history
        user = request.user
        DiabetesPatientHistory.objects.create(
            user=user,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            pregnancies=pregnancies,
            glucose=glucose,
            blood_pressure=blood_pressure,
            skin_thickness=skin_thickness,
            insulin=insulin,
            bmi=bmi,
            diabetes_pedigree_function=diabetes_pedigree_function,
            age=age,
            prediction_result=result
        )

        return render(request, 'predict_diabetes.html', {
            'title': title,
            'result': result,
            'input_data': input_data[0].tolist()
        })

    return render(request, 'predict_diabetes.html', {'title': title})

@login_required(login_url='login')
def diabetes_attributes_help(request):
    return render(request, 'diabetes_help.html', {'title': 'Diabetes Attribute Help'})

@login_required(login_url='login')
def about(request):
    return render(request, "about.html", {'title': 'About Us'})  

@login_required(login_url="login")
def explore(request):
    return render(request, "explore_more.html", {'title': 'Explore'})

@login_required(login_url='login')
def contact_view(request):
    title = 'Contact Us'

    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Manually create and save the contact message
            contact = Contact(
                name=form.cleaned_data['name'],
                email=form.cleaned_data['email'],
                phone=form.cleaned_data['phone'],
                subject=form.cleaned_data['subject'],
                message=form.cleaned_data['message'],
            )
            contact.save()  # Save to the database

            # Send an email to the user who submitted the form
            subject = f"Thank you for contacting us, {form.cleaned_data['name']}!"
            message = f"Dear {form.cleaned_data['name']},\n\n" \
                      f"Thank you for reaching out to us regarding: {form.cleaned_data['subject']}.\n\n" \
                      f"We have received your message:\n\nMessage: {form.cleaned_data['message']}\n\n" \
                      "Our team will get back to you soon.\n\nBest regards,\nMedifAI Team"
            recipient = form.cleaned_data['email']



            send_mail(
                subject,  # Subject of the email
                message,  # Email content
                settings.DEFAULT_FROM_EMAIL,  # Sender's email (configured in settings)
                [recipient],  # Recipient's email (user's email)
                fail_silently=False,  # Raise an error if email sending fails
            )

            return redirect('success')  # Redirect to a success page

    else:
        form = ContactForm()

    return render(request, 'contact.html', {'form': form, 'title': title})


@login_required(login_url='login')
def success_view(request):
    return render(request, 'success.html')

@login_required(login_url='login')
def developer(request):
    return render(request, "developer.html", {'title': 'Developers'})  

@login_required(login_url='login')
def blog(request):
    return render(request, "blog.html", {'title': 'Blog'})  

@login_required(login_url='login')
def privacy(request):
    return render(request, "privacy.html", {'title': 'Privacy Policy'})  

@login_required(login_url='login')
def terms(request):
    return render(request, "terms.html", {'title': 'Terms of Services'})  

@login_required(login_url='login')
def HomePage(request):
    return render(request, 'home.html', {'title': 'Home Page'})

def SignupPage(request):
    if request.method == 'POST':
        # Get form data
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists! Please choose a different username.")
            context = {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
            }
            return render(request, 'signup.html', context)

        # Check if passwords match
        if password1 != password2:
            messages.error(request, "Your password and confirm password do not match!")
            context = {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
            }
            return render(request, 'signup.html', context)

        # Password strength validation (at least 8 characters)
        if len(password1) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            context = {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
            }
            return render(request, 'signup.html', context)

        # Additional checks for password strength (optional)
        if not re.search(r'[A-Z]', password1):  # Check for uppercase letter
            messages.error(request, "Password must contain at least one uppercase letter.")
            context = {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
            }
            return render(request, 'signup.html', context)

        if not re.search(r'[0-9]', password1):  # Check for digit
            messages.error(request, "Password must contain at least one number.")
            context = {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
            }
            return render(request, 'signup.html', context)

        if not re.search(r'[\W_]', password1):  # Check for special character
            messages.error(request, "Password must contain at least one special character (e.g., @, #, $, etc.).")
            context = {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
            }
            return render(request, 'signup.html', context)

        # Create the user if all checks pass
        user = User.objects.create_user(username=username, email=email, password=password1)
        user.first_name = first_name
        user.last_name = last_name
        user.save()

        # Add success message
        messages.success(request, f"Hello {username}, your account has been registered successfully! Please log in.")
        return redirect('login')  # Redirect to the login page after successful registration

    return render(request, 'signup.html')


def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)

            messages.success(request, f"{username}, you have logged in successfully! Now you can explore our web app")

            return redirect('homepage')
        else:
            # Return to login page with error and username
            messages.error(request, "Username or Password is incorrect!")
            return render(request, 'login.html', {'username': username})
    
    return render(request, 'login.html')

@login_required(login_url='login')
def LogoutPage(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')  
def ProfilePage(request):
    user = request.user  
    title = 'User Profile'

    context = {
        'first_name': user.first_name,
        'last_name': user.last_name,
        'username': user.username,
        'email': user.email,
        'title': title
    }
    return render(request, 'profile.html', context)


@login_required(login_url='login')
def EditProfilePage(request):
    user = request.user

    if request.method == 'POST':
        # Get the updated data from the form
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        # Check if the passwords match
        if password1 and password1 != password2:
            messages.error(request, "Your passwords do not match.")
            return redirect('profile_edit')

        # Check if any data has been changed
        if user.username != username:
            user.username = username
        if user.first_name != first_name:
            user.first_name = first_name
        if user.last_name != last_name:
            user.last_name = last_name
        
        # Update password if a new one is provided
        if password1:
            user.set_password(password1)
            update_session_auth_hash(request, user) 

        user.save()

        messages.success(request, "Your profile has been updated successfully.")
        return redirect('profile')

    context = {
        'first_name': user.first_name,
        'last_name': user.last_name,
        'username': user.username
    }
    return render(request, 'profile_edit.html', context)

def ForgotPasswordPage(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        user = User.objects.filter(email=email).first()
        
        if user:
            # Generate a random 6-digit OTP
            otp = random.randint(100000, 999999)
            
            # Save the OTP in the user's session
            request.session['otp'] = otp
            request.session['otp_timestamp'] = time.time()
            request.session['email'] = email
            
            # Send the OTP to the user's email
            send_mail(
                'Password Reset Request - MedifAI Team',
                f'''
                Hello,

                We received a request to reset the password for your MedifAI account associated with the email address {email}.
                To proceed with resetting your password, please use the One-Time Password (OTP) below:

                OTP: {otp}

                Please note that this OTP is valid for 60 seconds only. If you did not request this password reset, please disregard this email.

                If you have any questions or need assistance, feel free to reach out to our support team.

                Best regards,
                The MedifAI Team
                ''',
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
            )
            
            messages.success(request, "OTP has been sent to your email.")
            return redirect('verify_otp')
        else:
            messages.error(request, "No user found with this email address.")
    
    return render(request, 'forgot_password.html')

def VerifyOTPPage(request):
    if request.method == 'POST':
        otp = request.POST.get('otp')
        session_otp = request.session.get('otp')
        otp_timestamp = request.session.get('otp_timestamp')
        
        if otp_timestamp and time.time() - otp_timestamp > 60:
            messages.error(request, "OTP has expired. Please request a new OTP.")
            return redirect('forgot_password')
        
        if int(otp) == session_otp:
            messages.success(request, "OTP verified. Please reset your password.")
            return redirect('reset_password')
        else:
            messages.error(request, "Invalid OTP. Please try again.")
    
    return render(request, 'verify_otp.html')


def ResetPasswordPage(request):
    if request.method == 'POST':
        new_password1 = request.POST.get('new_password1')
        new_password2 = request.POST.get('new_password2')
        
        if new_password1 != new_password2:
            messages.error(request, "Passwords do not match.")
            return redirect('reset_password')

        # Password strength validation (at least 8 characters)
        if len(new_password1) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return redirect('reset_password')

        # Check for uppercase letter
        if not re.search(r'[A-Z]', new_password1):
            messages.error(request, "Password must contain at least one uppercase letter.")
            return redirect('reset_password')

        # Check for digit
        if not re.search(r'[0-9]', new_password1):
            messages.error(request, "Password must contain at least one number.")
            return redirect('reset_password')

        # Check for special character
        if not re.search(r'[\W_]', new_password1):
            messages.error(request, "Password must contain at least one special character (e.g., @, #, $, etc.).")
            return redirect('reset_password')
        
        # Retrieve user by email from session
        email = request.session.get('email')
        user = User.objects.get(email=email)
        
        # Set the new password
        user.set_password(new_password1)
        user.save()
        
        messages.success(request, "Password reset successfully. Please login with your new password.")
        return redirect('login')
    
    return render(request, 'reset_password.html')
