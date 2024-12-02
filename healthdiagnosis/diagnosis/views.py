from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth import update_session_auth_hash
from django.contrib import messages
import numpy as np
import pandas as pd
import pickle
import warnings
from .forms import ContactForm
from django.core.mail import send_mail
from django.conf import settings
from django.core.mail import EmailMessage
from .models import Contact
import re
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
def landingpage(request):
    return render(request, "landing_page.html")

@login_required(login_url='login')
def index(request):
    return render(request, "index.html")

@login_required(login_url='login')
def predict(request):
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
            return render(request, 'index.html', {'message2': message2})

        # Check if any symptoms are the same (after filtering out empty selections)
        if len(set(selected_symptoms)) != len(selected_symptoms):
            message = "You cannot select the same symptom more than once."
            return render(request, 'index.html', {'message': message})

        # Combine symptoms into a string for displaying
        user_symptoms_string = ', '.join(selected_symptoms)

        # Call the prediction logic
        predicted_disease = get_predicted_value(selected_symptoms)
        print(predicted_disease)
        if predicted_disease.lower() == 'aids':
            message = "No disease matches your symptoms."
            return render(request, 'index.html', {'message': message, 'symptoms': user_symptoms_string})

        if predicted_disease not in diseases_list.values():
            message = "No disease matches your symptoms."
            return render(request, 'index.html', {'message': message, 'symptoms': user_symptoms_string})

        # Get additional details for the predicted disease
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        my_precautions = [i for i in precautions[0] if pd.notna(i)]
        
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
                'workout': workout
            }
        )
    
    # Render the form if not a POST request
    return render(request, 'index.html')


@login_required(login_url='login')
def about(request):
    return render(request, "about.html")  

@login_required(login_url='login')
def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Manually create and save the contact message
            contact = Contact(
                name=form.cleaned_data['name'],
                email=form.cleaned_data['email'],
                phone=form.cleaned_data['phone'],
                subject=form.cleaned_data['subject'],
                message=form.cleaned_data['message']
            )
            contact.save()  # Save to the database

            # Send an email to the user who submitted the form
            subject = f"Thank you for contacting us, {form.cleaned_data['name']}!"
            message = f"Dear {form.cleaned_data['name']},\n\n" \
                      f"Thank you for reaching out to us regarding: {form.cleaned_data['subject']}.\n\n" \
                      f"We have received your message:\n\n{form.cleaned_data['message']}\n\n" \
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

    return render(request, 'contact.html', {'form': form})

@login_required(login_url='login')
def success_view(request):
    return render(request, 'success.html')

@login_required(login_url='login')
def contact_us(request):
    if request.method == "POST":
        form = ContactForm(request.POST)
        if form.is_valid():
            # Save the contact message to the database
            form.save()

            # After saving, send an email to the user
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            try:
                # Sending email
                email_message = EmailMessage(
                    'Thank you for contacting us!',
                    f'<p>Hello {name},</p><p>Thank you for reaching out to us regarding "{subject}". We will respond to you soon.</p>',
                    settings.DEFAULT_FROM_EMAIL,  # Sender's email address
                    [email],  # Recipient's email address
                )
                email_message.content_subtype = "html"  # Specify email type as HTML
                email_message.send()

                # Redirect to the thank you page after sending the email
                return redirect('thank_you')  
            except Exception as e:
                messages.error(request, 'There was an error sending your message. Please try again later.')

        else:
            messages.error(request, 'There was an error in your form submission. Please try again.')

    else:
        form = ContactForm()

    return render(request, 'contact.html', {'form': form})

@login_required(login_url='login')
def thank_you(request):
    return render(request, 'thank_you.html')

@login_required(login_url='login')
def developer(request):
    return render(request, "developer.html")  

@login_required(login_url='login')
def blog(request):
    return render(request, "blog.html")  

@login_required(login_url='login')
def privacy(request):
    return render(request, "privacy.html")  

@login_required(login_url='login')
def terms(request):
    return render(request, "terms.html")  

@login_required(login_url='login')
def HomePage(request):
    return render(request, 'home.html')

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
    context = {
        'first_name': user.first_name,
        'last_name': user.last_name,
        'username': user.username,
        'email': user.email,
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
