from django import forms
from django.core.validators import EmailValidator
from .models import Contact

class ContactForm(forms.Form):
    class Meta:
        model = Contact
        fields = ['name', 'email', 'phone', 'subject', 'message']
        
    name = forms.CharField(max_length=100)
    email = forms.CharField(validators=[EmailValidator()])
    phone = forms.CharField(max_length=10)
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea)

    

    