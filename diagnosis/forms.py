from django import forms
from django.core.validators import EmailValidator
from .models import Contact
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field

class ContactForm(forms.Form):
    class Meta:
        model = Contact
        fields = ['name', 'email', 'phone', 'subject', 'message']

    name = forms.CharField(max_length=100)
    email = forms.CharField(validators=[EmailValidator()])
    phone = forms.CharField(max_length=10)
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Field('name', css_class='custom-input', label="Full Name"),
            Field('email', css_class='custom-input', label="Email Address"),
            Field('phone', css_class='custom-input', label="Phone Number"),
            Field('subject', css_class='custom-input', label="Subject"),
            Field('message', css_class='custom-textarea', label="Message"),
        )
