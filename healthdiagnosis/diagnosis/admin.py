# admin.py
from django.contrib import admin
from .models import ContactMessage

class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'subject', 'created_at')  # Fields to display in the list view
    search_fields = ('name', 'email', 'subject')  # Allow searching by these fields
    list_filter = ('created_at',)  # Allow filtering by date

admin.site.register(ContactMessage, ContactMessageAdmin)
