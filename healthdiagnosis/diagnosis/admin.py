from django.contrib import admin
from .models import Contact

class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'subject', 'created_at', 'message')  # Now created_at is a valid field
    search_fields = ('name', 'email', 'subject')  # Allow searching by these fields
    list_filter = ('created_at',)  # Allow filtering by date

admin.site.register(Contact, ContactMessageAdmin)
