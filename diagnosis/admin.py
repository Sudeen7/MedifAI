from django.contrib import admin
from .models import Contact, HeartPatientHistory, DiabetesPatientHistory, SymptomsPredictionHistory

class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'subject', 'created_at', 'message')  # Now created_at is a valid field
    search_fields = ('name', 'email', 'subject')  # Allow searching by these fields
    list_filter = ('created_at',)  # Allow filtering by date

admin.site.register(Contact, ContactMessageAdmin)

@admin.register(HeartPatientHistory)
class HeartPatientHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'first_name', 'last_name', 'email', 'prediction_result', 'created_at')
    search_fields = ('user__username', 'first_name', 'last_name', 'email', 'prediction_result')
    list_filter = ('prediction_result', 'created_at', 'user')
    ordering = ('-created_at',)

@admin.register(DiabetesPatientHistory)
class DiabetesPatientHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'first_name', 'last_name', 'email', 'glucose', 'bmi', 'prediction_result', 'created_at')
    search_fields = ('user__username', 'first_name', 'last_name', 'email', 'prediction_result')
    list_filter = ('prediction_result', 'created_at', 'user')
    ordering = ('-created_at',)

@admin.register(SymptomsPredictionHistory)
class SymptomsPredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'first_name', 'last_name', 'email', 'predicted_disease', 'created_at')
    search_fields = ('user__username', 'first_name', 'last_name', 'email', 'predicted_disease')
    list_filter = ('predicted_disease', 'created_at', 'user')
    ordering = ('-created_at',)