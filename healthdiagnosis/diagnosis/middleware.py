from datetime import datetime, timedelta
from django.conf import settings
from django.shortcuts import redirect
from django.contrib.auth import logout

class SessionTimeoutMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Ensure request.user exists before accessing it
        if hasattr(request, 'user') and request.user.is_authenticated:
            session_expiry_time = request.session.get('expiry_time')
            now = datetime.now()

            if session_expiry_time and datetime.strptime(session_expiry_time, '%Y-%m-%d %H:%M:%S') < now:
                # If session expired, log out the user
                logout(request)
                return redirect(settings.LOGIN_URL)
            else:
                # Update the session expiry time
                expiry_time = now + timedelta(seconds=settings.SESSION_COOKIE_AGE)
                request.session['expiry_time'] = expiry_time.strftime('%Y-%m-%d %H:%M:%S')

        response = self.get_response(request)
        return response
