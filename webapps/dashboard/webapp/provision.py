from django.contrib.auth.models import User
import os


def provision_admin_user():

    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        user = User.objects.create(username='admin', email='no-reply@synaptics.com', is_superuser=True, is_staff=True)        

    password = os.environ.get("ADMIN_PASSWORD", "password")
    user.set_password(password)
    user.save()

    print("Admin user 'admin' provisioned")
