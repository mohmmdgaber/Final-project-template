from django.test import TestCase
from django.contrib.auth.models import User
from django.test import Client

class registerTest(TestCase):
    def setUp(self):
        user = User.objects.create(username="user")
        user.set_password('123456')
        user.save()

    def test_register(self):
        u = User.objects.get(username="user")
        self.assertNotEqual(u, None)

    def test_login(self):
        c = Client()
        logged_in = c.login(username='user', password='123456')
        self.assertNotEqual(logged_in, None)
