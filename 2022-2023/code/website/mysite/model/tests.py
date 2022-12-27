from django.test import TestCase
from django.contrib.auth.models import User
from django.test import Client
from .models import Image, Compare

class modelTest(TestCase):
    def setUp(self):
        compare = Compare.objects.create(image1 = "img1", image2 = "img2", user = "user", result = "True")

    def test_compare(self):
        c = Compare.objects.get(image1 = "img1", image2 = "img2", user="user")
        self.assertEqual(c.result, True)


