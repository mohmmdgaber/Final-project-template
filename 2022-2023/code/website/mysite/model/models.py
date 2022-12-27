from django.db import models
from django.conf import settings
# from image_cropping import ImageRatioField
# from djongo.storage import GridFSStorage



class Image(models.Model):
    image = models.ImageField()
    user = models.CharField(max_length=30)
    title = models.CharField(max_length=60)


class Compare(models.Model):
    image1 = models.ImageField()
    image2 = models.ImageField()
    user = models.CharField(max_length=30)
    result = models.BooleanField()



