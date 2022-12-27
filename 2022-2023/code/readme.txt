
Author: Rotem Vertman | Oren Shaya 


Final project - Writer verification from handwriting:

The program trains Saimise network model to verify writer using images of handwritten documents.


Instructions to run the program:

Run from command line using the format:
cd Website/mysite
python manage.py runserver

Website is directory to files for a Django website interface
preprocessing.py includes:
 pre-processing for the image
 train\load model
 show accuracy\confusion matrix


Libraries used in the program:

cv2
matplotlib
pillow
tensorflow
PDF2image
numpy
os
sklearn
pandas