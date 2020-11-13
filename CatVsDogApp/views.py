from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import model_from_json
import tensorflow as tf
import json
import numpy as np

# Create your views here.

with open('./models/model.json','r') as f:
    labelInfo=f.read()

model_c = model_from_json(labelInfo)

model_c.load_weights('./models/new_model.h5')
model_c.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])

def index(request):
    context = {'a':1}
    return render(request,"index.html",context)

def predictImage(request):
    if request.method=='POST':
        request_file = request.FILES['filePath']
        if request_file:
            fs = FileSystemStorage()
            file = fs.save(request_file.name,request_file)
            file_url = fs.url(file)
            testimage='.'+file_url
            img = image.load_img(testimage, target_size=(224, 224))
            x = image.img_to_array(img)
            # Reshape data for the model
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            # Prepare the image for the CNN Model model
            x = preprocess_input(x)
            # Pass image into model to get encoded features
            feature = model_c.predict(x, verbose=0)
            # Store encoded features for the image
            predictedLabel = int(np.argmax(feature[0]))
            if predictedLabel==0:
                predictedLabel = "Cat"
            else:
                predictedLabel = "Dog"
            print(predictedLabel)
            context = {'filePathName':file_url,'predictedLabel':predictedLabel}
            return render(request,'predict.html',context)

    return render(request,'index.html')
