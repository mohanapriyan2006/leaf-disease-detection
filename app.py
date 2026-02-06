'''
    First RUN the train.py file using CMD 'py train.py'
    and then RUN it.
'''

from keras.models import load_model , model_from_json
from keras.preprocessing import image
import os
import numpy

print("\nModel is Loading...\n")

# for new version
model = load_model("model.keras")
model.load_weights("model1.h5")

# # (OR) for old version
# modelfile = open("model1.json",'r')
# modelJson = modelfile.read()
# modelfile.close()
# model = model_from_json(modelJson)
# model.load_weights("model1.h5")

print("\n  Model was Loaded\n")

CLASSES=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
       "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
       "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
       "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
       "Potato___Early_blight","Potato___Healthy","Potato___Late_blight","Tomato___Bacterial_spot",
       "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold",
       "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
       "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus"]

def classify(f):
    filePath = f
    img = image.load_img(filePath,target_size=(128,128))
    img = image.img_to_array(img)
    img = numpy.expand_dims(img,axis=0)
    
    result = model.predict(img)
    label = CLASSES[result.argmax()]
    print("%s =====> [ %s ]" % (filePath,label))
    

# load dataset for check
print("\nDatasets are loading...\n")

path = "dataset/check"
files = []
for rt, dirs, filenames in os.walk(path):
    for f in filenames:
        if f.lower().endswith(".jpg"):
            files.append(os.path.join(rt, f))
        
print(" Datesets are Loaded\n\nLeaf Disease Prediction started:")

print("--------------------------------------------------")
print("            INPUT                 =====>   RESULT")
print("--------------------------------------------------")
for f in files:
    classify(f)
    print()
print("--------------------------------------------------")
    
print("  Leaf Disease Prediction stoped.")


