# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# required library
import cv2
import numpy as np
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
from utils import detect_lp
import urllib.request


# library for MobileNet
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

#library base64 for API
import base64
import io
from PIL import Image

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


#Function load model for MobileNet
def get_model():
    # load pretrain weitght imagenet
    base_model=MobileNet(weights=None, input_shape=(224, 224, 3), include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
    base_model.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(6)(x)
    model = keras.Model(inputs, outputs)
    return model

################ LOAD NECCESARY ################    
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Load model architecture, weight and labels
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")


#Load model MobileNet
modelMobile = get_model()
print("[INFO] MobileNet is downloaded successfully...")
modelMobile.load_weights("./mobilenet-kaggle.h5")
print("[INFO] MobileNet loaded successfully...")

def preprocess_image(image_path,resize=False):
    # img = cv2.imread(image_path)
    req = urllib.request.urlopen(image_path)
    img = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def preprocess_ver2(image,resize=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    if resize:
        img = cv2.resize(image, (224,224))
    return image

def get_plate(image_path, Dmax=608, Dmin = 250):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

def get_plate_ver2(image, Dmax=608, Dmin = 250):
    vehicle = preprocess_ver2(image)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def predict_type_of_vehicle(image):
    class_name = [ 'Bus','Car','Limousine','minivan','motorcycle','Truck']
    # class_name = ['Ambulance','Barge','Bicycle','Boat', 'Bus','Car','Cart','Caterpillar','Helicopter','Limousine','Motorcycle','Segway','Snowmobile','Tank','Taxi','Truck','Van']
    image = cv2.resize(image, dsize=(224, 224))
    image = image.astype('float')*1./255
    image = np.expand_dims(image, axis=0)
    predict = modelMobile.predict(image)
    vehicle_type = class_name[np.argmax(predict)]
    return vehicle_type

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def get_coordinate(image, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    pts = np.array(pts, np.int32)
    # pts = pts.reshape((-1,1,2))
    # vehicle_image = preprocess_ver2(image)
    # cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return pts.tolist()

    
def draw_box(image, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    # vehicle_image = preprocess_ver2(image)
    cv2.polylines(image,[pts],True,(0,1,0),thickness)
    return image

def encode_image(arr_Image):
    im = Image.fromarray(arr_Image.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "jpeg")
    rawBytes.seek(0)  # return to the start of the file
    base64_data = base64.b64encode(rawBytes.read()).decode('utf-8')
    return base64_data

def main(test_image_path):
    try: 
        vehicle, LpImg,cor = get_plate_ver2(test_image_path, 608, 260)
        final_string = ''
        if (len(LpImg)): #check if there is at least one license image
            # Scales, calculates absolute values, and converts the result to 8-bit.
            plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            
            # convert to grayscale and blur the image
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(7,7),0)
            
            # Applied inversed thresh_binary 
            binary = cv2.threshold(blur, 180, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

        cont, _  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Before RETR_EXTERNAL

        # creat a copy version "test_roi" of plat_image to draw bounding box
        test_roi = plate_image.copy()

        # Initialize a list which will be used to append charater image
        crop_characters = []

        # define standard width and height of character
        digit_w, digit_h = 30, 60

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.5 and 1200<w*h<=4000: # Only select contour with defined ratio  -----------old doesnot have w*H---------
                if h/plate_image.shape[0]>=0.4: # Select contour which has the height larger than 50% of the plate
                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

        for i,character in enumerate(crop_characters):
            title = np.array2string(predict_from_model(character,model,labels))
            final_string+=title.strip("'[]")
        #Coordinates of LP
        pts = get_coordinate(vehicle,cor)
        image_withBox = draw_box(vehicle,cor)
        image_withBox = 255*image_withBox
        encoded_image = encode_image(image_withBox)
        return final_string, pts, encoded_image
    except Exception as e:
        print(e)
        return "No license plate is founded. Try a different image",[],''
if __name__ == "__main__":
    main(test_image_path)