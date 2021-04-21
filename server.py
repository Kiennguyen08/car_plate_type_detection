import main
import flask
from flask import Flask, request
import json
import numpy as np
import cv2
import time
app = Flask(__name__)


@app.route("/", methods=["GET"])
def _hello_world():
	return "Hello world"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    # req_data = request.get_json()
    # img_path = req_data["image_url"]
    # print("img_image",img_path)
    # number_plate = main.main(img_path)
    start = time.time()
    if request.files:
        image = request.files["image"].read()
        npimg = np.fromstring(image, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        number_plate, pts, encoded_image = main.main(img)
        vehicle_type = main.predict_type_of_vehicle(img)
        end = time.time()
        data["number_plate"] = str(number_plate)
        data["vehicle_type"] = str(vehicle_type)
        data["time_execution"] = round((end-start),2)
        # data["coordinate"] = pts
        data["encoded_image"] = encoded_image
        data["success"] = True

    return json.dumps(data, ensure_ascii=False)

if __name__ == "__main__":
    print("App run!")
    app.run(debug=True, host="0.0.0.0")