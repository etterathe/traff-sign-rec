import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from tensorflow import keras
from labels import LABELS

app = Flask(__name__)
MODEL = keras.models.load_model('./model/model.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
HOME_PAGE = 'index.html'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    images = []
    img = Image.open(image_path)
    img = img.resize((50,50))
    img = np.array(img)
    images.append(img)
    images = np.array(images)
    images = images/255

    return images

@app.route('/', methods=['GET'])
def render():
    return render_template(HOME_PAGE)

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']

    if imagefile and allowed_file(imagefile.filename):
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path) 
        processed_image = preprocess_image(image_path)
        y_pred = np.argmax(MODEL.predict(processed_image),axis=1)
        y_label = LABELS[y_pred[0]]
        return render_template(HOME_PAGE, prediction = y_label)
    else:
        no_image = f"Available formats: {ALLOWED_EXTENSIONS} "
        return render_template(HOME_PAGE, no_image = no_image)

if __name__ == '__main__':
    app.run(port = 3000, debug=True)