import os
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'


class_names = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy', 'unknown']
DISEASE_Name =  [ "Cassava Bacterial Blight (CBB)",
                  "Cassava Brown Streak Disease (CBSD)",
                  "Cassava Green Mottle (CGM)",
                  "Cassava Mosaic Disease (CMD)",
                  "Healthy",
                  'Unknown' ]

def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    global model
    model = tf.keras.models.load_model(MODEL_FOLDER +'/xor/')    # load the previouly saved model
    print('[INFO] : Model loaded')


def predict(fullpath):
    data = image.load_img(fullpath, target_size=(224, 224, 3))

    data = np.expand_dims(data, axis=0)

    data = data.astype('float') / 255.0

    probabilities = model.predict(data)

    predictions = tf.math.argmax(probabilities, axis=-1)

    predictions = np.array(predictions)

    idx = predictions[0]

    label  = DISEASE_Name[idx]

    proba = probabilities[0][idx]*100

    proba = round(proba, 2)

    return label, proba


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        label, proba = predict(fullname)

        return render_template('predict.html', image_file_name=file.filename,
                                label=label, accuracy=proba)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
