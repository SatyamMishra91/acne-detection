
from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
#tf libs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.models import load_model

#img processing libs
import cv2
import numpy as np

def cvcal(ph):
    print(ph)
    

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos',IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo=FileField(
        validators=[
            FileAllowed(photos,'Only images are allowed'),
                    FileRequired('File field should not be empty')
                    ]
                     )
    submit=SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'],filename)

@app.route('/',methods=['GET','POST'])
def upload_image():
    form=UploadForm()
    dims=''
    val=''
    if form.validate_on_submit():
        #print('filename' +filename)
        #print('phot data' +form.photo.data)
        filename = photos.save(form.photo.data)
        #print('filename nw' +filename)
        file_url = url_for('get_file',filename=filename)
        ph=file_url[1:]
        cvcal(ph)
        img=cv2.imread(ph)
        dims=img.shape
        print(dims)


        model = load_model('../Acne detection/acne_detection_model')

        #dir_path = "C:/acne_detection/testing/1.jpg"
        img = cv2.imread(ph)
        resz = cv2.resize(img, (225, 225))
        x = np.expand_dims(resz, axis=0)
        images = np.vstack([x])
        val = model.predict(images)
        if val == 0:
            val = "Acne Face"
            print("Acne Face")
        else:
            val = "Clean Face"
            print("Clean Face")
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url, pha=str(val))



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
