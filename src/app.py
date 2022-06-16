from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from firebase_admin import credentials, firestore, initialize_app, storage
from google.cloud import storage as gcp_storage
import json
import datetime


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "healthcure-9f50b-eaeed22fd8f2.json"

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'storageBucket': 'healthcure-9f50b.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()
users_ref = db.collection('users')


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = gcp_storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


covid_model = load_model('../models/covid-19_model.h5')
braintumor_model = load_model('../models/brain_tumor_model.h5')
alzheimer_model = load_model('../models/alzheimer_model.h5')
diabetes_model = pickle.load(open('../models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('../models/heart_disease_model.dat', 'rb'))
pneumonia_model = load_model('../models/pneumonia_model.h5')
breastcancer_model = joblib.load('../models/breast_cancer_model.pkl')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

############################################# BRAIN TUMOR FUNCTIONS ################################################


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

##################################################################################################


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/covid')
def covid():
    return render_template('covid.html')


@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')


@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')


###################################################################


@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']

        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = covid_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            
            # Upload Image to Firebase Storage
            blob_image_name = "covid_" + str(pred) + "_" + filename
            try:
                upload_blob(bucket.name, 'static/uploads/' + filename, blob_image_name)
            except Exception as e:
                print("Upload Failed...")
                print(e)
            
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']

        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img], (224, 224))
            pred = braintumor_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Brain Tumor test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
            
            # Upload Image to Firebase Storage
            blob_image_name = "braintumor_" + str(pred) + "_" + filename
            try:
                upload_blob(bucket.name, 'static/uploads/' + filename, blob_image_name)
            except Exception as e:
                print("Upload Failed...")
                print(e)
            
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        
        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)
        
        pred = diabetes_model.predict(
            [[insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']
        
        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)
        
        pred = breastcancer_model.predict(
            np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resulta', methods=['GET', 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']

        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img/255.0
            pred = alzheimer_model.predict(img)
            pred = pred[0].argmax()
            print(pred)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            
            # Upload Image to Firebase Storage
            blob_image_name = "alzheimer_" + str(pred) + "_" + filename
            try:
                upload_blob(bucket.name, 'static/uploads/' + filename, blob_image_name)
            except Exception as e:
                print("Upload Failed...")
                print(e)

            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=0, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']

        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            
            # Upload Image to Firebase Storage
            blob_image_name = "pneumonia_" + str(pred) + "_" + filename
            try:
                upload_blob(bucket.name, 'static/uploads/' + filename, blob_image_name)
            except Exception as e:
                print("Upload Failed...")
                print(e)
            
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        
        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)
        
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
