from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import google.generativeai as genai

# Configure the generative AI model
genai.configure(api_key="AIzaSyCvjDaJDwlIq9GWx3goPdt-DsOSYjL4AdM")
gen_model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained TensorFlow model
tf_model = load_model('raw_cnn_model.h5')

# Define classes (update according to your dataset)
classes = [
    "Alopecia Areata", "Contact Dermatitis", "Folliculitis", "Head Lice",
    "Lichen Planus", "Male Pattern Baldness", "Psoriasis",
    "Seborrheic Dermatitis", "Telogen Effluvium", "Tinea Capitis"
]

def preprocess_image(img_path):
    """Preprocess the uploaded image for prediction."""
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size for your model
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    """Render the landing page."""
    diseases = [
        {"name": "Alopecia Areata", "image": "images/Alopecia Areata.jpg", "description": "Patchy hair loss due to autoimmune response."},
        {"name": "Contact Dermatitis", "image": "images/contact_dermatitis_0002.jpg", "description": "Scalp irritation caused by allergens or irritants."},
        {"name": "Folliculitis", "image": "images/folliculitis_0005.jpg", "description": "Inflamed hair follicles due to infection or irritation."},
        {"name": "Head Lice", "image": "images/head_lice_0002.jpg", "description": "Parasitic infestation causing intense itching."},
        {"name": "Lichen Planus", "image": "images/lichen_planus_0003.jpg", "description": "Chronic inflammatory condition affecting the scalp."},
        {"name": "Male Pattern Baldness", "image": "images/male_pattern_baldness_0004.jpg", "description": "Hair thinning and loss in a characteristic pattern."},
        {"name": "Psoriasis", "image": "images/psoriasis_0005.jpg", "description": "Autoimmune condition with scaly, red patches on the scalp."},
        {"name": "Seborrheic Dermatitis", "image": "images/seborrheic_dermatitis_0003.jpg", "description": "Chronic condition causing greasy, yellow scales."},
        {"name": "Telogen Effluvium", "image": "images/telogen_effluvium_0005.jpg", "description": "Temporary hair loss due to stress or hormonal changes."},
        {"name": "Tinea Capitis", "image": "images/tineacapitis.jpg", "description": "Fungal infection causing circular bald patches."},
    ]
    return render_template('index.html', diseases=diseases)
@app.route('/alopecia_areata')
def alopecia_areata():
    return render_template('Alopecia Areata.html')
@app.route('/contact_dermatitis')
def contact_dermatitis():
    return render_template('Contact Dermatitis.html')
@app.route('/folliculitis')
def folliculitis():
    return render_template('Folliculitis.html')
@app.route('/head_lice')
def head_lice():
    return render_template('Head Lice.html')
@app.route('/lichen_planus')
def lichen_planus():
    return render_template('Lichen Planus.html')
@app.route('/male_pattern_baldness')
def male_pattern_baldness():
        return render_template('Male Pattern Baldness.html')
@app.route('/psoriasis')
def psoriasis():
    return render_template('Psoriasis.html')
@app.route('/seborrheic_dermatitis')
def seborrheic_dermatitis():
    return render_template('Seborrheic Dermatitis.html')
@app.route('/telogen_effluvium')
def telogen_effluvium():
    return render_template('Telogen Effluvium.html')
@app.route('/tinea_capitis')
def tinea_capitis():
    return render_template('Tinea Capitis.html')
# Add similar routes for other diseases

@app.route('/predict-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    if file:
        # Secure filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Simulate a prediction result
        prediction = "Alopecia"
        confidence = 0.95

        return render_template(
            'output.html',
            file_name=filename,
            prediction=prediction,
            confidence=confidence
        )

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    # Save the uploaded file in the static/uploads directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess and predict
    processed_image = preprocess_image(file_path)
    prediction = tf_model.predict(processed_image)
    predicted_class = classes[np.argmax(prediction)]
    confidence_score = float(np.max(prediction))

    # Pass the file name instead of the full path for rendering
    file_name = file.filename

    # Render the result
    return render_template('output.html',
                           prediction=predicted_class,
                           confidence=confidence_score,
                           file_name=file_name)


@app.route('/static/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/common-diseases')
def common_diseases():
    """Render common diseases page."""
    diseases = [
        {"name": "Alopecia Areata", "image": "images/Alopecia Areata.jpg", "description": "Patchy hair loss due to autoimmune response."},
        {"name": "Contact Dermatitis", "image": "images/contact_dermatitis_0002.jpg", "description": "Scalp irritation caused by allergens or irritants."},
        {"name": "Folliculitis", "image": "images/folliculitis_0005.jpg", "description": "Inflamed hair follicles due to infection or irritation."},
        {"name": "Head Lice", "image": "images/head_lice_0002.jpg", "description": "Parasitic infestation causing intense itching."},
        {"name": "Lichen Planus", "image": "images/lichen_planus_0003.jpg", "description": "Chronic inflammatory condition affecting the scalp."},
        {"name": "Male Pattern Baldness", "image": "images/male_pattern_baldness_0004.jpg", "description": "Hair thinning and loss in a characteristic pattern."},
        {"name": "Psoriasis", "image": "images/psoriasis_0005.jpg", "description": "Autoimmune condition with scaly, red patches on the scalp."},
        {"name": "Seborrheic Dermatitis", "image": "images/seborrheic_dermatitis_0003.jpg", "description": "Chronic condition causing greasy, yellow scales."},
        {"name": "Telogen Effluvium", "image": "images/telogen_effluvium_0005.jpg", "description": "Temporary hair loss due to stress or hormonal changes."},
        {"name": "Tinea Capitis", "image": "images/tineacapitis.jpg", "description": "Fungal infection causing circular bald patches."},
    ]
    return render_template('index.html', diseases=diseases)

@app.route('/predict-image')
def predict_image():
    """Render the predict image page."""
    return render_template('predict_image.html')

@app.route('/solutions')
def solutions():
    """Render the solutions page."""
    return render_template('solutions.html')

@app.route('/developer')
def developer():
    """Render the developer page."""
    return render_template('developer.html')  # You need to create this HTML file

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        user_input = request.json.get('input')

        # Generate response using Google Gemini API
        response = gen_model.generate_content(user_input)

        print(f"API Response: {response}")

        # Check if the response contains the expected field
        if hasattr(response, 'text'):
            # Remove unwanted special characters
            clean_response = ''.join(e for e in response.text if e.isalnum() or e.isspace() or e in ['.', ','])

            # Split response into sentences
            sentences = clean_response.split('.')

            # Get the first 3-4 sentences to provide a more detailed answer
            detailed_response = '. '.join(sentences[:4]).strip() + '.' if len(sentences) > 4 else '. '.join(sentences).strip()

            # Simplify and make it child-friendly
            detailed_response = detailed_response.replace("AI", "I")  # Replace AI with I
            detailed_response = detailed_response.replace("please", "can you tell me")  # Simplify requests

            return jsonify(response=detailed_response)

        else:
            return jsonify(error="Unexpected response structure: " + str(response))

    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e))

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
