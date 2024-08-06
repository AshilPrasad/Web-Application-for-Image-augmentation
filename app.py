import os
from flask import Flask, request, send_file, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import zipfile
from io import BytesIO
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    Blur, RandomScale
)
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUGMENTED_FOLDER'] = 'augmented'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUGMENTED_FOLDER'], exist_ok=True)

def horizontal_flip(image):
    return HorizontalFlip(p=1.0)(image=image)['image']

def vertical_flip(image):
    return VerticalFlip(p=1.0)(image=image)['image']

def rotate(image):
    return Rotate(limit=30, p=1.0)(image=image)['image']

def brightness_contrast(image):
    return RandomBrightnessContrast(p=1.0)(image=image)['image']

def blur(image):
    return Blur(blur_limit=(3, 7), p=1.0)(image=image)['image']

def zoom(image):
    return RandomScale(scale_limit=0.9, p=1.0)(image=image)['image']

def augment_image(image):
    augmented_images = []
    image_np = np.array(image)
    
    for augmentation_name, func in zip(['horizontal_flip', 'vertical_flip', 'rotate', 'brightness_contrast', 'blur', 'zoom'],
                                       [horizontal_flip, vertical_flip, rotate, brightness_contrast, blur, zoom]):
        augmented = func(image_np)
        if augmented is not None:
            augmented_pil = Image.fromarray(augmented)
            augmented_images.append((augmentation_name, augmented_pil))
        else:
            print(f"Augmentation {augmentation_name} failed")
    return augmented_images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    if not files:
        return jsonify({'error': 'No selected file'}), 400

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to {filepath}")

        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': f'Error reading image file {filename}'}), 400
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        augmented_images = augment_image(pil_image)

        for aug_name, aug_img in augmented_images:
            aug_filename = f"{os.path.splitext(filename)[0]}_{aug_name}.jpg"
            aug_filepath = os.path.join(app.config['AUGMENTED_FOLDER'], aug_filename)
            aug_img.save(aug_filepath)
            print(f"Augmented image saved to {aug_filepath}")

    return jsonify({'message': 'File(s) successfully uploaded and augmented'}), 200

@app.route('/download')
def download_zip():
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for root, _, files in os.walk(app.config['AUGMENTED_FOLDER']):
            for file in files:
                zf.write(os.path.join(root, file), arcname=file)
    memory_file.seek(0)
    return send_file(memory_file, download_name='augmented_images.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
