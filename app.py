from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

saved_model = load_model('ModelCNN9.h5')

labels = {
    0: "Ba", 1: "Ca", 2: "Da", 3: "Ga",
    4: "Ha", 5: "Ja", 6: "Ka", 7: "La",
    8: "Ma", 9: "Na", 10: "Nga", 11: "Nya",
    12: "Pa", 13: "Ra", 14: "Sa", 15: "Ta",
    16: "Wa", 17: "Ya"
}

# Folder tempat menyimpan file gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memproses gambar
def prepare_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0  # Normalisasi
    return x

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Simpan file yang diunggah
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Memproses gambar dan melakukan prediksi
            processed_image = prepare_image(filepath)
            predictions = saved_model.predict(processed_image)
            
            # Menentukan kelas berdasarkan hasil prediksi
            predicted_class = np.argmax(predictions[0])
            predicted_label = labels[predicted_class]
            confidence = float(np.max(predictions[0]))

            # Render hasil prediksi di halaman web
            return render_template('index.html', label=predicted_label, confidence=confidence, image_path=filepath)
    
    return render_template('index.html', label=None)

if __name__ == '__main__':
    app.run(debug=True)
