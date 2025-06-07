
# Load the model for predictions
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image


loaded_model = load_model('crop_disease_model.h5')
image_size=(224,224)
# Example: Predict on a new image

class_labels = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

def predict_disease(image_path):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = loaded_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    class_label = class_labels[predicted_class[0]]
    return class_label

# Test the function
image_path = "image.png"
print(f"Predicted Disease: {predict_disease(image_path)}".encode('utf-8', errors='ignore').decode())
