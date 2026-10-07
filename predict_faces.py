import tensorflow as tf
import cv2
import numpy as np
import os
import shutil

MODEL_PATH = "model/face_classifier.keras"
INPUT_FOLDER = "image_input"
OUTPUT_FOLDER = "image_classification_output"
CLASS_NAMES = ["mask", "mask_sunglasses", "neutral", "sunglasses"]

for class_name in CLASS_NAMES:
    os.makedirs(
        os.path.join(OUTPUT_FOLDER, class_name),
        exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)
print("Classification model loaded.")
for image_file in os.listdir(INPUT_FOLDER):
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    image_path = os.path.join(INPUT_FOLDER, image_file)
    image = cv2.imread(image_path)

    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image, verbose=0)
    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index]
    predicted_class = CLASS_NAMES[class_index]
    output_path = os.path.join(
        OUTPUT_FOLDER,
        predicted_class,
        image_file)

    shutil.copy2(os.path.join(INPUT_FOLDER, image_file),output_path)

    print(image_file,"->",predicted_class,f"({confidence * 100:.2f}%)")

print("\nClassification completed.")
print("Results saved in:", OUTPUT_FOLDER)