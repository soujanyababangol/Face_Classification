import tensorflow as tf
import os

TRAIN_FOLDER = "dataset_split/train"
VALIDATION_FOLDER = "dataset_split/validation"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15

train_data = tf.keras.utils.image_dataset_from_directory(TRAIN_FOLDER,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_FOLDER,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical")

class_names = train_data.class_names
print("Classes:", class_names)
model_base = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet")

model_base.trainable = False
model = tf.keras.Sequential([
    model_base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation="softmax")])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()
model.fit(train_data,validation_data=validation_data,epochs=EPOCHS)
os.makedirs("model", exist_ok=True)
model.save("model/face_classifier.keras")
print("\nModel training completed.")
print("Saved model: model/face_classifier.keras")