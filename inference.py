import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("fish_presence_model.h5")

def predict_fish(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    if prob > 0.5:
        return f"Fish Present (confidence: {prob:.2f})"
    else:
        return f"No Fish (confidence: {1-prob:.2f})"

print(predict_fish("test.jpg"))
