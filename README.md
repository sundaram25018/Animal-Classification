# Animal Classification using CNN 🐄🐃🐐

This project uses Convolutional Neural Networks (CNNs) to classify images of cows, buffaloes, and goats. The model is trained using TensorFlow/Keras and supports data augmentation for improved performance.

# 📌 Features
✅ Image classification for Cow, Buffalo, and Goat.

✅ Uses CNN for deep learning-based image recognition.

✅ Data Augmentation to handle small datasets.

✅ Supports real-time image prediction.

✅ Transfer Learning option for better accuracy.

# 📂 Dataset Structure
Your dataset should be structured as follows:

```File
dataset/
│── cow/
│   ├── cow1.jpg
│   ├── cow2.jpg
│── buffalo/
│   ├── buffalo1.jpg
│   ├── buffalo2.jpg
│── goat/
│   ├── goat1.jpg
│   ├── goat2.jpg
```
# 🚀 Installation
1️⃣ Clone this repository:
```bash
git clone https://github.com/sundaram25018/animal-classification-cnn.git
cd animal-classification-cnn
```
# 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
# 📊 Data Augmentation
If your dataset is small, you can increase it using augmentation:
```bash
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest"
)
```
### Augmented images are generated automatically during training.

🧠 Model Architecture (CNN)

The model consists of:

✔️ 3 Convolutional Layers (with ReLU activation)

✔️ MaxPooling Layers (to reduce dimensions)

✔️ Flatten & Dense Layers (for classification)

✔️ Softmax Output (3 classes: Cow, Buffalo, Goat)


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
# 🖼️ Training the Model
```python

history = model.fit(train_data, validation_data=val_data, epochs=20)
```

# To visualize accuracy/loss curves:

```python

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

```

# 🔍 Predicting an Image
To classify a new image, use the trained model:

```python

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("animal_classifier.h5")

img_path = "test_image.jpg"  
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
class_names = ["Cow", "Buffalo", "Goat"]
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicted Class: {predicted_class}")
```

# 📌 Future Improvements

✅ Fine-tuning CNN architecture

✅ Deploying the model using Streamlit

✅ Using a larger dataset for better accuracy

✅ Exploring MobileNetV2 for Transfer Learning


 
# 🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

# 📜 License
This project is MIT Licensed. Feel free to use and modify it.

