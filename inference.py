import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
prediction_path = '/home/shephali/Desktop/Resnet50/prediction'  # Update with your prediction folder path
model_path = 'car_brand_clf_resnet50.h5'  # Update with your saved model path

# Load the saved model
model = load_model(model_path)

# Load the prediction dataset
prediction_dataset = image_dataset_from_directory(
    prediction_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'  # Load labels as integers for easier metrics calculation
)

# Extract true labels from the dataset
true_labels = np.concatenate([y for x, y in prediction_dataset], axis=0)

# Generate predictions
predictions = model.predict(prediction_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate evaluation metrics
accuracy = np.mean(predicted_labels == true_labels)
report = classification_report(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)

# Print and display results
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
