# Required libraries
import numpy as np
import pandas as pd
import os
import random
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xml.dom import minidom
import csv
from ultralytics import YOLO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# Directories
image_dir = 'C:\\Weapon_Detection\\images'
annot_dir = 'C:\\Weapon_Detection\\annotations\\xmls'


# Data rescaling
def rescaling(path_image, targetSize, xmin, ymin, xmax, ymax):
    imageToPredict = cv2.imread(path_image)
    y_, x_ = imageToPredict.shape[:2]
    
    x_scale = targetSize / x_
    y_scale = targetSize / y_
    img = cv2.resize(imageToPredict, (targetSize, targetSize))
    
    # Scale the bounding box coordinates
    xmin = int(np.round(xmin * x_scale))
    ymin = int(np.round(ymin * y_scale))
    xmax = int(np.round(xmax * x_scale))
    ymax = int(np.round(ymax * y_scale))
    
    return img, xmin, ymin, xmax, ymax

def extract_xml_contents(annot_directory, image_dir, target_size=300):
    file = minidom.parse(annot_directory)
    
    # Extract bounding box coordinates
    xmin = float(file.getElementsByTagName('xmin')[0].firstChild.data)
    ymin = float(file.getElementsByTagName('ymin')[0].firstChild.data)
    xmax = float(file.getElementsByTagName('xmax')[0].firstChild.data)
    ymax = float(file.getElementsByTagName('ymax')[0].firstChild.data)
    
    # Update class assignment
    class_name = file.getElementsByTagName('name')[0].firstChild.data
    if class_name == 'knife':
        class_num = 0
    elif class_name == 'pistol':
        class_num = 1
    else:  # Assuming 'no weapon'
        class_num = 2
    
    file_name = file.getElementsByTagName('filename')[0].firstChild.data
    
    # Rescale images and bounding boxes
    img, xmin, ymin, xmax, ymax = rescaling(image_dir, target_size, xmin, ymin, xmax, ymax)
    return file_name, img.shape[1], img.shape[0], class_num, xmin, ymin, xmax, ymax

def xml_to_csv(image_dir, annot_dir):
    xml_list = []
    img_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for image in img_files:
        mat_path = os.path.join(annot_dir, (image.split('.')[0] + '.xml'))
        img_path = os.path.join(image_dir, image)

        # Check if the corresponding XML file exists
        if os.path.exists(mat_path):
            try:
                value = extract_xml_contents(mat_path, img_path)
                xml_list.append(value)
            except Exception as e:
                print(f"Error processing file {mat_path}: {e}")
        else:
            print(f"Warning: No XML annotation found for {image}")

    # Define column names
    columns_name = ['file_name', 'width', 'height', 'class_num', 'xmin', 'ymin', 'xmax', 'ymax']

    # Convert list to DataFrame
    df = pd.DataFrame(xml_list, columns=columns_name)
    return df

# Save annotations to CSV
train_labels_df = xml_to_csv(image_dir, annot_dir)
train_labels_df.to_csv('L_dataset.csv', index=None)
print('L_dataset.csv is created successfully')



# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Generator function to load and preprocess images in batches with augmentation
def image_generator(image_dir, csv_file, batch_size=16, img_size=(300, 300)):
    with open(csv_file) as csvfile:
        rows = list(csv.reader(csvfile))[1:]

    while True:
        random.shuffle(rows)
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i + batch_size]
            img_batch, box_batch, label_batch = [], [], []

            for row in batch_rows:
                img_path = row[0]
                full_path = os.path.join(image_dir, img_path)
                img = cv2.imread(full_path)
                
                if img is not None:
                    # Preprocess image and bounding box
                    img, box = preprocess_image(img, [float(row[4]), float(row[5]), float(row[6]), float(row[7])], img_size)

                    # Apply augmentation
                    img = train_datagen.random_transform(img)
                    img_batch.append(img)
                    box_batch.append(box)
                    label_batch.append(int(row[3]))

            yield np.array(img_batch), np.array(box_batch), np.array(label_batch)


# Custom Attention Layer
class CustomAttention(Layer):
    def __init__(self):
        super(CustomAttention, self).__init__()

    def call(self, inputs):
        q = tf.expand_dims(inputs, axis=1)
        k = tf.expand_dims(inputs, axis=2)
        v = inputs
        # Calculate attention scores
        attention_scores = tf.matmul(q, k)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        v_expanded = tf.expand_dims(v, axis=1)
        # Apply attention weights
        output = tf.matmul(attention_weights, v_expanded)
        return tf.squeeze(output, axis=1)



# Load NASNetMobile for feature extraction
input_shape = (300, 300, 3)
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)

# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Add global average pooling
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Use the custom attention layer
attention_output = CustomAttention()(x)
# Add a dense layer for classification
x = Dense(256, activation='relu')(attention_output)
x = Dense(3, activation='softmax')(x)

# Create the final model
feature_extraction_model = Model(inputs=base_model.input, outputs=x)

# Compile the model
feature_extraction_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Load and preprocess dataset
def load_and_preprocess_data():
    images, labels, boxes = [], [], []
    gen = image_generator(image_dir, 'L_dataset.csv', batch_size=8, img_size=(300, 300))

    total_rows = sum(1 for row in open('L_dataset.csv')) - 1
    steps = total_rows // 32 + 1 

    for step in range(steps):
        img_batch, box_batch, label_batch = next(gen)
        images.extend(img_batch)
        labels.extend(label_batch)
        boxes.extend(box_batch)
    
    return np.array(images), np.array(labels), np.array(boxes)


# Load data
X_train, y_train, _ = load_and_preprocess_data()

# Fit the model with EarlyStopping and ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)


# Define dataset paths
image_dir = 'C:\\WeaponDetection\\images'
csv_file = 'L_dataset.csv'

# Load all rows from the CSV file
with open(csv_file) as csvfile:
    rows = list(csv.reader(csvfile))[1:]

# Split into training and validation sets (80% train, 20% validation)
train_rows, val_rows = train_test_split(rows, test_size=0.2, random_state=42)

# Print the total number of training and validation images
print(f'Total Training Images: {len(train_rows)}, Total Validation Images: {len(val_rows)}')



# Train the model
history = feature_extraction_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)


# Save the model
feature_extraction_model.save('weapon_detection_model.h5')


# Visualizing training history
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Plot the training history
plot_training_history(history)


# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


with open(csv_file) as csvfile:
    rows = list(csv.reader(csvfile))[1:] 

# Split into training and test sets (80% train, 20% test)
train_rows, test_rows = train_test_split(rows, test_size=0.2, random_state=42)

# Separate features (X) and labels (y) for the test set
X_test = []
y_test = []
for row in test_rows:
    img_path = row[0]  # Assuming the first column contains the image paths
    full_path = os.path.join(image_dir, img_path)
    img = cv2.imread(full_path)
    
    if img is not None:
        img = cv2.resize(img, (300, 300)).astype("float32") / 255.0  
        X_test.append(img)
        y_test.append(int(row[3])) 

X_test = np.array(X_test)
y_test = np.array(y_test)

# Get predictions for the test set
y_pred = np.argmax(feature_extraction_model.predict(X_test), axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plot_confusion_matrix(cm, classes=['Knife', 'Pistol', 'No Weapon'], title='Confusion Matrix')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Assuming you have 3 classes: 'Knife', 'Pistol', 'No Weapon'
n_classes = 3

# Binarize the test labels for ROC/AUC computation
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Replace with appropriate label values for your case

# Get the model's probability predictions (not class predictions) for each class
y_pred_proba = feature_extraction_model.predict(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()

colors = ['blue', 'green', 'red']  # Different colors for each class
class_names = ['Knife', 'Pistol', 'No Weapon']  # Adjust based on your classes

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve (area = {roc_auc[i]:.2f}) for {class_names[i]}')

# Plot random guess line
plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-Class Classification')
plt.legend(loc="lower right")
plt.show()

# Compute macro and micro AUC scores for overall model performance
macro_roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro')
micro_roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='micro')

print(f'Macro AUC: {macro_roc_auc:.2f}')
print(f'Micro AUC: {micro_roc_auc:.2f}')


# Extract features from the last layer before classification
feature_model = Model(inputs=feature_extraction_model.input, outputs=feature_extraction_model.layers[-2].output)

# Get feature outputs for the test data
features = feature_model.predict(X_test)

# Create a DataFrame from the feature array for better visualization
feature_df = pd.DataFrame(features)

# Compute the correlation matrix
corr_matrix = feature_df.corr()

# Set the background to white
sns.set(style="white")

# Assuming 'corr_matrix' is your correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")

# Add a title to the plot
plt.title("Feature Correlation Matrix", fontsize=16)

# Show the plot
plt.show()


# Get predictions for the test set
y_pred = np.argmax(feature_extraction_model.predict(X_test), axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Convert confusion matrix into a correlation-like format
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create a DataFrame for visualization
cm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Matrix based on Confusion Matrix")
plt.show()

from keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(300, 300))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

features = []
labels = []

for row in train_rows:
    img_path = row[0]  
    full_path = os.path.join(image_dir, img_path)
    img = load_and_preprocess_image(full_path)  
    feature = feature_extraction_model.predict(img) 
    features.append(feature[0]) 
    labels.append(row[1])

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features)

# Check shapes
print(f'tsne_results shape: {tsne_results.shape}')
print(f'encoded_labels shape: {encoded_labels.shape}')

# Plotting t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=encoded_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of extracted features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()



from ultralytics import YOLO

# Load the YOLOv8n model pretrained on COCO dataset
model = YOLO('yolov8n.pt') 

# Set fine-tuning parameters
epochs = 100  
patience = 10  
batch_size = 32 
input_size = 300  

# Train the model with fine-tuning and early stopping
history = model.train(
    data='weapon_detection_config.yaml',  
    epochs=epochs,
    batch=batch_size,
    imgsz=input_size,  
    optimizer='Adam',  
    patience=patience,  
    lr0=1e-3, 
    augment=True,  
    project="WeaponDetectionYOLO",
    name="yolov8n_finetuning",
    verbose=True,
    save=True  
)


# Evaluate the model on the validation set
metrics = model.val(data='weapon_detection_config.yaml')

# Display Precision, Recall, and F1 Score
precision = metrics.box.map50 
recall = metrics.box.map50 
f1_score = (2 * precision * recall) / (precision + recall + 1e-6)
print(f"Precision (mAP@50): {precision:.4f}")
print(f"Recall (mAP@50): {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")


import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the dataset directory and class labels
dataset_dir = 'C://DataSampleFolder//images'  # Adjust path based on your setup
classes = ['knife', 'pistol', 'no_weapon']  # Your class names

# Parameters for the grid visualization
num_images_per_class = 10  # Number of images to display per class
image_size = (128, 128)  # Resizing the images for display
rows = len(classes)  # One row per class
cols = num_images_per_class  # Number of columns based on images per class

# Create a figure for displaying the images
fig, axes = plt.subplots(rows, cols, figsize=(15, 8))

# Function to load and resize an image
def load_and_resize_image(img_path, img_size):
    img = Image.open(img_path)
    img = img.resize(img_size)
    return np.array(img)

# Loop through each class and display images
for row, cls in enumerate(classes):
    # Get the image paths for the current class
    class_dir = os.path.join(dataset_dir, cls)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select random images from the class
    selected_images = np.random.choice(images, num_images_per_class, replace=False)
    
    for col, img_path in enumerate(selected_images):
        img = load_and_resize_image(img_path, image_size)
        
        # Display the image in the grid
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    # Add the class name under each row
    axes[row, 0].set_ylabel(cls, fontsize=12, labelpad=20, rotation=0, ha='right')

# Adjust layout
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)

# Title below the figure
fig.text(0.5, 0.05, 'Sample of the Dataset', ha='center', fontsize=14)

# Show the plot
plt.show()

# Load YOLOv8 model
yolo_model = YOLO('WeaponDetectionYOLO/yolov8n_finetuning/weights/best.pt')
# Load your feature extraction + classification model 
feature_extraction_model = load_model('weapon_detection_model.h5')

# Visualization function
def visualize_detections(img, boxes, class_ids, confidences):
    class_labels = {0: 'Knife', 1: 'Pistol', 2: 'No Weapon'}
    
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = map(int, box[:4])  
        conf = confidences[i]
        class_id = int(class_ids[i])
        
        # Draw the bounding box in red
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
        
        # Get the label from class ID
        label = class_labels.get(class_id, 'Unknown')
        
        # Draw the class label and confidence on the image
        cv2.putText(img, f'{label}, Conf: {conf:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Load and process the input image
image_path = 'C:\\Weapon_Detection\\images\\pistol_5180.jpg'
if os.path.exists(image_path):
   
    img = cv2.imread(image_path)

    # Perform inference with YOLOv8
    
    results = yolo_model.predict(image_path)

    # Extract boxes, class IDs, and confidences
    boxes = []
    class_ids = []
    confidences = []

    for result in results
        for box in result.boxes.data.tolist():  
            boxes.append(box)  
            confidences.append(box[4])  
            class_ids.append(int(box[5]))  

    # Process each detected bounding box
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = map(int, box[:4])  

        # Extract the region of interest (ROI) from the image
        roi = img[y1:y2, x1:x2]
        
        # Skip processing if ROI is empty
        if roi.size == 0:
            continue

        # Preprocess ROI for feature extraction
        roi_preprocessed = preprocess_image(roi)

        # Get classification result from the feature extraction model
        predictions = feature_extraction_model.predict(roi_preprocessed)
        predicted_class = np.argmax(predictions, axis=-1)

        print(f'Detected {i}: YOLO class: {class_ids[i]}, NASNetMobile predicted class: {predicted_class}
        confidence(":", "{np.max(predictions)}')")

    # Visualize detections on the original image
    visualize_detections(img, boxes, class_ids, confidences)

    # Define the output path
    output_dir = 'C:\\Weapon_Detection\\output'
    output_image_path = os.path.join(output_dir, 'detected_image.jpg') 

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    # Save the image with bounding boxes
    cv2.imwrite(output_image_path, img)
    print(f"Output image saved to {output_image_path}")
else:
    print(f"File not found: {image_path}")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming you have the following lists from your model's output:
y_true = [] 
y_pred = [] 
confidences = []  

# Simulated data for demonstration purposes
# You should replace this with your actual data
y_true = [0, 1, 2, 1, 0, 2, 0, 1, 1, 2]  
y_pred = [0, 1, 2, 0, 0, 2, 1, 1, 1, 2]  
confidences = [0.95, 0.88, 0.76, 0.80, 0.92, 0.87, 0.60, 0.90, 0.95, 0.82]  

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=["Knife", "Pistol", "No Weapon"])
plt.figure(figsize=(8, 6))
cm_display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 2. Confidence Scores Distribution
plt.figure(figsize=(10, 6))
sns.histplot(confidences, bins=10, kde=True, color='blue')
plt.title("Distribution of Confidence Scores")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.axvline(x=0.80, color='red', linestyle='--', label='Threshold = 0.80') 
plt.legend()
plt.show()

# 3. Box Counts per Class
unique, counts = np.unique(y_pred, return_counts=True)
class_counts = dict(zip(unique, counts))

plt.figure(figsize=(8, 6))
plt.bar(class_counts.keys(), class_counts.values(), color=['green', 'red', 'blue'])
plt.title("Number of Detections per Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1, 2], labels=["Knife", "Pistol", "No Weapon"])
plt.show()

# 4. True Positives vs. False Positives
tp = [1 if true == pred else 0 for true, pred in zip(y_true, y_pred)]
fp = [1 if true != pred else 0 for true, pred in zip(y_true, y_pred)]

plt.figure(figsize=(10, 6))
plt.bar(['True Positives', 'False Positives'], [sum(tp), sum(fp)], color=['green', 'red'])
plt.title("True Positives vs. False Positives")
plt.ylabel("Count")
plt.show()





