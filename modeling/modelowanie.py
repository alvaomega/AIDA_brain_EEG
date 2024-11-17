import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Parameters
fs = 256  # Sampling frequency
window_size = 7 * fs  # Window size
overlap = 3 * fs  # Overlap size
channels = [f"ch_{i}" for i in range(1, 9)]
file_paths = ["amelka_uwu_ciekawe.csv", "amelka_uwu_nuda.csv", "amelka_uwu_sciana.csv"]  # Corrected filename
labels_list = ["ciekawe", "nuda", "baseline"]  # Ensure labels match filenames

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Preprocess and segment data
def preprocess_data(file_path, label):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None, None

    data = data.dropna()

    # Remove outliers using Z-score
    z_scores = np.abs(stats.zscore(data[channels]))
    data = data[(z_scores < 3).all(axis=1)]
    data = data.reset_index(drop=True)

    # Handle missing values by interpolation
    data = data.interpolate(method='linear').dropna()

    # Filter signals for each channel
    filtered_alpha = []
    filtered_beta = []

    for channel in channels:
        signal = data[channel]

        # Filter the signals
        alpha = bandpass_filter(signal, 8.0, 12.0, fs)
        beta = bandpass_filter(signal, 13.0, 30.0, fs)

        filtered_alpha.append(alpha)
        filtered_beta.append(beta)

    # Convert to numpy arrays
    filtered_alpha = np.array(filtered_alpha)
    filtered_beta = np.array(filtered_beta)

    # Segment the data with overlap
    segments = []
    for i in range(0, filtered_alpha.shape[1] - window_size, window_size - overlap):
        segment_alpha = filtered_alpha[:, i:i + window_size]
        segment_beta = filtered_beta[:, i:i + window_size]
        # Stack alpha and beta bands along the last dimension
        segment = np.stack([segment_alpha, segment_beta], axis=-1)
        segments.append(segment)

    segment_labels = [label] * len(segments)
    return segments, segment_labels

# Prepare the dataset
all_segments = []
all_labels = []

for file_path, label in zip(file_paths, labels_list):
    segments, segment_labels = preprocess_data(file_path, label)
    if segments is not None:
        print(f"Loaded {len(segments)} segments for label '{label}' from file '{file_path}'.")
        all_segments.extend(segments)
        all_labels.extend(segment_labels)
    else:
        print(f"No data loaded for label '{label}'.")

# Check if all labels are present
import collections
label_counts = collections.Counter(all_labels)
print("Label counts:", label_counts)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)
print("Encoded labels:", np.unique(encoded_labels))
categorical_labels = to_categorical(encoded_labels)

# Determine the number of classes dynamically
num_classes = len(label_encoder.classes_)
print("Number of classes:", num_classes)

# Ensure that categorical_labels has the correct shape
print("Categorical labels shape:", categorical_labels.shape)

# Convert to numpy arrays
X = np.array(all_segments)
y = np.array(categorical_labels)

# Reshape X for input into Conv1D and LSTM layers
num_samples, num_channels, window_size_data, num_features = X.shape
X = X.reshape(num_samples, window_size_data, num_channels * num_features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=encoded_labels
)

# Build the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                 input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(LSTM(32, return_sequences=True, kernel_regularizer=L2(0.01)))
model.add(Dropout(0.6))
model.add(LSTM(16, kernel_regularizer=L2(0.01)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plot training accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid()
plt.show()

# Save the model
model.save('eeg_classification_model.h5')
