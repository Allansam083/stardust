import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # For oversampling

# Load your dataset (replace with your actual dataset)
df = pd.read_csv('huge_snack_products_dataset.csv')

# Data Preprocessing
# Assume 'df' is your DataFrame, and the necessary columns are already set.

# Encode 'Product' and 'Age Group'
product_encoder = LabelEncoder()
age_group_encoder = LabelEncoder()
df['Product'] = product_encoder.fit_transform(df['Product'])
df['Age Group'] = age_group_encoder.fit_transform(df['Age Group'])

# Scale numeric columns
scaler = StandardScaler()
numeric_columns = ['Calories', 'Sugar (g)', 'Protein (g)', 'Fiber (g)', 'Fat (g)']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Encode the target column (Healthy/Unhealthy)
target_encoder = LabelEncoder()
df['Healthy/Unhealthy'] = target_encoder.fit_transform(df['Healthy/Unhealthy'])

# Split the dataset into features (X) and target (y)
X = df.drop(columns=['Healthy/Unhealthy'])
y = df['Healthy/Unhealthy']

# **Balancing the Dataset using SMOTE (Oversampling the minority class)**
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# **Early Stopping Callback**: Stop training when validation loss doesn't improve for 5 consecutive epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('ann_model.h5')

# Evaluate the model on the test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary classification

# Classification Report
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualization of training history
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

