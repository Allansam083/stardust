import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = load_model('ann_mmodel.h5')

# Assuming you have the necessary encoders and scalers saved from training phase
product_encoder = LabelEncoder()
age_group_encoder = LabelEncoder()
scaler = StandardScaler()

# Function to deploy the model
def deploy_model(input_data):
    """
    Predicts whether a product is healthy or unhealthy based on the input data and returns the result.
    input_data: Array-like, input data to predict (should match the input format used during training)
    output: Formatted string like '78% healthy for the given age category'
    """

    # Step 1: Preprocess the input
    # Assuming input_data is an array like [Product, Age Group, Calories, Sugar, Protein, Fiber, Fat]
    product_name = input_data[0]  # Product
    age_group = input_data[1]  # Age Group
    calories = input_data[2]
    sugar = input_data[3]
    protein = input_data[4]
    fiber = input_data[5]
    fat = input_data[6]

    # Encode 'Product' and 'Age Group'
    product_encoded = product_encoder.transform([product_name])[0]
    age_group_encoded = age_group_encoder.transform([age_group])[0]

    # Scale the numeric values
    input_features = np.array([calories, sugar, protein, fiber, fat]).reshape(1, -1)
    input_features = scaler.transform(input_features)

    # Combine the preprocessed features into a single array
    input_data_preprocessed = np.array([product_encoded, age_group_encoded] + input_features[0].tolist()).reshape(1, -1)

    # Step 2: Predict using the model
    prediction_prob = model.predict(input_data_preprocessed)  # Predict probability for Healthy (1)
    prediction_label = (prediction_prob > 0.5).astype(int)  # 1 for healthy, 0 for unhealthy

    # Step 3: Format the output
    health_status = 'Healthy' if prediction_label == 1 else 'Unhealthy'
    health_percentage = prediction_prob[0][0] * 100  # Convert to percentage

    # Format the output as per the required output
    return f"{health_percentage:.2f}% {health_status} for the given age category"

# Example usage:
input_example = ['Bourbon', '6-12', 200, 12.5, 3.2, 1.5, 8.0]  
output = deploy_model(input_example)
print(output)  # Example output: '78.32% Healthy for the given age category'
