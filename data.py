import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_sample_csv(file_path):
    if not os.path.exists(file_path):
        sample_data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'heart_rate': np.random.randint(60, 100, size=100),
            'activity_level': np.random.randint(1000, 10000, size=100),
            'sleep_duration': np.random.uniform(5, 9, size=100),
            'sleep_quality': np.random.uniform(1, 10, size=100),
            'room_temperature': np.random.uniform(18, 25, size=100),
            'caffeine_consumption': np.random.randint(0, 3, size=100),
            'alcohol_consumption': np.random.randint(0, 3, size=100),
            'screen_time': np.random.uniform(0, 3, size=100)
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(file_path, index=False)
        print(f"Sample CSV file created at {file_path}")

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.dropna()
    features = data[['heart_rate', 'activity_level', 'sleep_duration', 'room_temperature', 'caffeine_consumption', 'alcohol_consumption', 'screen_time']]
    target = data['sleep_quality']
    return features, target

def feature_engineering(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    return model

def predict_sleep_quality(model, new_data):
    prediction = model.predict(new_data)
    return prediction

def plot_sleep_data(data):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 3, 1)
    sns.lineplot(x='timestamp', y='heart_rate', data=data)
    plt.title('Heart Rate Over Time')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate')
    
    plt.subplot(2, 3, 2)
    sns.lineplot(x='timestamp', y='activity_level', data=data)
    plt.title('Activity Level Over Time')
    plt.xlabel('Time')
    plt.ylabel('Activity Level')
    
    plt.subplot(2, 3, 3)
    sns.histplot(data['sleep_duration'], bins=10, kde=True)
    plt.title('Sleep Duration Distribution')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 4)
    sns.histplot(data['sleep_quality'], bins=10, kde=True)
    plt.title('Sleep Quality Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 5)
    sns.lineplot(x='timestamp', y='room_temperature', data=data)
    plt.title('Room Temperature Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.show()

def get_user_feedback():
    while True:
        try:
            feedback = float(input("Rate the accuracy of the sleep quality prediction (1-10): "))
            if 1 <= feedback <= 10:
                return feedback
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 10.")

def provide_recommendations(sleep_quality):
    if sleep_quality < 5:
        print("Recommendation: Try to maintain a consistent sleep schedule and avoid caffeine before bed.")
    elif sleep_quality < 7:
        print("Recommendation: Ensure your bedroom is cool, quiet, and dark. Consider using a sleep mask or earplugs.")
    else:
        print("Recommendation: Keep up the good work! Continue following healthy sleep habits.")

def detect_anomalies(data):
    anomalies = data[(data['sleep_quality'] < data['sleep_quality'].mean() - 2 * data['sleep_quality'].std()) | 
                     (data['sleep_quality'] > data['sleep_quality'].mean() + 2 * data['sleep_quality'].std())]
    return anomalies

if __name__ == "__main__":
    file_path = 'fitness_data.csv'
    
    create_sample_csv(file_path)
    
    data = load_data(file_path)
    features, target = preprocess_data(data)
    features = feature_engineering(features)
    
    model = train_and_evaluate_model(features, target)
    
    new_data = np.array([[70, 5000, 7, 22, 1, 0, 2]])  
    prediction = predict_sleep_quality(model, new_data)
    print(f'Predicted Sleep Quality: {prediction[0]}')
    
    user_feedback = get_user_feedback()
    print(f'User Feedback: {user_feedback}')
    
    provide_recommendations(prediction[0])
    
    anomalies = detect_anomalies(data)
    if not anomalies.empty:
        print("Anomalies detected in sleep data:")
        print(anomalies)
    
    plot_sleep_data(data)