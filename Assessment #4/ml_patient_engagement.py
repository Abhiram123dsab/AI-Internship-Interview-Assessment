import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import joblib

class MLPatientEngagement:
    """Machine Learning based patient engagement prediction system"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = [
            'age',
            'language_preference',
            'preferred_channel',
            'response_rate',
            'avg_response_time',
            'appointment_adherence',
            'message_open_rate',
            'time_of_day',
            'day_of_week'
        ]
        
    def prepare_features(self, patient_data):
        """Convert patient interaction data into ML features"""
        # Encode categorical variables
        language_map = {'Tamil': 0, 'Telugu': 1, 'Malayalam': 2, 'Hindi': 3, 'English': 4}
        channel_map = {'SMS': 0, 'WhatsApp': 1, 'IVR': 2}
        
        features = [
            patient_data['age'],
            language_map.get(patient_data['language'], -1),
            channel_map.get(patient_data['preferred_channel'], -1),
            patient_data['response_rate'],
            patient_data['avg_response_time'],
            patient_data['appointment_adherence'],
            patient_data['message_open_rate'],
            datetime.now().hour,
            datetime.now().weekday()
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, training_data):
        """Train the ML model with historical patient interaction data"""
        X = np.array([self.prepare_features(data)[0] for data in training_data])
        y = np.array([data['engagement_score'] for data in training_data])
        
        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(self.model, 'patient_engagement_model.joblib')
        joblib.dump(self.scaler, 'feature_scaler.joblib')
        
        return self.model.score(X_test_scaled, y_test)
    
    def predict_engagement(self, patient_data):
        """Predict patient engagement likelihood"""
        features = self.prepare_features(patient_data)
        scaled_features = self.scaler.transform(features)
        
        engagement_score = self.model.predict_proba(scaled_features)[0]
        
        return {
            'engagement_likelihood': float(engagement_score[1]),
            'recommended_channel': self.get_recommended_channel(engagement_score[1]),
            'best_time': self.predict_best_contact_time(patient_data)
        }
    
    def get_recommended_channel(self, engagement_score):
        """Recommend best communication channel based on engagement score"""
        if engagement_score >= 0.8:
            return 'WhatsApp'  # High engagement - use rich media
        elif engagement_score >= 0.4:
            return 'SMS'       # Medium engagement - use simple text
        else:
            return 'IVR'       # Low engagement - use voice calls
    
    def predict_best_contact_time(self, patient_data):
        """Predict the best time to contact the patient"""
        # Analyze historical response patterns
        response_times = patient_data.get('historical_response_times', [])
        if not response_times:
            # Default to common working hours if no history
            return {
                'hour': 10,
                'day_of_week': 1  # Monday
            }
        
        # Find the hour with highest response rate
        response_hours = [dt.hour for dt in response_times]
        best_hour = max(set(response_hours), key=response_hours.count)
        
        # Find the day with highest response rate
        response_days = [dt.weekday() for dt in response_times]
        best_day = max(set(response_days), key=response_days.count)
        
        return {
            'hour': best_hour,
            'day_of_week': best_day
        }