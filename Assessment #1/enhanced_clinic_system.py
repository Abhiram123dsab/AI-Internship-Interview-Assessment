import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta, timezone
import re
import random
import json
from twilio.rest import Client
import pywhatkit
import requests
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Initialize Flask app for real-time dashboard
app = Flask(__name__)
socketio = SocketIO(app)

# Load appointment data
def load_data():
    try:
        df = pd.read_csv("appointments.csv")  # Contains scheduled_time, actual_time, doctor_id, patient_id
        return df
    except FileNotFoundError:
        # Create sample data for testing if file doesn't exist
        print("Warning: appointments.csv not found. Creating sample data for demonstration.")
        # Generate sample data
        num_samples = 1000
        doctors = list(range(1, 16))  # 15 doctors
        start_date = datetime(2024, 1, 1)
        
        scheduled_times = []
        actual_times = []
        doctor_ids = []
        patient_ids = []
        
        for i in range(num_samples):
            # Generate random date within 3 months
            days_offset = random.randint(0, 90)
            hours_offset = random.randint(9, 20)  # Clinic hours 9am-8pm
            minutes_offset = random.randint(0, 59)
            
            scheduled_time = start_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
            
            # Add delay (0-60 minutes)
            delay_minutes = random.randint(0, 60)
            actual_time = scheduled_time + timedelta(minutes=delay_minutes)
            
            scheduled_times.append(scheduled_time)
            actual_times.append(actual_time)
            doctor_ids.append(random.choice(doctors))
            patient_ids.append(i + 1000)  # Patient IDs starting from 1000
        
        sample_df = pd.DataFrame({
            'scheduled_time': scheduled_times,
            'actual_time': actual_times,
            'doctor_id': doctor_ids,
            'patient_id': patient_ids
        })
        
        # Convert datetime to string format
        sample_df['scheduled_time'] = sample_df['scheduled_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        sample_df['actual_time'] = sample_df['actual_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save sample data
        sample_df.to_csv("appointments.csv", index=False)
        return sample_df

# Load or create sample data
df = load_data()

# Feature Engineering
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert to datetime
    df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    
    # Calculate delay
    df['delay'] = (df['actual_time'] - df['scheduled_time']).dt.total_seconds() / 60
    
    # Extract time features
    df['hour'] = df['scheduled_time'].dt.hour
    df['day_of_week'] = df['scheduled_time'].dt.dayofweek
    df['month'] = df['scheduled_time'].dt.month
    df['day'] = df['scheduled_time'].dt.day
    df['is_peak'] = df['hour'].between(17, 20)
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Calculate historical patterns
    df['historical_delay'] = df.groupby(['doctor_id', 'hour'])['delay'].transform('mean')
    df['peak_hour_delay'] = df.groupby(['doctor_id', 'is_peak'])['delay'].transform('mean')
    df['seasonal_pattern'] = df.groupby(['doctor_id', 'month'])['delay'].transform('mean')
    df['day_of_week_pattern'] = df.groupby(['doctor_id', 'day_of_week'])['delay'].transform('mean')
    df['weekend_pattern'] = df.groupby(['doctor_id', 'is_weekend'])['delay'].transform('mean')
    
    # Calculate doctor-specific consultation patterns
    doctor_stats = df.groupby('doctor_id')['delay'].agg(['mean', 'std', 'median', 'count']).reset_index()
    doctor_stats.columns = ['doctor_id', 'doc_mean_delay', 'doc_delay_std', 'doc_median_delay', 'doc_appointment_count']
    df = pd.merge(df, doctor_stats, on='doctor_id')
    
    # Calculate patient arrival deviations
    df['arrival_deviation'] = (df['actual_time'] - df['scheduled_time']).dt.total_seconds() / 60
    
    # Calculate rolling averages for recent trends
    df = df.sort_values('scheduled_time')
    df['recent_delay_trend'] = df.groupby('doctor_id')['delay'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64] and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    return df

# Preprocess data
df_processed = preprocess_data(df)

# Define features and target variable
features = [
    'doctor_id', 'hour', 'day_of_week', 'month', 'day', 'is_peak', 'is_weekend',
    'doc_mean_delay', 'doc_delay_std', 'doc_median_delay', 'doc_appointment_count',
    'historical_delay', 'peak_hour_delay', 'seasonal_pattern', 'day_of_week_pattern',
    'weekend_pattern', 'recent_delay_trend'
]
target = 'delay'

# Train AI Model with advanced techniques
def train_advanced_model(df, features, target):
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Split data for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Train Random Forest model with hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_model = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    
    # 2. Train Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # 3. Train Neural Network model
    nn_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate models
    rf_pred = best_rf.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    nn_pred = nn_model.predict(X_test_scaled).flatten()
    
    # Ensemble prediction (weighted average)
    ensemble_pred = 0.5 * rf_pred + 0.3 * gb_pred + 0.2 * nn_pred
    
    # Calculate metrics
    rf_mae = mean_absolute_error(y_test, rf_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    nn_mae = mean_absolute_error(y_test, nn_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"Random Forest MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
    print(f"Gradient Boosting MAE: {gb_mae:.2f}, RMSE: {gb_rmse:.2f}")
    print(f"Neural Network MAE: {nn_mae:.2f}, RMSE: {nn_rmse:.2f}")
    print(f"Ensemble MAE: {ensemble_mae:.2f}, RMSE: {ensemble_rmse:.2f}")
    
    # Save models
    joblib.dump(best_rf, 'rf_model.joblib')
    joblib.dump(gb_model, 'gb_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    nn_model.save('nn_model')
    
    # Store feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 important features:")
    print(feature_importance.head(10))
    
    return {
        'rf_model': best_rf,
        'gb_model': gb_model,
        'nn_model': nn_model,
        'scaler': scaler,
        'metrics': {
            'rf_mae': rf_mae,
            'gb_mae': gb_mae,
            'nn_mae': nn_mae,
            'ensemble_mae': ensemble_mae,
            'rf_rmse': rf_rmse,
            'gb_rmse': gb_rmse,
            'nn_rmse': nn_rmse,
            'ensemble_rmse': ensemble_rmse
        },
        'feature_importance': feature_importance
    }

# Train models if there's enough data
if len(df_processed) > 100:  # Only train if we have sufficient data
    model_results = train_advanced_model(df_processed, features, target)
    rf_model = model_results['rf_model']
    gb_model = model_results['gb_model']
    nn_model = model_results['nn_model']
    scaler = model_results['scaler']
    model_metrics = model_results['metrics']
    feature_importance = model_results['feature_importance']
else:
    # Use default RandomForest model for small datasets
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(df_processed[features], df_processed[target])
    gb_model = None
    nn_model = None
    scaler = None
    model_metrics = {}

# Enhanced prediction function using ensemble model
def predict_wait_time_enhanced(doctor_id, scheduled_time):
    # Convert to pandas Series for .dt accessor
    time_series = pd.to_datetime(pd.Series([scheduled_time]))
    hour = time_series.dt.hour.iloc[0]
    day_of_week = time_series.dt.dayofweek.iloc[0]
    month = time_series.dt.month.iloc[0]
    day = time_series.dt.day.iloc[0]
    is_peak = hour >= 17 and hour <= 20
    is_weekend = day_of_week >= 5
    
    # Get doctor statistics
    doc_stats = df_processed[df_processed['doctor_id'] == doctor_id].iloc[0] if len(df_processed[df_processed['doctor_id'] == doctor_id]) > 0 else None
    
    if doc_stats is None:
        # Use average values if doctor not found
        doc_mean_delay = df_processed['delay'].mean()
        doc_delay_std = df_processed['delay'].std()
        doc_median_delay = df_processed['delay'].median()
        doc_appointment_count = 10  # Default value
        historical_delay = df_processed[df_processed['hour'] == hour]['delay'].mean()
        peak_hour_delay = df_processed[df_processed['is_peak'] == is_peak]['delay'].mean()
        seasonal_pattern = df_processed[df_processed['month'] == month]['delay'].mean()
        day_of_week_pattern = df_processed[df_processed['day_of_week'] == day_of_week]['delay'].mean()
        weekend_pattern = df_processed[df_processed['is_weekend'] == is_weekend]['delay'].mean()
        recent_delay_trend = df_processed['delay'].mean()
    else:
        # Use doctor-specific statistics
        doc_mean_delay = doc_stats['doc_mean_delay']
        doc_delay_std = doc_stats['doc_delay_std']
        doc_median_delay = doc_stats['doc_median_delay']
        doc_appointment_count = doc_stats['doc_appointment_count']
        
        # Calculate historical patterns
        historical_delay = df_processed[(df_processed['doctor_id'] == doctor_id) & 
                                      (df_processed['hour'] == hour)]['delay'].mean()
        peak_hour_delay = df_processed[(df_processed['doctor_id'] == doctor_id) & 
                                     (df_processed['is_peak'] == is_peak)]['delay'].mean()
        seasonal_pattern = df_processed[(df_processed['doctor_id'] == doctor_id) & 
                                      (df_processed['month'] == month)]['delay'].mean()
        day_of_week_pattern = df_processed[(df_processed['doctor_id'] == doctor_id) & 
                                         (df_processed['day_of_week'] == day_of_week)]['delay'].mean()
        weekend_pattern = df_processed[(df_processed['doctor_id'] == doctor_id) & 
                                     (df_processed['is_weekend'] == is_weekend)]['delay'].mean()
        recent_delay_trend = df_processed[df_processed['doctor_id'] == doctor_id]['recent_delay_trend'].iloc[-1] \
                            if not df_processed[df_processed['doctor_id'] == doctor_id].empty else df_processed['delay'].mean()
    
    # Handle missing values
    historical_delay = 0 if pd.isna(historical_delay) else historical_delay
    peak_hour_delay = 0 if pd.isna(peak_hour_delay) else peak_hour_delay
    seasonal_pattern = 0 if pd.isna(seasonal_pattern) else seasonal_pattern
    day_of_week_pattern = 0 if pd.isna(day_of_week_pattern) else day_of_week_pattern
    weekend_pattern = 0 if pd.isna(weekend_pattern) else weekend_pattern
    recent_delay_trend = 0 if pd.isna(recent_delay_trend) else recent_delay_trend
    
    # Create feature vector
    features_dict = {
        'doctor_id': doctor_id,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'day': day,
        'is_peak': is_peak,
        'is_weekend': is_weekend,
        'doc_mean_delay': doc_mean_delay,
        'doc_delay_std': doc_delay_std,
        'doc_median_delay': doc_median_delay,
        'doc_appointment_count': doc_appointment_count,
        'historical_delay': historical_delay,
        'peak_hour_delay': peak_hour_delay,
        'seasonal_pattern': seasonal_pattern,
        'day_of_week_pattern': day_of_week_pattern,
        'weekend_pattern': weekend_pattern,
        'recent_delay_trend': recent_delay_trend
    }
    
    # Create DataFrame with features
    X_pred = pd.DataFrame([features_dict])
    
    # Make predictions with available models
    predictions = []
    
    # Random Forest prediction
    rf_pred = rf_model.predict(X_pred)[0]
    predictions.append((rf_pred, 0.5))  # 50% weight
    
    # Gradient Boosting prediction if available
    if gb_model is not None:
        gb_pred = gb_model.predict(X_pred)[0]
        predictions.append((gb_pred, 0.3))  # 30% weight
    
    # Neural Network prediction if available
    if nn_model is not None and scaler is not None:
        X_pred_scaled = scaler.transform(X_pred)
        nn_pred = nn_model.predict(X_pred_scaled)[0][0]
        predictions.append((nn_pred, 0.2))  # 20% weight
    
    # Calculate weighted average
    if len(predictions) > 1:
        weighted_sum = sum(pred * weight for pred, weight in predictions)
        total_weight = sum(weight for _, weight in predictions)
        final_prediction = weighted_sum / total_weight
    else:
        final_prediction = predictions[0][0]
    
    # Ensure prediction is positive
    return max(0, final_prediction)

# Enhanced real-time monitoring dashboard
class EnhancedClinicMonitor:
    def __init__(self):
        self.current_metrics = {
            'active_patients': 0,
            'avg_wait_time': 0,
            'peak_hour_load': 0,
            'doctor_utilization': {},
            'wait_time_by_hour': {},
            'wait_time_by_doctor': {},
            'patient_satisfaction': 0.0
        }
        self.historical_metrics = []
        self.alerts = []
        self.satisfaction_scores = []  # Track patient satisfaction scores
        
    def update_metrics(self, df):
        """Update real-time clinic metrics with enhanced analytics"""
        current_time = datetime.now()
        
        # Calculate active patients and wait times
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
        df['actual_time'] = pd.to_datetime(df['actual_time'])
        
        active_mask = (df['scheduled_time'] <= current_time) & (df['actual_time'] >= current_time)
        self.current_metrics['active_patients'] = active_mask.sum()
        
        if active_mask.any():
            wait_times = (df.loc[active_mask, 'actual_time'] - df.loc[active_mask, 'scheduled_time'])
            self.current_metrics['avg_wait_time'] = wait_times.dt.total_seconds().mean() / 60
        
        # Calculate peak hour metrics
        df['hour'] = df['scheduled_time'].dt.hour
        peak_mask = df['hour'].between(17, 20)
        self.current_metrics['peak_hour_load'] = peak_mask.sum() / len(df) if len(df) > 0 else 0
        
        # Calculate doctor utilization
        for doctor_id in df['doctor_id'].unique():
            doc_mask = (df['doctor_id'] == doctor_id) & active_mask
            if doc_mask.any():
                self.current_metrics['doctor_utilization'][doctor_id] = {
                    'current_patients': doc_mask.sum(),
                    'avg_delay': df.loc[doc_mask, 'delay'].mean() if 'delay' in df else 0,
                    'utilization_rate': doc_mask.sum() / active_mask.sum() if active_mask.sum() > 0 else 0
                }
        
        # Calculate wait time by hour
        for hour in range(9, 21):  # Clinic hours 9am-8pm
            hour_mask = (df['hour'] == hour)
            if hour_mask.any() and 'delay' in df:
                self.current_metrics['wait_time_by_hour'][hour] = df.loc[hour_mask, 'delay'].mean()
            else:
                self.current_metrics['wait_time_by_hour'][hour] = 0
        
        # Calculate wait time by doctor
        for doctor_id in df['doctor_id'].unique():
            doc_mask = (df['doctor_id'] == doctor_id)
            if doc_mask.any() and 'delay' in df:
                self.current_metrics['wait_time_by_doctor'][doctor_id] = df.loc[doc_mask, 'delay'].mean()
            else:
                self.current_metrics['wait_time_by_doctor'][doctor_id] = 0
        
        # Simulate patient satisfaction based on wait times
        if 'delay' in df:
            avg_delay = df['delay'].mean()
            # Satisfaction decreases as delay increases
            satisfaction = max(0, min(100, 100 - (avg_delay * 2)))
            self.current_metrics['patient_satisfaction'] = satisfaction
            self.satisfaction_scores.append(satisfaction)
        
        # Store historical data
        self.historical_metrics.append({
            'timestamp': current_time,
            'metrics': self.current_metrics.copy()
        })
        
        # Generate alerts
        self._generate_alerts()
        
        # Emit updated metrics via Socket.IO
        socketio.emit('metrics_update', self.get_current_status())
    
    def _generate_alerts(self):
        """Generate alerts based on current metrics"""
        self.alerts = []
        
        if self.current_metrics['avg_wait_time'] > 30:
            self.alerts.append({
                'level': 'high',
                'message': f'High wait time alert: Average wait time exceeds 30 minutes ({self.current_metrics["avg_wait_time"]:.1f} min)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        elif self.current_metrics['avg_wait_time'] > 20:
            self.alerts.append({
                'level': 'medium',
                'message': f'Medium wait time alert: Average wait time exceeds 20 minutes ({self.current_metrics["avg_wait_time"]:.1f} min)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if self.current_metrics['active_patients'] > 50:
            self.alerts.append({
                'level': 'high',
                'message': f'High patient load alert: {self.current_metrics["active_patients"]} active patients',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        elif self.current_metrics['active_patients'] > 30:
            self.alerts.append({
                'level': 'medium',
                'message': f'Medium patient load alert: {self.current_metrics["active_patients"]} active patients',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Doctor-specific alerts
        for doctor_id, stats in self.current_metrics['doctor_utilization'].items():
            if stats.get('avg_delay', 0) > 40:
                self.alerts.append({
                    'level': 'high',
                    'message': f'Doctor {doctor_id} has high average delay: {stats["avg_delay"]:.1f} minutes',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            if stats.get('current_patients', 0) > 10:
                self.alerts.append({
                    'level': 'medium',
                    'message': f'Doctor {doctor_id} has high patient load: {stats["current_patients"]} patients',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    
    def get_current_status(self):
        """Get current clinic status and alerts"""
        return {
            'current_metrics': self.current_metrics,
            'alerts': self.alerts,
            'historical_data': self._get_historical_trends()
        }
    
    def _get_historical_trends(self):
        """Calculate historical trends from stored metrics"""
        if len(self.historical_metrics) < 2:
            return {}
        
        # Get last 24 data points (or all if less than 24)
        recent_metrics = self.historical_metrics[-24:]
        
        # Extract wait times and patient counts
        timestamps = [m['timestamp'].strftime('%H:%M') for m in recent_metrics]
        wait_times = [m['metrics']['avg_wait_time'] for m in recent_metrics]
        patient_counts = [m['metrics']['active_patients'] for m in recent_metrics]
        satisfaction = self.satisfaction_scores[-24:] if len(self.satisfaction_scores) > 0 else [0]
        
        return {
            'timestamps': timestamps,
            'wait_times': wait_times,
            'patient_counts': patient_counts,
            'satisfaction': satisfaction,
            'wait_time_trend': 'increasing' if wait_times[-1] > wait_times[0] else 'decreasing',
            'patient_count_trend': 'increasing' if patient_counts[-1] > patient_counts[0] else 'decreasing',
            'satisfaction_trend': 'increasing' if satisfaction[-1] > satisfaction[0] else 'decreasing'
        }

# Initialize enhanced clinic monitor
clinic_monitor = EnhancedClinicMonitor()

# Enhanced dynamic scheduling function
def optimize_schedule_enhanced(df):
    """Dynamically adjust schedule based on predicted delays with advanced optimization"""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert to datetime
    df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df['hour'] = df['scheduled_time'].dt.hour
    df['month'] = df['scheduled_time'].dt.month
    df['day_of_week'] = df['scheduled_time'].dt.dayofweek
    df['day'] = df['scheduled_time'].dt.day
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Calculate delay if it doesn't exist
    if 'delay' not in df.columns:
        df['delay'] = (df['actual_time'] - df['scheduled_time']).dt.total_seconds() / 60
    
    # Calculate predicted delays with enhanced model
    df['predicted_delay'] = [
        predict_wait_time_enhanced(row['doctor_id'], row['scheduled_time'])
        for _, row in df.iterrows()
    ]
    
    # Identify peak hours (17:00-20:00) if not already defined
    if 'is_peak' not in df.columns:
        df['is_peak'] = df['hour'].between(17, 20)
    
    # Calculate historical patterns for real-time monitoring
    df['historical_delay'] = df.groupby(['doctor_id', 'hour'])['delay'].transform('mean')
    df['peak_hour_delay'] = df.groupby(['doctor_id', 'is_peak'])['delay'].transform('mean')
    df['seasonal_pattern'] = df.groupby(['doctor_id', 'month'])['delay'].transform('mean')
    df['day_of_week_pattern'] = df.groupby(['doctor_id', 'day_of_week'])['delay'].transform('mean')
    df['weekend_pattern'] = df.groupby(['doctor_id', 'is_weekend'])['delay'].transform('mean')
    
    # Calculate patient arrival deviations if not exists
    if 'arrival_deviation' not in df.columns:

        # Simulate early arrivals (negative values) and late arrivals (positive values)
        df['arrival_deviation'] = np.random.normal(-10),