import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedWaitTimePredictor:
    """Advanced ML models for wait time prediction with XGBoost and Deep Learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        
    def preprocess_data(self, df):
        """Enhanced preprocessing with more sophisticated feature engineering"""
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['scheduled_time']):
            df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
        if not pd.api.types.is_datetime64_any_dtype(df['actual_time']):
            df['actual_time'] = pd.to_datetime(df['actual_time'])
        
        # Calculate delay
        df['delay'] = (df['actual_time'] - df['scheduled_time']).dt.total_seconds() / 60
        
        # Extract time features
        df['hour'] = df['scheduled_time'].dt.hour
        df['minute'] = df['scheduled_time'].dt.minute
        df['day_of_week'] = df['scheduled_time'].dt.dayofweek
        df['month'] = df['scheduled_time'].dt.month
        df['day'] = df['scheduled_time'].dt.day
        df['is_peak'] = df['hour'].between(17, 20).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['time_category'] = pd.cut(
            df['hour'], 
            bins=[0, 9, 12, 15, 18, 21, 24], 
            labels=['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night']
        )
        
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
        
        # Calculate patient load features
        df['appointments_per_hour'] = df.groupby(['scheduled_time.dt.date', 'hour'])['patient_id'].transform('count')
        df['doctor_appointments_per_day'] = df.groupby(['doctor_id', 'scheduled_time.dt.date'])['patient_id'].transform('count')
        
        # Calculate rolling averages for recent trends
        df = df.sort_values('scheduled_time')
        df['recent_delay_trend'] = df.groupby('doctor_id')['delay'].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['recent_delay_std'] = df.groupby('doctor_id')['delay'].transform(lambda x: x.rolling(10, min_periods=1).std())
        
        # Calculate time since last appointment for doctor
        df['prev_appointment'] = df.groupby('doctor_id')['scheduled_time'].shift(1)
        df['time_since_last_appointment'] = (df['scheduled_time'] - df['prev_appointment']).dt.total_seconds() / 60
        df['time_since_last_appointment'] = df['time_since_last_appointment'].fillna(df['time_since_last_appointment'].median())
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64] and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def train_xgboost_model(self, X_train, y_train, X_test, y_test):
        """Train an XGBoost model with hyperparameter tuning"""
        # Define parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1]
        }
        
        # Create XGBoost regressor
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_xgb = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_xgb.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and metrics
        self.models['xgboost'] = best_xgb
        self.model_metrics['xgboost'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        # Get feature importance
        feature_importance = best_xgb.feature_importances_
        self.feature_importance['xgboost'] = feature_importance
        
        return best_xgb, mae, feature_importance
    
    def train_deep_learning_model(self, X_train, y_train, X_test, y_test):
        """Train a deep learning model with advanced architecture"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['deep_learning'] = scaler
        
        # Define model architecture
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test_scaled).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and metrics
        self.models['deep_learning'] = model
        self.model_metrics['deep_learning'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'history': history.history
        }
        
        return model, mae
    
    def train_models(self, df, features, target):
        """Train multiple models and select the best one"""
        # Preprocess data if needed
        if 'delay' not in df.columns:
            df = self.preprocess_data(df)
        
        # Prepare data
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        xgb_model, xgb_mae, xgb_feature_importance = self.train_xgboost_model(X_train, y_train, X_test, y_test)
        
        # Train deep learning model
        dl_model, dl_mae = self.train_deep_learning_model(X_train, y_train, X_test, y_test)
        
        # Select best model
        if xgb_mae <= dl_mae:
            self.best_model = xgb_model
            self.best_model_name = 'xgboost'
        else:
            self.best_model = dl_model
            self.best_model_name = 'deep_learning'
        
        print(f"Best model: {self.best_model_name} with MAE: {min(xgb_mae, dl_mae):.2f} minutes")
        
        return self.best_model
    
    def predict(self, features):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Convert to DataFrame if needed
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame([features])
        
        # Scale features if using deep learning model
        if self.best_model_name == 'deep_learning':
            features_scaled = self.scalers['deep_learning'].transform(features)
            predictions = self.best_model.predict(features_scaled).flatten()
        else:
            predictions = self.best_model.predict(features)
        
        return predictions
    
    def save_models(self, path='./models'):
        """Save trained models to disk"""
        import os
        
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save XGBoost model if available
        if 'xgboost' in self.models:
            self.models['xgboost'].save_model(f"{path}/xgboost_model.json")
        
        # Save deep learning model if available
        if 'deep_learning' in self.models:
            self.models['deep_learning'].save(f"{path}/deep_learning_model")
            
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{path}/{name}_scaler.pkl")
        
        # Save metrics and feature importance
        joblib.dump(self.model_metrics, f"{path}/model_metrics.pkl")
        joblib.dump(self.feature_importance, f"{path}/feature_importance.pkl")
        
        print(f"Models saved to {path}")
    
    def load_models(self, path='./models'):
        """Load trained models from disk"""
        import os
        
        # Check if directory exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        
        # Load XGBoost model if available
        xgb_path = f"{path}/xgboost_model.json"
        if os.path.exists(xgb_path):
            self.models['xgboost'] = xgb.XGBRegressor()
            self.models['xgboost'].load_model(xgb_path)
        
        # Load deep learning model if available
        dl_path = f"{path}/deep_learning_model"
        if os.path.exists(dl_path):
            self.models['deep_learning'] = keras.models.load_model(dl_path)
        
        # Load scalers
        for name in ['deep_learning']:
            scaler_path = f"{path}/{name}_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scalers[name] = joblib.load(scaler_path)
        
        # Load metrics and feature importance
        metrics_path = f"{path}/model_metrics.pkl"
        if os.path.exists(metrics_path):
            self.model_metrics = joblib.load(metrics_path)
        
        importance_path = f"{path}/feature_importance.pkl"
        if os.path.exists(importance_path):
            self.feature_importance = joblib.load(importance_path)
        
        # Set best model
        if self.model_metrics:
            # Find model with lowest MAE
            best_model_name = min(self.model_metrics, key=lambda x: self.model_metrics[x]['mae'])
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
        
        print(f"Models loaded from {path}")
        
    def plot_feature_importance(self, feature_names=None, top_n=10):
        """Plot feature importance for tree-based models"""
        if 'xgboost' not in self.feature_importance:
            raise ValueError("No feature importance available. Train XGBoost model first.")
        
        # Get feature importance
        importance = self.feature_importance['xgboost']
        
        # Use provided feature names or default
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        return importance_df

# Function to predict wait time using advanced models
def predict_wait_time_advanced(doctor_id, scheduled_time, model_path='./models'):
    """Predict wait time using advanced ML models"""
    # Create predictor
    predictor = AdvancedWaitTimePredictor()
    
    # Load models if available
    try:
        predictor.load_models(model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Training new models...")
        
        # Load data
        from enhanced_clinic_system import load_data, preprocess_data
        df = load_data()
        df_processed = preprocess_data(df)
        
        # Define features
        features = [
            'doctor_id', 'hour', 'day_of_week', 'month', 'day', 'is_peak', 'is_weekend',
            'doc_mean_delay', 'doc_delay_std', 'doc_median_delay', 'doc_appointment_count',
            'historical_delay', 'peak_hour_delay', 'seasonal_pattern', 'day_of_week_pattern',
            'weekend_pattern', 'recent_delay_trend'
        ]
        
        # Train models
        predictor.train_models(df_processed, features, 'delay')
        
        # Save models
        try:
            predictor.save_models(model_path)
        except Exception as e:
            print(f"Error saving models: {e}")
    
    # Prepare features for prediction
    if isinstance(scheduled_time, str):
        scheduled_time = pd.to_datetime(scheduled_time)
    
    # Extract features
    features = {
        'doctor_id': doctor_id,
        'hour': scheduled_time.hour,
        'day_of_week': scheduled_time.dayofweek,
        'month': scheduled_time.month,
        'day': scheduled_time.day,
        'is_peak': 1 if scheduled_time.hour >= 17 and scheduled_time.hour <= 20 else 0,
        'is_weekend': 1 if scheduled_time.dayofweek >= 5 else 0
    }
    
    # Add dummy values for other features (will be ignored by the model if not needed)
    for feature in ['doc_mean_delay', 'doc_delay_std', 'doc_median_delay', 'doc_appointment_count',
                   'historical_delay', 'peak_hour_delay', 'seasonal_pattern', 'day_of_week_pattern',
                   'weekend_pattern', 'recent_delay_trend']:
        features[feature] = 0
    
    # Make prediction
    try:
        wait_time = predictor.predict(features)[0]
        return max(0, wait_time)  # Ensure non-negative wait time
    except Exception as e:
        print(f"Error making prediction: {e}")
        # Fallback to simpler prediction
        from enhanced_clinic_system import predict_wait_time_enhanced
        return predict_wait_time_enhanced(doctor_id, scheduled_time)