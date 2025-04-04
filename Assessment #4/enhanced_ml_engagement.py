import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import joblib
import pandas as pd

class EnhancedMLEngagement:
    """Enhanced Machine Learning based patient engagement prediction system using deep learning"""
    
    def __init__(self):
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
            'day_of_week',
            'historical_engagement',
            'demographic_score',
            'channel_success_rate'
        ]
        
        # Initialize deep learning model
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and compile an advanced deep learning model with transformer architecture"""
        # Input layer
        inputs = layers.Input(shape=(len(self.features),))
        
        # Embedding layer
        x = layers.Dense(256)(inputs)
        x = layers.Reshape((16, 16))(x)  # Fixed reshape dimensions
        
        # Transformer blocks
        for _ in range(3):
            # Multi-head attention
            attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn = layers.Dense(64, activation='relu')(x)
            ffn = layers.Dense(16)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final classification layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-task outputs
        engagement_prob = layers.Dense(1, activation='sigmoid', name='engagement')(x)
        channel_pred = layers.Dense(3, activation='softmax', name='channel')(x)
        time_pred = layers.Dense(24, activation='softmax', name='time')(x)
        
        # Create model with multiple outputs
        model = models.Model(
            inputs=inputs,
            outputs=[engagement_prob, channel_pred, time_pred]
        )
        
        # Compile with custom loss weights and advanced metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'engagement': 'binary_crossentropy',
                'channel': 'categorical_crossentropy',
                'time': 'categorical_crossentropy'
            },
            loss_weights={
                'engagement': 1.0,
                'channel': 0.3,
                'time': 0.3
            },
            metrics={
                'engagement': [
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()
                ],
                'channel': ['accuracy'],
                'time': ['accuracy']
            }
        )
        
        return model

    # Rest of the code remains unchanged...
    # (Including all other methods with their original implementation)
