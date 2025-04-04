from ml_patient_engagement import MLPatientEngagement
from visualization_dashboard import VisualizationDashboard
from advanced_effectiveness_measurement import AdvancedEffectivenessMeasurement
from datetime import datetime, timedelta
import threading
import time

class EnhancedAnalyticsIntegration:
    """Integration module for ML-based analytics and visualization"""
    
    def __init__(self):
        self.ml_engine = MLPatientEngagement()
        self.dashboard = VisualizationDashboard()
        self.metrics_system = AdvancedEffectivenessMeasurement()
        
        # Cache for analytics data
        self.analytics_cache = {
            'metrics_data': None,
            'prediction_data': None,
            'last_update': datetime.now(),
            'cache_duration': 300  # 5 minutes
        }
        
        # Start background analytics update thread
        self.update_thread = threading.Thread(target=self._update_analytics_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _update_analytics_loop(self):
        """Background thread to update analytics data"""
        while True:
            try:
                self._update_analytics_data()
                time.sleep(60)  # Update every minute
            except Exception as e:
                print(f'Error updating analytics: {str(e)}')
                time.sleep(300)  # Wait 5 minutes on error
    
    def _update_analytics_data(self):
        """Update cached analytics data"""
        now = datetime.now()
        
        # Check if cache needs update
        if (self.analytics_cache['metrics_data'] is None or
            (now - self.analytics_cache['last_update']).total_seconds() >= self.analytics_cache['cache_duration']):
            
            # Get current metrics
            metrics_data = {
                'delivery_rate': {
                    'SMS': self.metrics_system.get_delivery_rate('SMS'),
                    'WhatsApp': self.metrics_system.get_delivery_rate('WhatsApp'),
                    'IVR': self.metrics_system.get_delivery_rate('IVR')
                },
                'response_rate': {
                    'SMS': self.metrics_system.get_response_rate('SMS'),
                    'WhatsApp': self.metrics_system.get_response_rate('WhatsApp'),
                    'IVR': self.metrics_system.get_response_rate('IVR')
                },
                'language_engagement': self.metrics_system.get_language_engagement_metrics(),
                'age_response_rate': self.metrics_system.get_age_group_metrics(),
                'hourly_activity': self.metrics_system.get_hourly_activity_metrics()
            }
            
            # Get ML predictions
            prediction_data = {
                'engagement_likelihood': [],
                'channel_recommendations': {
                    'SMS': 0,
                    'WhatsApp': 0,
                    'IVR': 0
                },
                'best_times': [0] * 24,
                'prediction_confidence': []
            }
            
            # Process each patient's data
            for patient in self.metrics_system.get_active_patients():
                prediction = self.ml_engine.predict_engagement(patient)
                
                # Aggregate predictions
                prediction_data['engagement_likelihood'].append(prediction['engagement_likelihood'])
                prediction_data['channel_recommendations'][prediction['recommended_channel']] += 1
                prediction_data['best_times'][prediction['best_time']['hour']] += 1
                prediction_data['prediction_confidence'].append(
                    max(prediction['engagement_likelihood'], 1 - prediction['engagement_likelihood'])
                )
            
            # Update cache
            self.analytics_cache.update({
                'metrics_data': metrics_data,
                'prediction_data': prediction_data,
                'last_update': now
            })
    
    def get_dashboard_data(self):
        """Get current dashboard visualizations"""
        self._update_analytics_data()  # Ensure data is current
        
        return self.dashboard.update_dashboard(
            self.analytics_cache['metrics_data'],
            self.analytics_cache['prediction_data']
        )
    
    def get_patient_insights(self, patient_id):
        """Get ML-based insights for a specific patient"""
        patient_data = self.metrics_system.get_patient_data(patient_id)
        if not patient_data:
            return None
        
        prediction = self.ml_engine.predict_engagement(patient_data)
        
        return {
            'engagement_prediction': prediction,
            'historical_metrics': {
                'response_rate': patient_data.get('response_rate', 0),
                'avg_response_time': patient_data.get('avg_response_time', 0),
                'preferred_channel': patient_data.get('preferred_channel', 'SMS'),
                'appointment_adherence': patient_data.get('appointment_adherence', 0)
            }
        }
    
    def optimize_communication_strategy(self):
        """Generate optimized communication strategies based on ML insights"""
        strategies = []
        
        # Analyze each demographic segment
        for language in ['Tamil', 'Telugu', 'Malayalam', 'Hindi', 'English']:
            language_data = self.metrics_system.get_language_specific_metrics(language)
            
            # Get ML predictions for this segment
            predictions = self.ml_engine.predict_engagement({
                'language': language,
                **language_data
            })
            
            strategies.append({
                'language': language,
                'recommended_channel': predictions['recommended_channel'],
                'best_contact_time': predictions['best_time'],
                'expected_engagement': predictions['engagement_likelihood']
            })
        
        return strategies