import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from grafana_api.grafana_face import GrafanaFace
import threading
import time

class EnhancedAnalyticsDashboard:
    """Enhanced real-time analytics dashboard with Prometheus integration"""
    
    def __init__(self, prometheus_port=8000):
        self.figures = {}
        self.update_interval = 60  # 1 minute in seconds
        self.last_update = datetime.now()
        
        # Initialize Prometheus metrics
        self.setup_prometheus_metrics(prometheus_port)
        
        # Initialize real-time metrics storage
        self.metrics_store = {
            'engagement_scores': [],
            'channel_performance': {},
            'response_times': [],
            'prediction_accuracy': []
        }
    
    def setup_prometheus_metrics(self, port):
        """Setup Prometheus metrics and start server"""
        # Start Prometheus server
        start_http_server(port)
        
        # Define metrics
        self.metrics = {
            'engagement_score': Gauge('patient_engagement_score', 
                                    'Current patient engagement score'),
            'response_time': Histogram('patient_response_time_seconds', 
                                     'Patient response time in seconds',
                                     buckets=[60, 300, 900, 1800, 3600]),
            'channel_success': Counter('channel_success_total', 
                                     'Successful message deliveries by channel',
                                     ['channel']),
            'prediction_accuracy': Gauge('ml_prediction_accuracy', 
                                       'ML model prediction accuracy'),
            'active_patients': Gauge('active_patients_total', 
                                    'Number of currently active patients')
        }
    
    def create_real_time_dashboard(self, metrics_data, prediction_data):
        """Create enhanced real-time dashboard with ML insights"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Real-time Engagement Metrics',
                'Channel Performance Trends',
                'Response Time Distribution',
                'ML Prediction Accuracy',
                'Demographic Insights',
                'Engagement Patterns'
            ),
            specs=[[{'type': 'indicator'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'heatmap'}, {'type': 'scatter3d'}]]
        )
        
        # Real-time Engagement Gauge
        fig.add_trace(
            go.Indicator(
                mode='gauge+number+delta',
                value=metrics_data['current_engagement_score'],
                delta={'reference': metrics_data['previous_engagement_score']},
                gauge={'axis': {'range': [0, 100]},
                       'steps': [
                           {'range': [0, 30], 'color': 'red'},
                           {'range': [30, 70], 'color': 'yellow'},
                           {'range': [70, 100], 'color': 'green'}
                       ]},
                title={'text': 'Overall Engagement Score'}
            ),
            row=1, col=1
        )
        
        # Channel Performance Trends
        for channel in ['SMS', 'WhatsApp', 'IVR']:
            fig.add_trace(
                go.Scatter(
                    x=metrics_data['timestamps'],
                    y=metrics_data['channel_performance'][channel],
                    name=f'{channel} Performance',
                    mode='lines+markers'
                ),
                row=1, col=2
            )
        
        # Response Time Distribution by Channel
        channels = ['SMS', 'WhatsApp', 'IVR']
        response_rates = [metrics_data['response_rate'][ch] for ch in channels]
        delivery_rates = [metrics_data['delivery_rate'][ch] for ch in channels]
        
        fig.add_trace(
            go.Bar(
                x=channels,
                y=response_rates,
                name='Response Rate',
                marker_color='rgb(55, 83, 109)'
            ),
            row=2, col=1
        )
        
        # ML Prediction Accuracy Distribution
        fig.add_trace(
            go.Histogram(
                x=prediction_data['accuracy_scores'],
                nbinsx=20,
                name='Prediction Accuracy',
                marker_color='rgb(26, 118, 255)'
            ),
            row=2, col=2
        )
        
        # Demographic Engagement Heatmap
        age_groups = ['<30', '30-50', '50-65', '>65']
        languages = ['Tamil', 'Telugu', 'Malayalam', 'Hindi', 'English']
        
        fig.add_trace(
            go.Heatmap(
                z=metrics_data['demographic_engagement'],
                x=age_groups,
                y=languages,
                colorscale='Viridis'
            ),
            row=3, col=1
        )
        
        # 3D Engagement Patterns
        fig.add_trace(
            go.Scatter3d(
                x=metrics_data['time_data'],
                y=metrics_data['response_data'],
                z=metrics_data['engagement_data'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=metrics_data['engagement_data'],
                    colorscale='Viridis',
                    opacity=0.8
                )
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text='Enhanced Patient Engagement Analytics Dashboard',
            title_x=0.5
        )
        
        self.figures['real_time'] = fig
        return fig
    
    def create_ml_insights_dashboard(self, prediction_data):
        """Create dashboard for ML model insights"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Feature Importance Analysis',
                'Prediction Confidence Distribution',
                'Channel Recommendation Analysis',
                'Temporal Pattern Analysis'
            )
        )
        
        # Feature Importance
        features = list(prediction_data['feature_importance'].keys())
        importance_scores = list(prediction_data['feature_importance'].values())
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=importance_scores,
                marker_color='rgb(55, 83, 109)',
                name='Feature Importance'
            ),
            row=1, col=1
        )
        
        # Prediction Confidence
        fig.add_trace(
            go.Histogram(
                x=prediction_data['confidence_scores'],
                nbinsx=20,
                marker_color='rgb(26, 118, 255)',
                name='Prediction Confidence'
            ),
            row=1, col=2
        )
        
        # Channel Recommendations
        channels = ['SMS', 'WhatsApp', 'IVR']
        recommendations = [prediction_data['channel_recommendations'][ch] 
                          for ch in channels]
        
        fig.add_trace(
            go.Bar(
                x=channels,
                y=recommendations,
                marker_color='rgb(158, 202, 225)',
                name='Channel Recommendations'
            ),
            row=2, col=1
        )
        
        # Temporal Patterns
        hours = list(range(24))
        activity = [prediction_data['hourly_patterns'][hour] 
                   for hour in hours]
        
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=activity,
                mode='lines+markers',
                line=dict(color='rgb(128, 128, 128)', width=2),
                name='Temporal Patterns'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='ML Model Insights Dashboard',
            title_x=0.5
        )
        
        self.figures['ml_insights'] = fig
        return fig
    
    def update_prometheus_metrics(self, metrics_data):
        """Update Prometheus metrics with latest data"""
        # Update engagement score
        self.metrics['engagement_score'].set(
            metrics_data['current_engagement_score']
        )
        
        # Update response times
        for response_time in metrics_data['response_times']:
            self.metrics['response_time'].observe(response_time)
        
        # Update channel success metrics
        for channel in ['SMS', 'WhatsApp', 'IVR']:
            self.metrics['channel_success'].labels(channel=channel).inc(
                metrics_data['channel_success'][channel]
            )
        
        # Update ML metrics
        self.metrics['prediction_accuracy'].set(
            metrics_data['ml_accuracy']
        )
        
        # Update active patients
        self.metrics['active_patients'].set(
            metrics_data['active_patients']
        )
    
    def start_metrics_collection(self):
        """Start background metrics collection"""
        def collect_metrics():
            while True:
                # Collect metrics here
                time.sleep(self.update_interval)
        
        metrics_thread = threading.Thread(target=collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
    
    def update_dashboard(self, metrics_data, prediction_data):
        """Update all dashboard visualizations"""
        if (datetime.now() - self.last_update).total_seconds() >= self.update_interval:
            # Update Prometheus metrics
            self.update_prometheus_metrics(metrics_data)
            
            # Update dashboards
            self.create_real_time_dashboard(metrics_data, prediction_data)
            self.create_ml_insights_dashboard(prediction_data)
            
            self.last_update = datetime.now()
        
        return {
            'real_time': self.figures.get('real_time'),
            'ml_insights': self.figures.get('ml_insights')
        }