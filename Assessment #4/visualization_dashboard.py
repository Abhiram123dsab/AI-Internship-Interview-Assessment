import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class VisualizationDashboard:
    """Real-time visualization dashboard for patient engagement metrics"""
    
    def __init__(self):
        self.figures = {}
        self.update_interval = 300  # 5 minutes in seconds
        self.last_update = datetime.now()
    
    def create_engagement_overview(self, metrics_data):
        """Create overview of engagement metrics across channels"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Channel Performance',
                'Language-wise Engagement',
                'Age Group Response Rate',
                'Hourly Activity'
            )
        )
        
        # Channel Performance
        channels = ['SMS', 'WhatsApp', 'IVR']
        delivery_rates = [metrics_data['delivery_rate'][ch] for ch in channels]
        response_rates = [metrics_data['response_rate'][ch] for ch in channels]
        
        fig.add_trace(
            go.Bar(
                name='Delivery Rate',
                x=channels,
                y=delivery_rates,
                marker_color='rgb(55, 83, 109)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Response Rate',
                x=channels,
                y=response_rates,
                marker_color='rgb(26, 118, 255)'
            ),
            row=1, col=1
        )
        
        # Language-wise Engagement
        languages = ['Tamil', 'Telugu', 'Malayalam', 'Hindi', 'English']
        engagement_scores = [metrics_data['language_engagement'][lang] for lang in languages]
        
        fig.add_trace(
            go.Pie(
                labels=languages,
                values=engagement_scores,
                hole=.3
            ),
            row=1, col=2
        )
        
        # Age Group Response Rate
        age_groups = ['<30', '30-50', '50-65', '>65']
        response_rates = [metrics_data['age_response_rate'][age] for age in age_groups]
        
        fig.add_trace(
            go.Bar(
                x=age_groups,
                y=response_rates,
                marker_color='rgb(158, 202, 225)'
            ),
            row=2, col=1
        )
        
        # Hourly Activity
        hours = list(range(24))
        activity = [metrics_data['hourly_activity'][hour] for hour in hours]
        
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=activity,
                mode='lines+markers',
                line=dict(color='rgb(128, 128, 128)', width=2)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Patient Engagement Analytics Dashboard',
            title_x=0.5
        )
        
        self.figures['overview'] = fig
        return fig
    
    def create_ml_predictions_view(self, prediction_data):
        """Create visualization for ML-based predictions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Engagement Likelihood Distribution',
                'Channel Recommendations',
                'Best Contact Times',
                'Prediction Confidence'
            )
        )
        
        # Engagement Likelihood Distribution
        likelihood_scores = prediction_data['engagement_likelihood']
        fig.add_trace(
            go.Histogram(
                x=likelihood_scores,
                nbinsx=20,
                marker_color='rgb(55, 83, 109)'
            ),
            row=1, col=1
        )
        
        # Channel Recommendations
        channels = ['SMS', 'WhatsApp', 'IVR']
        recommendations = [prediction_data['channel_recommendations'][ch] for ch in channels]
        
        fig.add_trace(
            go.Bar(
                x=channels,
                y=recommendations,
                marker_color='rgb(26, 118, 255)'
            ),
            row=1, col=2
        )
        
        # Best Contact Times
        hours = list(range(24))
        contact_times = [prediction_data['best_times'][hour] for hour in hours]
        
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=contact_times,
                mode='lines+markers',
                line=dict(color='rgb(128, 128, 128)', width=2)
            ),
            row=2, col=1
        )
        
        # Prediction Confidence
        confidence_scores = prediction_data['prediction_confidence']
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=np.mean(confidence_scores),
                gauge={'axis': {'range': [0, 1]}},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text='ML Prediction Insights',
            title_x=0.5
        )
        
        self.figures['ml_predictions'] = fig
        return fig
    
    def update_dashboard(self, metrics_data, prediction_data):
        """Update all dashboard visualizations"""
        if (datetime.now() - self.last_update).total_seconds() >= self.update_interval:
            self.create_engagement_overview(metrics_data)
            self.create_ml_predictions_view(prediction_data)
            self.last_update = datetime.now()
        
        return {
            'overview': self.figures.get('overview'),
            'ml_predictions': self.figures.get('ml_predictions')
        }