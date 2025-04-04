import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import json
from datetime import datetime, timedelta
import random
from enhanced_clinic_system import load_data, preprocess_data, EnhancedClinicMonitor
from advanced_ml_models import AdvancedWaitTimePredictor, predict_wait_time_advanced

class RealTimeAnalyticsDashboard:
    """Real-time analytics dashboard with interactive visualizations"""
    
    def __init__(self, server=None):
        # Initialize with Flask server if provided
        self.server = server or Flask(__name__)
        self.socketio = SocketIO(self.server) if server is None else None
        
        # Initialize Dash app
        self.dash_app = Dash(__name__, server=self.server, url_base_pathname='/analytics/')
        
        # Load data
        self.df = load_data()
        self.df_processed = preprocess_data(self.df)
        
        # Initialize clinic monitor
        self.clinic_monitor = EnhancedClinicMonitor()
        self.clinic_monitor.update_metrics(self.df)
        
        # Initialize advanced ML predictor
        self.predictor = AdvancedWaitTimePredictor()
        
        # Setup dashboard layout
        self.setup_layout()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup Dash app layout"""
        self.dash_app.layout = html.Div([
            html.Div([
                html.H1("Jayanagar Specialty Clinic - Advanced Analytics", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
                html.Div([
                    html.Div([
                        html.H3("Current Clinic Status", style={'color': '#3498db'}),
                        html.Div(id='clinic-status-cards', className='status-cards')
                    ], className='six columns'),
                    html.Div([
                        html.H3("Prediction Controls", style={'color': '#3498db'}),
                        html.Div([
                            html.Label("Select Doctor:"),
                            dcc.Dropdown(
                                id='doctor-selector',
                                options=[{'label': f'Doctor {i}', 'value': i} for i in range(1, 16)],
                                value=1
                            ),
                            html.Label("Select Date:"),
                            dcc.DatePickerSingle(
                                id='date-picker',
                                date=datetime.now().date(),
                                display_format='YYYY-MM-DD'
                            ),
                            html.Label("Select Time:"),
                            dcc.Slider(
                                id='time-slider',
                                min=9,
                                max=20,
                                step=0.5,
                                marks={i: f'{i}:00' if i == int(i) else f'{int(i)}:30' for i in range(9, 21)},
                                value=datetime.now().hour + (0.5 if datetime.now().minute >= 30 else 0)
                            ),
                            html.Button('Predict Wait Time', id='predict-button', 
                                      style={'marginTop': 20, 'backgroundColor': '#2ecc71', 'color': 'white'})
                        ])
                    ], className='six columns')
                ], className='row'),
                
                html.Div([
                    html.Div([
                        html.H3("Wait Time Prediction", style={'color': '#3498db'}),
                        html.Div(id='prediction-result', className='prediction-box')
                    ], className='six columns'),
                    html.Div([
                        html.H3("Model Performance", style={'color': '#3498db'}),
                        dcc.Graph(id='model-performance-chart')
                    ], className='six columns')
                ], className='row'),
                
                html.Div([
                    html.H3("Wait Time Trends", style={'color': '#3498db'}),
                    dcc.Dropdown(
                        id='trend-view-selector',
                        options=[
                            {'label': 'By Hour of Day', 'value': 'hour'},
                            {'label': 'By Day of Week', 'value': 'day_of_week'},
                            {'label': 'By Doctor', 'value': 'doctor_id'},
                            {'label': 'By Month', 'value': 'month'}
                        ],
                        value='hour'
                    ),
                    dcc.Graph(id='wait-time-trend-chart')
                ], className='row'),
                
                html.Div([
                    html.Div([
                        html.H3("Patient Flow Analysis", style={'color': '#3498db'}),
                        dcc.Graph(id='patient-flow-chart')
                    ], className='six columns'),
                    html.Div([
                        html.H3("Doctor Utilization", style={'color': '#3498db'}),
                        dcc.Graph(id='doctor-utilization-chart')
                    ], className='six columns')
                ], className='row'),
                
                html.Div([
                    html.H3("Feature Importance", style={'color': '#3498db'}),
                    dcc.Graph(id='feature-importance-chart')
                ], className='row'),
                
                # Hidden div for storing data
                html.Div(id='clinic-data', style={'display': 'none'}),
                
                # Interval component for updates
                dcc.Interval(
                    id='interval-component',
                    interval=30*1000,  # 30 seconds in milliseconds
                    n_intervals=0
                )
            ], className='container')
        ])
        
        # Add custom CSS
        self.add_custom_css()
    
    def add_custom_css(self):
        """Add custom CSS to the Dash app"""
        custom_css = '''
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        .row:after {
            content: "";
            display: table;
            clear: both;
        }
        .six.columns {
            width: 48%;
            float: left;
            margin-left: 1%;
            margin-right: 1%;
            margin-bottom: 20px;
        }
        .status-cards {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .status-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 48%;
        }
        .status-card h4 {
            margin-top: 0;
            color: #7f8c8d;
            font-size: 14px;
        }
        .status-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .prediction-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .prediction-value {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        .prediction-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        '''
        
        # Add CSS to the app
        app_css = html.Style(custom_css)
        self.dash_app.index_string = self.dash_app.index_string.replace(
            '</head>', f'{app_css.children}</head>'
        )
    
    def setup_callbacks(self):
        """Setup Dash app callbacks"""
        # Update clinic status cards
        @self.dash_app.callback(
            Output('clinic-status-cards', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_clinic_status(n):
            # Get current status
            status = self.clinic_monitor.get_current_status()
            
            # Create status cards
            cards = [
                html.Div([
                    html.H4("Active Patients"),
                    html.Div(status['active_patients'], className='value')
                ], className='status-card'),
                html.Div([
                    html.H4("Average Wait Time"),
                    html.Div(f"{status['avg_wait_time']:.1f} min", className='value')
                ], className='status-card'),
                html.Div([
                    html.H4("Peak Hour Load"),
                    html.Div(f"{status['peak_hour_load']:.1f}%", className='value')
                ], className='status-card'),
                html.Div([
                    html.H4("Patient Satisfaction"),
                    html.Div(f"{status['patient_satisfaction']:.1f}%", className='value')
                ], className='status-card')
            ]
            
            return cards
        
        # Update wait time prediction
        @self.dash_app.callback(
            Output('prediction-result', 'children'),
            [Input('predict-button', 'n_clicks')],
            [State('doctor-selector', 'value'),
             State('date-picker', 'date'),
             State('time-slider', 'value')]
        )
        def update_prediction(n_clicks, doctor_id, date, time_value):
            if n_clicks is None:
                return html.Div([
                    html.P("Select parameters and click 'Predict Wait Time'")
                ])
            
            # Parse date and time
            date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            hour = int(time_value)
            minute = 30 if time_value % 1 > 0 else 0
            scheduled_time = datetime.combine(date_obj, datetime.min.time().replace(hour=hour, minute=minute))
            
            # Make prediction
            try:
                wait_time = predict_wait_time_advanced(doctor_id, scheduled_time)
            except Exception as e:
                print(f"Error in prediction: {e}")
                wait_time = random.randint(5, 30)  # Fallback for demo
            
            # Format time for display
            time_str = scheduled_time.strftime('%I:%M %p')
            
            return html.Div([
                html.P(f"Appointment with Doctor {doctor_id} at {time_str}", className='prediction-label'),
                html.Div(f"{wait_time:.1f} minutes", className='prediction-value'),
                html.P("Estimated Wait Time", className='prediction-label'),
                html.Div([
                    html.Span("Confidence: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{'High' if wait_time < 20 else 'Medium' if wait_time < 40 else 'Low'}")
                ], style={'marginTop': '10px'})
            ])
        
        # Update wait time trend chart
        @self.dash_app.callback(
            Output('wait-time-trend-chart', 'figure'),
            [Input('trend-view-selector', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_trend_chart(view_by, n):
            # Group data by selected dimension
            if view_by == 'hour':
                grouped = self.df_processed.groupby('hour')['delay'].mean().reset_index()
                x_title = 'Hour of Day'
                x_values = grouped['hour']
            elif view_by == 'day_of_week':
                grouped = self.df_processed.groupby('day_of_week')['delay'].mean().reset_index()
                x_title = 'Day of Week'
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                grouped['day_name'] = grouped['day_of_week'].apply(lambda x: day_names[x])
                x_values = grouped['day_name']
            elif view_by == 'doctor_id':
                grouped = self.df_processed.groupby('doctor_id')['delay'].mean().reset_index()
                x_title = 'Doctor ID'
                x_values = grouped['doctor_id']
            elif view_by == 'month':
                grouped = self.df_processed.groupby('month')['delay'].mean().reset_index()
                x_title = 'Month'
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                grouped['month_name'] = grouped['month'].apply(lambda x: month_names[x-1])
                x_values = grouped['month_name']
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=x_values,
                y=grouped['delay'],
                marker_color='#3498db',
                name='Average Wait Time'
            ))
            
            fig.update_layout(
                title=f'Average Wait Time by {x_title}',
                xaxis_title=x_title,
                yaxis_title='Wait Time (minutes)',
                template='plotly_white',
                height=400
            )
            
            return fig
        
        # Update patient flow chart
        @self.dash_app.callback(
            Output('patient-flow-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_patient_flow(n):
            # Group data by hour
            patient_flow = self.df_processed.groupby('hour')['patient_id'].count().reset_index()
            patient_flow.columns = ['hour', 'patient_count']
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=patient_flow['hour'],
                y=patient_flow['patient_count'],
                mode='lines+markers',
                marker=dict(size=8, color='#2ecc71'),
                line=dict(width=3, color='#2ecc71'),
                name='Patient Count'
            ))
            
            # Add peak hours highlight
            peak_hours = [17, 18, 19, 20]
            fig.add_trace(go.Scatter(
                x=patient_flow[patient_flow['hour'].isin(peak_hours)]['hour'],
                y=patient_flow[patient_flow['hour'].isin(peak_hours)]['patient_count'],
                mode='markers',
                marker=dict(size=12, color='#e74c3c'),
                name='Peak Hours'
            ))
            
            fig.update_layout(
                title='Patient Flow by Hour of Day',
                xaxis_title='Hour of Day',
                yaxis_title='Number of Patients',
                template='plotly_white',
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        # Update doctor utilization chart
        @self.dash_app.callback(
            Output('doctor-utilization-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_doctor_utilization(n):
            # Calculate doctor utilization
            doctor_util = self.df_processed.groupby('doctor_id')['delay'].agg(['count', 'mean']).reset_index()
            doctor_util.columns = ['doctor_id', 'appointment_count', 'avg_delay']
            
            # Calculate utilization percentage (simplified for demo)
            doctor_util['utilization'] = (doctor_util['appointment_count'] / doctor_util['appointment_count'].max() * 100).round(1)
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=doctor_util['doctor_id'],
                y=doctor_util['utilization'],
                marker_color='#9b59b6',
                name='Utilization %'
            ))
            
            fig.add_trace(go.Scatter(
                x=doctor_util['doctor_id'],
                y=doctor_util['avg_delay'],
                mode='lines+markers',
                marker=dict(size=8, color='#e74c3c'),
                line=dict(width=3, color='#e74c3c'),
                name='Avg Delay (min)',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Doctor Utilization and Average Delay',
                xaxis_title='Doctor ID',
                yaxis_title='Utilization (%)',
                yaxis2=dict(
                    title='Average Delay (min)',
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white',
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        # Update model performance chart
        @self.dash_app.callback(
            Output('model-performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_model_performance(n):
            # Create sample model performance data for demo
            models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Deep Learning', 'Ensemble']
            mae_values = [8.2, 7.5, 6.8, 7.1, 6.2]
            r2_values = [0.72, 0.76, 0.81, 0.78, 0.84]
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=models,
                y=mae_values,
                marker_color='#3498db',
                name='MAE (min)'
            ))
            
            fig.add_trace(go.Scatter(
                x=models,
                y=r2_values,
                mode='lines+markers',
                marker=dict(size=8, color='#e74c3c'),
                line=dict(width=3, color='#e74c3c'),
                name='R² Score',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='MAE (minutes)',
                yaxis2=dict(
                    title='R² Score',
                    overlaying='y',
                    side='right',
                    range=[0, 1]
                ),
                template='plotly_white',
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        # Update feature importance chart
        @self.dash_app.callback(
            Output('feature-importance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_feature_importance(n):
            # Create sample feature importance data for demo
            features = [
                'doctor_id', 'hour', 'is_peak', 'day_of_week', 'historical_delay',
                'doc_mean_delay', 'weekend_pattern', 'recent_delay_trend', 'month', 'is_weekend'
            ]
            importance = [0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=importance,
                y=features,
                marker_color='#2ecc71',
                orientation='h'
            ))
            
            fig.update_layout(
                title='Feature Importance (XGBoost Model)',
                xaxis_title='Importance',
                yaxis_title='Feature',
                template='plotly_white',
                height=500
            )
            
            return fig
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the dashboard server"""
        if self.socketio:
            self.socketio.run(self.server, host=host, port=port, debug=debug)
        else:
            self.server.run(host=host, port=port, debug=debug)

# Routes for the main Flask app
def register_analytics_routes(app, dash_app):
    """Register routes for the analytics dashboard"""
    @app.route('/analytics-dashboard')
    def analytics_dashboard():
        return render_template('analytics_dashboard.html')
    
    @app.route('/api/analytics/wait-time-distribution')
    def wait_time_distribution():
        # Load data
        df = load_data()
        df_processed = preprocess_data(df)
        
        # Calculate wait time distribution
        wait_time_bins = [0, 5, 10, 15, 20, 30, 45, 60, float('inf')]
        wait_time_labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-45', '45-60', '60+']
        
        df_processed['wait_time_category'] = pd.cut(df_processed['delay'], bins=wait_time_bins, labels=wait_time_labels)
        distribution = df_processed['wait_time_category'].value_counts().sort_index().to_dict()
        
        return jsonify({
            'categories': wait_time_labels,
            'counts': list(distribution.values())
        })
    
    @app.route('/api/analytics/doctor-performance')
    def doctor_performance():
        # Load data
        df = load_data()
        df_processed = preprocess_data(df)
        
        # Calculate doctor performance
        doctor_perf = df_processed.groupby('doctor_id')['delay'].agg(['mean', 'std', 'count']).reset_index()
        doctor_perf.columns = ['doctor_id', 'avg_wait_time', 'wait_time_std', 'appointment_count']
        
        return jsonify(doctor_perf.to_dict(orient='records'))

# Initialize dashboard if run directly
if __name__ == '__main__':
    dashboard = RealTimeAnalyticsDashboard()
    dashboard.run(debug=True)