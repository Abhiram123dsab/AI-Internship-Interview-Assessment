import time
import random
import threading
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class AdvancedEffectivenessMeasurement:
    """Advanced system for measuring and analyzing communication effectiveness"""
    
    def __init__(self):
        # Initialize core metrics tracking
        self.delivery_status = {
            'SMS': {'sent': 0, 'delivered': 0, 'failed': 0},
            'WhatsApp': {'sent': 0, 'delivered': 0, 'failed': 0},
            'IVR': {'sent': 0, 'delivered': 0, 'failed': 0}
        }
        
        # Response time tracking (in seconds)
        self.response_times = {
            'SMS': [],
            'WhatsApp': [],
            'IVR': []
        }
        
        # Channel health monitoring
        self.channel_health = {
            'SMS': {'uptime': 100.0, 'latency': 0.5, 'error_rate': 0.0, 'last_check': datetime.now()},
            'WhatsApp': {'uptime': 100.0, 'latency': 0.8, 'error_rate': 0.0, 'last_check': datetime.now()},
            'IVR': {'uptime': 100.0, 'latency': 1.2, 'error_rate': 0.0, 'last_check': datetime.now()}
        }
        
        # Demographic tracking
        self.demographic_metrics = {
            'language': {
                'Tamil': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
                'Telugu': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
                'Malayalam': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
                'Hindi': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
                'English': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []}
            },
            'age_group': {
                '<30': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
                '30-50': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
                '50-65': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
                '>65': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []}
            }
        }
        
        # Hourly engagement metrics
        self.hourly_metrics = {hour: {'sent': 0, 'delivered': 0, 'responded': 0} for hour in range(24)}
        
        # Message type effectiveness
        self.message_type_metrics = {
            'appointment_confirmation': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
            'wait_time': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []},
            'prescription_reminder': {'sent': 0, 'delivered': 0, 'responded': 0, 'sentiment': []}
        }
        
        # Real-time monitoring data (last 24 hours, 5-minute intervals)
        self.monitoring_interval = 5  # minutes
        self.monitoring_window = 24 * 60 // self.monitoring_interval  # 24 hours of 5-min intervals
        self.real_time_data = {
            'delivery_rate': deque(maxlen=self.monitoring_window),
            'response_rate': deque(maxlen=self.monitoring_window),
            'sentiment': deque(maxlen=self.monitoring_window),
            'channel_errors': {
                'SMS': deque(maxlen=self.monitoring_window),
                'WhatsApp': deque(maxlen=self.monitoring_window),
                'IVR': deque(maxlen=self.monitoring_window)
            },
            'timestamps': deque(maxlen=self.monitoring_window)
        }
        
        # Initialize real-time monitoring
        self._initialize_real_time_data()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_channels)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _initialize_real_time_data(self):
        """Initialize real-time monitoring data with baseline values"""
        now = datetime.now()
        for i in range(self.monitoring_window):
            timestamp = now - timedelta(minutes=self.monitoring_interval * (self.monitoring_window - i))
            self.real_time_data['timestamps'].append(timestamp)
            self.real_time_data['delivery_rate'].append(0.95 + random.uniform(-0.05, 0.05))
            self.real_time_data['response_rate'].append(0.4 + random.uniform(-0.1, 0.1))
            self.real_time_data['sentiment'].append(0.2 + random.uniform(-0.2, 0.2))
            
            for channel in self.real_time_data['channel_errors']:
                self.real_time_data['channel_errors'][channel].append(random.uniform(0, 0.02))
    
    def _monitor_channels(self):
        """Background thread to monitor channel health"""
        while self.monitoring_active:
            # Simulate checking channel health
            for channel in self.channel_health:
                # Simulate random fluctuations in channel health
                self.channel_health[channel]['latency'] = max(0.1, min(5.0, 
                    self.channel_health[channel]['latency'] + random.uniform(-0.2, 0.2)))
                
                # Simulate occasional errors
                error_chance = random.random()
                if error_chance < 0.05:  # 5% chance of error spike
                    self.channel_health[channel]['error_rate'] = random.uniform(0.05, 0.2)
                else:
                    self.channel_health[channel]['error_rate'] = max(0, min(0.05, 
                        self.channel_health[channel]['error_rate'] + random.uniform(-0.01, 0.01)))
                
                # Update uptime based on error rate
                if self.channel_health[channel]['error_rate'] > 0.1:
                    self.channel_health[channel]['uptime'] = max(90.0, 
                        self.channel_health[channel]['uptime'] - random.uniform(0, 1.0))
                else:
                    self.channel_health[channel]['uptime'] = min(100.0, 
                        self.channel_health[channel]['uptime'] + random.uniform(0, 0.5))
                
                self.channel_health[channel]['last_check'] = datetime.now()
            
            # Update real-time monitoring data
            self._update_real_time_data()
            
            # Sleep for monitoring interval
            time.sleep(self.monitoring_interval * 60 / 10)  # Accelerated for simulation
    
    def _update_real_time_data(self):
        """Update real-time monitoring data with current metrics"""
        now = datetime.now()
        
        # Calculate current delivery rate across all channels
        total_sent = sum(self.delivery_status[channel]['sent'] for channel in self.delivery_status)
        total_delivered = sum(self.delivery_status[channel]['delivered'] for channel in self.delivery_status)
        delivery_rate = total_delivered / total_sent if total_sent > 0 else 0.95
        
        # Calculate current response rate
        total_responded = sum(self.demographic_metrics['language'][lang]['responded'] 
                             for lang in self.demographic_metrics['language'])
        response_rate = total_responded / total_delivered if total_delivered > 0 else 0.4
        
        # Calculate average sentiment
        all_sentiments = []
        for lang in self.demographic_metrics['language']:
            all_sentiments.extend(self.demographic_metrics['language'][lang]['sentiment'])
        avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.2
        
        # Update real-time data
        self.real_time_data['timestamps'].append(now)
        self.real_time_data['delivery_rate'].append(delivery_rate)
        self.real_time_data['response_rate'].append(response_rate)
        self.real_time_data['sentiment'].append(avg_sentiment)
        
        # Update channel error rates
        for channel in self.channel_health:
            self.real_time_data['channel_errors'][channel].append(self.channel_health[channel]['error_rate'])
    
    def track_message_delivery(self, patient, channel, message_type, status='delivered'):
        """Track message delivery status"""
        # Update delivery status counts
        self.delivery_status[channel]['sent'] += 1
        if status == 'delivered':
            self.delivery_status[channel]['delivered'] += 1
        elif status == 'failed':
            self.delivery_status[channel]['failed'] += 1
        
        # Update demographic metrics
        language = patient['language']
        if language not in self.demographic_metrics['language']:
            language = 'English'  # Default to English if language not found
        
        self.demographic_metrics['language'][language]['sent'] += 1
        if status == 'delivered':
            self.demographic_metrics['language'][language]['delivered'] += 1
        
        # Update age group metrics
        age = patient['age']
        age_group = '<30' if age < 30 else '30-50' if age < 50 else '50-65' if age < 65 else '>65'
        
        self.demographic_metrics['age_group'][age_group]['sent'] += 1
        if status == 'delivered':
            self.demographic_metrics['age_group'][age_group]['delivered'] += 1
        
        # Update hourly metrics
        current_hour = datetime.now().hour
        self.hourly_metrics[current_hour]['sent'] += 1
        if status == 'delivered':
            self.hourly_metrics[current_hour]['delivered'] += 1
        
        # Update message type metrics
        self.message_type_metrics[message_type]['sent'] += 1
        if status == 'delivered':
            self.message_type_metrics[message_type]['delivered'] += 1
        
        return {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'channel': channel,
            'message_type': message_type
        }
    
    def track_patient_response(self, patient, channel, message_type, response_text, response_time):
        """Track patient response and analyze sentiment"""
        # Update response counts
        language = patient['language']
        if language not in self.demographic_metrics['language']:
            language = 'English'
        
        self.demographic_metrics['language'][language]['responded'] += 1
        
        # Update age group response
        age = patient['age']
        age_group = '<30' if age < 30 else '30-50' if age < 50 else '50-65' if age < 65 else '>65'
        self.demographic_metrics['age_group'][age_group]['responded'] += 1
        
        # Update hourly metrics
        current_hour = datetime.now().hour
        self.hourly_metrics[current_hour]['responded'] += 1
        
        # Update message type metrics
        self.message_type_metrics[message_type]['responded'] += 1
        
        # Track response time
        self.response_times[channel].append(response_time)
        
        # Perform sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(response_text)
        sentiment_score = sentiment_scores['compound']
        
        # Store sentiment score
        self.demographic_metrics['language'][language]['sentiment'].append(sentiment_score)
        self.demographic_metrics['age_group'][age_group]['sentiment'].append(sentiment_score)
        self.message_type_metrics[message_type]['sentiment'].append(sentiment_score)
        
        return {
            'response_time': response_time,
            'sentiment_score': sentiment_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_channel_effectiveness_index(self):
        """Calculate effectiveness index for each channel"""
        effectiveness_index = {}
        
        for channel in self.delivery_status:
            # Calculate delivery success rate
            sent = self.delivery_status[channel]['sent']
            delivered = self.delivery_status[channel]['delivered']
            delivery_rate = delivered / sent if sent > 0 else 0
            
            # Calculate average response time
            avg_response_time = sum(self.response_times[channel]) / len(self.response_times[channel]) \
                if self.response_times[channel] else 0
            
            # Calculate response rate (based on demographic data)
            total_responses = 0
            for language in self.demographic_metrics['language']:
                total_responses += self.demographic_metrics['language'][language]['responded']
            
            response_rate = total_responses / delivered if delivered > 0 else 0
            
            # Calculate channel health score
            health_score = (self.channel_health[channel]['uptime'] / 100) * \
                          (1 - self.channel_health[channel]['error_rate']) * \
                          (1 / (1 + self.channel_health[channel]['latency'] / 10))
            
            # Calculate weighted effectiveness score
            effectiveness_score = (
                0.4 * delivery_rate +
                0.3 * response_rate +
                0.2 * (1 / (1 + avg_response_time / 300)) +  # Normalize response time (5 min optimal)
                0.1 * health_score
            )
            
            effectiveness_index[channel] = {
                'score': effectiveness_score,
                'delivery_rate': delivery_rate,
                'response_rate': response_rate,
                'avg_response_time': avg_response_time,
                'health_score': health_score
            }
        
        return effectiveness_index
    
    def get_demographic_performance_analysis(self):
        """Analyze performance across different demographic segments"""
        performance = {
            'language': {},
            'age_group': {}
        }
        
        # Analyze language performance
        for language, metrics in self.demographic_metrics['language'].items():
            if metrics['sent'] > 0:
                delivery_rate = metrics['delivered'] / metrics['sent']
                response_rate = metrics['responded'] / metrics['delivered'] if metrics['delivered'] > 0 else 0
                avg_sentiment = sum(metrics['sentiment']) / len(metrics['sentiment']) if metrics['sentiment'] else 0
                
                performance['language'][language] = {
                    'delivery_rate': delivery_rate,
                    'response_rate': response_rate,
                    'avg_sentiment': avg_sentiment,
                    'effectiveness_score': (0.4 * delivery_rate + 0.4 * response_rate + 0.2 * (avg_sentiment + 1) / 2)
                }
        
        # Analyze age group performance
        for age_group, metrics in self.demographic_metrics['age_group'].items():
            if metrics['sent'] > 0:
                delivery_rate = metrics['delivered'] / metrics['sent']
                response_rate = metrics['responded'] / metrics['delivered'] if metrics['delivered'] > 0 else 0
                avg_sentiment = sum(metrics['sentiment']) / len(metrics['sentiment']) if metrics['sentiment'] else 0
                
                performance['age_group'][age_group] = {
                    'delivery_rate': delivery_rate,
                    'response_rate': response_rate,
                    'avg_sentiment': avg_sentiment,
                    'effectiveness_score': (0.4 * delivery_rate + 0.4 * response_rate + 0.2 * (avg_sentiment + 1) / 2)
                }
        
        return performance
    
    def get_hourly_engagement_metrics(self):
        """Get engagement metrics by hour of day"""
        hourly_engagement = {}
        
        for hour, metrics in self.hourly_metrics.items():
            if metrics['sent'] > 0:
                delivery_rate = metrics['delivered'] / metrics['sent']
                response_rate = metrics['responded'] / metrics['delivered'] if metrics['delivered'] > 0 else 0
                
                hourly_engagement[hour] = {
                    'delivery_rate': delivery_rate,
                    'response_rate': response_rate,
                    'engagement_score': (delivery_rate + response_rate) / 2
                }
        
        # Identify peak engagement hours
        if hourly_engagement:
            peak_hours = sorted(hourly_engagement.items(), 
                               key=lambda x: x[1]['engagement_score'], 
                               reverse=True)[:3]
            
            return {
                'hourly_data': hourly_engagement,
                'peak_hours': [hour for hour, _ in peak_hours],
                'peak_scores': [metrics['engagement_score'] for _, metrics in peak_hours]
            }
        
        return {'hourly_data': {}, 'peak_hours': [], 'peak_scores': []}
    
    def get_message_type_effectiveness(self):
        """Analyze effectiveness by message type"""
        effectiveness = {}
        
        for msg_type, metrics in self.message_type_metrics.items():
            if metrics['sent'] > 0:
                delivery_rate = metrics['delivered'] / metrics['sent']
                response_rate = metrics['responded'] / metrics['delivered'] if metrics['delivered'] > 0 else 0
                avg_sentiment = sum(metrics['sentiment']) / len(metrics['sentiment']) if metrics['sentiment'] else 0
                
                effectiveness[msg_type] = {
                    'delivery_rate': delivery_rate,
                    'response_rate': response_rate,
                    'avg_sentiment': avg_sentiment,
                    'effectiveness_score': (0.4 * delivery_rate + 0.4 * response_rate + 0.2 * (avg_sentiment + 1) / 2)
                }
        
        return effectiveness
    
    def get_real_time_performance_metrics(self):
        """Get real-time performance metrics for monitoring"""
        # Get the most recent metrics
        current_metrics = {
            'timestamp': self.real_time_data['timestamps'][-1] if self.real_time_data['timestamps'] else datetime.now(),
            'delivery_rate': self.real_time_data['delivery_rate'][-1] if self.real_time_data['delivery_rate'] else 0,
            'response_rate': self.real_time_data['response_rate'][-1] if self.real_time_data['response_rate'] else 0,
            'sentiment': self.real_time_data['sentiment'][-1] if self.real_time_data['sentiment'] else 0,
            'channel_health': {}
        }
        
        # Add channel health metrics
        for channel in self.channel_health:
            current_metrics['channel_health'][channel] = {
                'uptime': self.channel_health[channel]['uptime'],
                'latency': self.channel_health[channel]['latency'],
                'error_rate': self.channel_health[channel]['error_rate'],
                'status': 'Healthy' if self.channel_health[channel]['error_rate'] < 0.05 else 'Degraded' if self.channel_health[channel]['error_rate'] < 0.1 else 'Critical'
            }
        
        # Add trend data (last hour)
        trend_window = 60 // self.monitoring_interval  # Last hour of data points
        
        if len(self.real_time_data['delivery_rate']) >= trend_window:
            current_metrics['trends'] = {
                'delivery_rate': {
                    'current': current_metrics['delivery_rate'],
                    'change': current_metrics['delivery_rate'] - self.real_time_data['delivery_rate'][-trend_window]
                },
                'response_rate': {
                    'current': current_metrics['response_rate'],
                    'change': current_metrics['response_rate'] - self.real_time_data['response_rate'][-trend_window]
                },
                'sentiment': {
                    'current': current_metrics['sentiment'],
                    'change': current_metrics['sentiment'] - self.real_time_data['sentiment'][-trend_window]
                }
            }
        
        return current_metrics
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'channel_effectiveness': self.get_channel_effectiveness_index(),
            'demographic_performance': self.get_demographic_performance_analysis(),
            'hourly_engagement': self.get_hourly_engagement_metrics(),
            'message_effectiveness': self.get_message_type_effectiveness(),
            'real_time_metrics': self.get_real_time_performance_metrics(),
            'summary_metrics': {
                'overall_delivery_rate': 0,
                'overall_response_rate': 0,
                'overall_sentiment': 0,
                'best_performing_channel': '',
                'best_performing_language': '',
                'best_performing_age_group': '',
                'best_performing_message_type': ''
            }
        }
        
        # Calculate overall metrics
        total_sent = sum(self.delivery_status[channel]['sent'] for channel in self.delivery_status)
        total_delivered = sum(self.delivery_status[channel]['delivered'] for channel in self.delivery_status)
        
        if total_sent > 0:
            report['summary_metrics']['overall_delivery_rate'] = total_delivered / total_sent
        
        total_responses = 0
        for language in self.demographic_metrics['language']:
            total_responses += self.demographic_metrics['language'][language]['responded']
        
        if total_delivered > 0:
            report['summary_metrics']['overall_response_rate'] = total_responses / total_delivered
        
        # Calculate overall sentiment
        all_sentiments = []
        for msg_type in self.message_type_metrics:
            all_sentiments.extend(self.message_type_metrics[msg_type]['sentiment'])
        
        if all_sentiments:
            report['summary_metrics']['overall_sentiment'] = sum(all_sentiments) / len(all_sentiments)
        
        # Find best performing elements
        if report['channel_effectiveness']:
            report['summary_metrics']['best_performing_channel'] = max(
                report['channel_effectiveness'].items(),
                key=lambda x: x[1]['score']
            )[0]
        
        if report['demographic_performance']['language']:
            report['summary_metrics']['best_performing_language'] = max(
                report['demographic_performance']['language'].items(),
                key=lambda x: x[1]['effectiveness_score']
            )[0]
        
        if report['demographic_performance']['age_group']:
            report['summary_metrics']['best_performing_age_group'] = max(
                report['demographic_performance']['age_group'].items(),
                key=lambda x: x[1]['effectiveness_score']
            )[0]
        
        if report['message_effectiveness']:
            report['summary_metrics']['best_performing_message_type'] = max(
                report['message_effectiveness'].items(),
                key=lambda x: x[1]['effectiveness_score']
            )[0]
        
        return report
    
    def visualize_performance_metrics(self, save_path=None):
        """Generate visualizations of key performance metrics"""
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('Advanced Effectiveness Measurement System - Performance Dashboard', fontsize=16)
        
        # 1. Channel Effectiveness Comparison
        ax1 = fig.add_subplot(2, 2, 1)
        channel_effectiveness = self.get_channel_effectiveness_index()
        channels = list(channel_effectiveness.keys())
        scores = [channel_effectiveness[channel]['score'] for channel in channels]
        
        ax1.bar(channels, scores, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.set_title('Channel Effectiveness Comparison')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Effectiveness Score')
        
        # 2. Demographic Performance by Language
        ax2 = fig.add_subplot(2, 2, 2)
        demo_performance = self.get_demographic_performance_analysis()
        languages = list(demo_performance['language'].keys())
        language_scores = [demo_performance['language'][lang]['effectiveness_score'] for lang in languages]
        
        ax2.bar(languages, language_scores, color='#9b59b6')
        ax2.set_title('Language Performance Comparison')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Effectiveness Score')
        
        # 3. Hourly Engagement Heatmap
        ax3 = fig.add_subplot(2, 2, 3)
        hourly_data = self.get_hourly_engagement_metrics()['hourly_data']
        hours = sorted(hourly_data.keys())
        engagement_scores = [hourly_data[hour]['engagement_score'] if hour in hourly_data else 0 for hour in range(24)]
        
        ax3.plot(range(24), engagement_scores, marker='o', linestyle='-', color='#f39c12')
        ax3.set_title('Hourly Engagement Pattern')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Engagement Score')
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Real-time Metrics Trend
        ax4 = fig.add_subplot(2, 2, 4)
        timestamps = list(self.real_time_data['timestamps'])
        delivery_rates = list(self.real_time_data['delivery_rate'])
        response_rates = list(self.real_time_data['response_rate'])
        
        # Convert timestamps to hours for better readability
        hours = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
        
        ax4.plot(hours, delivery_rates, label='Delivery Rate', color='#3498db')
        ax4.plot(hours, response_rates, label='Response Rate', color='#2ecc71')
        ax4.set_title('Real-time Performance Trends')
        ax4.set_xlabel('Hours (Relative)')
        ax4.set_ylabel('Rate')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            plt.show()
            return None
    
    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

# Example usage
def demo_advanced_effectiveness_measurement():
    # Initialize the system
    measurement_system = AdvancedEffectivenessMeasurement()
    
    # Sample patients
    patients = [
        {"id": 1, "name": "Ravi Kumar", "language": "Tamil", "age": 72, "channel": "SMS"},
        {"id": 2, "name": "Ananya Rao", "language": "Telugu", "age": 35, "channel": "WhatsApp"},
        {"id": 3, "name": "Joseph Mathew", "language": "Malayalam", "age": 68, "channel": "IVR"},
        {"id": 4, "name": "Rahul Sharma", "language": "Hindi", "age": 42, "channel": "SMS"},
        {"id": 5, "name": "David Thomas", "language": "English", "age": 29, "channel": "WhatsApp"},
    ]
    
    # Simulate message deliveries and responses
    message_types = ["appointment_confirmation", "wait_time", "prescription_reminder"]
    response_texts = [
        "Thank you for the reminder",
        "I'll be there on time",
        "Can I reschedule my appointment?",
        "The wait time is too long",
        "I've taken my medication"
    ]
    
    print("Simulating patient communications...")
    for _ in range(50):  # Simulate 50 messages
        # Select random patient, channel and message type
        patient = random.choice(patients)
        channel = patient['channel']
        message_type = random.choice(message_types)
        
        # Track message delivery
        delivery_status = 'delivered' if random.random() < 0.95 else 'failed'
        delivery_result = measurement_system.track_message_delivery(
            patient, channel, message_type, status=delivery_status
        )
        
        # If message was delivered, simulate patient response
        if delivery_status == 'delivered' and random.random() < 0.7:  # 70% response rate
            response_text = random.choice(response_texts)
            response_time = random.uniform(30, 600)  # 30 seconds to 10 minutes
            
            response_result = measurement_system.track_patient_response(
                patient, channel, message_type, response_text, response_time
            )
    
    # Generate and print comprehensive report
    print("\n" + "=" * 80)
    print("ADVANCED EFFECTIVENESS MEASUREMENT SYSTEM - COMPREHENSIVE REPORT")
    print("=" * 80)
    
    report = measurement_system.generate_comprehensive_report()
    
    # Print summary metrics
    print("\n📊 SUMMARY METRICS:")
    print(f"Overall Delivery Rate: {report['summary_metrics']['overall_delivery_rate']:.2%}")
    print(f"Overall Response Rate: {report['summary_metrics']['overall_response_rate']:.2%}")
    print(f"Overall Sentiment Score: {report['summary_metrics']['overall_sentiment']:.2f}")
    print(f"Best Performing Channel: {report['summary_metrics']['best_performing_channel']}")
    print(f"Best Performing Language: {report['summary_metrics']['best_performing_language']}")
    print(f"Best Performing Age Group: {report['summary_metrics']['best_performing_age_group']}")
    print(f"Best Performing Message Type: {report['summary_metrics']['best_performing_message_type']}")
    
    # Print channel effectiveness
    print("\n📱 CHANNEL EFFECTIVENESS INDEX:")
    for channel, metrics in report['channel_effectiveness'].items():
        print(f"{channel}:")
        print(f"  - Effectiveness Score: {metrics['score']:.2f}")
        print(f"  - Delivery Rate: {metrics['delivery_rate']:.2%}")
        print(f"  - Response Rate: {metrics['response_rate']:.2%}")
        print(f"  - Avg Response Time: {metrics['avg_response_time']:.1f} seconds")
    
    # Print demographic performance
    print("\n👥 DEMOGRAPHIC PERFORMANCE:")
    print("Language Performance:")
    for language, metrics in report['demographic_performance']['language'].items():
        print(f"  {language}:")
        print(f"    - Effectiveness Score: {metrics['effectiveness_score']:.2f}")
        print(f"    - Response Rate: {metrics['response_rate']:.2%}")
        print(f"    - Avg Sentiment: {metrics['avg_sentiment']:.2f}")
    
    print("\nAge Group Performance:")
    for age_group, metrics in report['demographic_performance']['age_group'].items():
        print(f"  {age_group}:")
        print(f"    - Effectiveness Score: {metrics['effectiveness_score']:.2f}")
        print(f"    - Response Rate: {metrics['response_rate']:.2%}")
        print(f"    - Avg Sentiment: {metrics['avg_sentiment']:.2f}")
    
    # Print real-time metrics
    print("\n⚡ REAL-TIME PERFORMANCE METRICS:")
    rt_metrics = report['real_time_metrics']
    print(f"Current Delivery Rate: {rt_metrics['delivery_rate']:.2%}")
    print(f"Current Response Rate: {rt_metrics['response_rate']:.2%}")
    print(f"Current Sentiment Score: {rt_metrics['sentiment']:.2f}")
    
    print("\nChannel Health Status:")
    for channel, health in rt_metrics['channel_health'].items():
        print(f"  {channel}:")
        print(f"    - Status: {health['status']}")
        print(f"    - Uptime: {health['uptime']:.2f}%")
        print(f"    - Latency: {health['latency']:.2f} seconds")
        print(f"    - Error Rate: {health['error_rate']:.2%}")
    
    # Generate visualization
    print("\nGenerating performance visualization...")
    visualization_path = measurement_system.visualize_performance_metrics("effectiveness_dashboard.png")
    print(f"Visualization saved to: {visualization_path}")
    
    # Stop monitoring thread
    measurement_system.stop_monitoring()
    print("\nAdvanced Effectiveness Measurement System Demo Complete")

if __name__ == "__main__":
    demo_advanced_effectiveness_measurement()