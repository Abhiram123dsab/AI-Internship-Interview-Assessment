import random
from datetime import datetime
from solution import CommunicationManager, ChannelOptimizer, MessageOptimizer
from advanced_effectiveness_measurement import AdvancedEffectivenessMeasurement

class EnhancedCommunicationManager(CommunicationManager):
    """
    Enhanced Communication Manager that integrates the Advanced Effectiveness Measurement System
    with the existing communication infrastructure.
    """
    
    def __init__(self):
        # Initialize the parent class
        super().__init__()
        
        # Initialize the advanced effectiveness measurement system
        self.advanced_metrics = AdvancedEffectivenessMeasurement()
        
        # Additional metrics for cross-channel analysis
        self.cross_channel_metrics = {
            'channel_switching': {
                'SMS_to_WhatsApp': 0,
                'SMS_to_IVR': 0,
                'WhatsApp_to_SMS': 0,
                'WhatsApp_to_IVR': 0,
                'IVR_to_SMS': 0,
                'IVR_to_WhatsApp': 0
            },
            'patient_channel_history': {}
        }
    
    def send_message(self, patient, message_type, **kwargs):
        # Get the result from the parent method
        result = super().send_message(patient, message_type, **kwargs)
        
        # Track message delivery in advanced measurement system
        delivery_status = 'delivered' if result['success'] else 'failed'
        self.advanced_metrics.track_message_delivery(
            patient, 
            result['channel'], 
            message_type, 
            status=delivery_status
        )
        
        # Track channel switching if applicable
        self._track_channel_switching(patient, result['channel'])
        
        return result
    
    def process_response(self, patient_id, response_text, timestamp):
        # Get the result from the parent method
        response_data = super().process_response(patient_id, response_text, timestamp)
        
        # Find the patient and their channel
        patient = None
        for p in self.message_history.get(patient_id, []):
            if p.get('patient'):
                patient = p['patient']
                break
        
        if patient:
            # Calculate response time
            last_message_time = None
            for msg in self.message_history.get(patient_id, []):
                if msg.get('timestamp'):
                    last_message_time = msg['timestamp']
            
            if last_message_time:
                response_time = (timestamp - last_message_time).total_seconds()
                
                # Track in advanced measurement system
                channel = patient.get('channel', 'SMS')  # Default to SMS if not found
                message_type = 'appointment_confirmation'  # Default if not found
                
                for msg in self.message_history.get(patient_id, []):
                    if msg.get('message_type'):
                        message_type = msg['message_type']
                        break
                
                self.advanced_metrics.track_patient_response(
                    patient,
                    channel,
                    message_type,
                    response_text,
                    response_time
                )
        
        return response_data
    
    def _track_channel_switching(self, patient, current_channel):
        """
        Track when patients switch between communication channels
        """
        patient_id = patient['id']
        
        # Initialize patient history if not exists
        if patient_id not in self.cross_channel_metrics['patient_channel_history']:
            self.cross_channel_metrics['patient_channel_history'][patient_id] = {
                'last_channel': current_channel,
                'channel_history': [current_channel]
            }
            return
        
        # Get last channel used
        last_channel = self.cross_channel_metrics['patient_channel_history'][patient_id]['last_channel']
        
        # If channel changed, track the switch
        if last_channel != current_channel:
            switch_key = f"{last_channel}_to_{current_channel}"
            if switch_key in self.cross_channel_metrics['channel_switching']:
                self.cross_channel_metrics['channel_switching'][switch_key] += 1
            
            # Update patient history
            self.cross_channel_metrics['patient_channel_history'][patient_id]['last_channel'] = current_channel
            self.cross_channel_metrics['patient_channel_history'][patient_id]['channel_history'].append(current_channel)
    
    def get_enhanced_system_performance(self):
        """
        Get comprehensive system performance metrics combining both basic and advanced analytics
        """
        # Get basic performance metrics
        basic_metrics = self.get_system_performance()
        
        # Get advanced performance metrics
        advanced_report = self.advanced_metrics.generate_comprehensive_report()
        
        # Combine metrics
        enhanced_metrics = {
            **basic_metrics,
            'advanced_metrics': {
                'channel_effectiveness': advanced_report['channel_effectiveness'],
                'demographic_performance': advanced_report['demographic_performance'],
                'message_effectiveness': advanced_report['message_effectiveness'],
                'real_time_metrics': advanced_report['real_time_metrics'],
                'summary_metrics': advanced_report['summary_metrics']
            },
            'cross_channel_analysis': self.cross_channel_metrics['channel_switching'],
        }
        
        return enhanced_metrics
    
    def generate_effectiveness_visualization(self, save_path=None):
        """
        Generate visualization of effectiveness metrics
        """
        return self.advanced_metrics.visualize_performance_metrics(save_path)
    
    def get_channel_health_status(self):
        """
        Get real-time health status of all communication channels
        """
        rt_metrics = self.advanced_metrics.get_real_time_performance_metrics()
        return rt_metrics['channel_health']
    
    def get_delivery_status_tracking(self):
        """
        Get detailed delivery status tracking across all channels
        """
        return {
            channel: {
                'success_rate': stats['delivered'] / stats['sent'] if stats['sent'] > 0 else 0,
                'failure_rate': stats['failed'] / stats['sent'] if stats['sent'] > 0 else 0,
                'total_sent': stats['sent'],
                'total_delivered': stats['delivered'],
                'total_failed': stats['failed']
            }
            for channel, stats in self.advanced_metrics.delivery_status.items()
        }
    
    def get_response_time_analytics(self):
        """
        Get detailed response time analytics across all channels
        """
        response_analytics = {}
        
        for channel, times in self.advanced_metrics.response_times.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times) if times else 0
                max_time = max(times) if times else 0
                
                # Calculate percentiles
                sorted_times = sorted(times)
                p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
                p90_idx = int(len(sorted_times) * 0.9)
                p90 = sorted_times[p90_idx] if p90_idx < len(sorted_times) else 0
                
                response_analytics[channel] = {
                    'average': avg_time,
                    'minimum': min_time,
                    'maximum': max_time,
                    'median': p50,
                    'p90': p90,
                    'sample_size': len(times)
                }
            else:
                response_analytics[channel] = {
                    'average': 0,
                    'minimum': 0,
                    'maximum': 0,
                    'median': 0,
                    'p90': 0,
                    'sample_size': 0
                }
        
        return response_analytics

# Enhanced demo function to showcase the integrated system
def demo_enhanced_communication_system():
    # Initialize the enhanced communication manager
    comm_manager = EnhancedCommunicationManager()
    
    # Sample patients
    patients = [
        {"id": 1, "name": "Ravi Kumar", "language": "Tamil", "age": 72, "channel": "SMS", "preferred_channels": ["SMS", "IVR"]},
        {"id": 2, "name": "Ananya Rao", "language": "Telugu", "age": 35, "channel": "WhatsApp", "preferred_channels": ["WhatsApp", "SMS"]},
        {"id": 3, "name": "Joseph Mathew", "language": "Malayalam", "age": 68, "channel": "IVR", "preferred_channels": ["IVR"]},
        {"id": 4, "name": "Rahul Sharma", "language": "Hindi", "age": 42, "channel": "SMS", "preferred_channels": ["SMS", "WhatsApp"]},
        {"id": 5, "name": "David Thomas", "language": "English", "age": 29, "channel": "WhatsApp", "preferred_channels": ["WhatsApp"]},
    ]
    
    # Message types
    message_types = ["appointment_confirmation", "wait_time", "prescription_reminder"]
    
    # Simulate communications
    print("\n🏥 Enhanced Multi-Language Patient Communication System Demo")
    print("=" * 70)
    
    print("\n📱 Simulating patient communications...")
    for _ in range(30):  # Simulate 30 messages
        # Select random patient and message type
        patient = random.choice(patients)
        message_type = random.choice(message_types)
        
        # Prepare message parameters
        params = {}
        if message_type == "appointment_confirmation":
            params = {"date": "2023-04-15", "time": "10:30 AM"}
        elif message_type == "wait_time":
            params = {"wait_time": random.randint(5, 30)}
        elif message_type == "prescription_reminder":
            params = {"medicine": "Metformin", "next_time": "6:00 PM"}
        
        # Send message
        result = comm_manager.send_message(patient, message_type, **params)
        
        # Simulate patient response (70% chance)
        if random.random() < 0.7:
            response_options = [
                "Thank you for the information",
                "I confirm my appointment",
                "I'll be there on time",
                "Thanks for the reminder",
                "I need to reschedule",
                "The wait time is too long"
            ]
            
            response_text = random.choice(response_options)
            response_time = datetime.now()
            
            # Process response
            comm_manager.process_response(patient["id"], response_text, response_time)
    
    # Generate and display comprehensive performance report
    print("\n📊 Generating Enhanced Performance Report...")
    performance = comm_manager.get_enhanced_system_performance()
    
    # Display key metrics
    print("\n🔍 KEY PERFORMANCE INDICATORS:")
    print(f"Overall Delivery Success Rate: {performance['advanced_metrics']['summary_metrics']['overall_delivery_rate']:.2%}")
    print(f"Overall Patient Response Rate: {performance['advanced_metrics']['summary_metrics']['overall_response_rate']:.2%}")
    print(f"Average Sentiment Score: {performance['advanced_metrics']['summary_metrics']['overall_sentiment']:.2f}")
    
    # Display channel effectiveness
    print("\n📱 CHANNEL EFFECTIVENESS:")
    for channel, metrics in performance['advanced_metrics']['channel_effectiveness'].items():
        print(f"{channel}: {metrics['score']:.2f} effectiveness score")
    
    # Display real-time channel health
    print("\n⚡ REAL-TIME CHANNEL HEALTH:")
    channel_health = comm_manager.get_channel_health_status()
    for channel, health in channel_health.items():
        print(f"{channel}: {health['status']} (Uptime: {health['uptime']:.1f}%, Error Rate: {health['error_rate']:.2%})")
    
    # Display delivery status tracking
    print("\n📬 DELIVERY STATUS TRACKING:")
    delivery_stats = comm_manager.get_delivery_status_tracking()
    for channel, stats in delivery_stats.items():
        print(f"{channel}: {stats['success_rate']:.2%} success rate ({stats['total_delivered']}/{stats['total_sent']} messages)")
    
    # Display response time analytics
    print("\n⏱️ RESPONSE TIME ANALYTICS:")
    response_analytics = comm_manager.get_response_time_analytics()
    for channel, analytics in response_analytics.items():
        if analytics['sample_size'] > 0:
            print(f"{channel}: Avg {analytics['average']:.1f}s, Median {analytics['median']:.1f}s, 90th %ile {analytics['p90']:.1f}s")
    
    # Generate visualization
    print("\n🖼️ Generating Effectiveness Visualization...")
    visualization_path = comm_manager.generate_effectiveness_visualization("enhanced_effectiveness_dashboard.png")
    print(f"Visualization saved to: {visualization_path}")
    
    print("\n✅ Enhanced Communication System Demo Complete")

if __name__ == "__main__":
    demo_enhanced_communication_system()