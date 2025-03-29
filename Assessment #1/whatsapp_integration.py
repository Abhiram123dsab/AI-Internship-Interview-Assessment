import os
import requests
import json
from datetime import datetime
from twilio.rest import Client
import pywhatkit
from flask import Flask, request, jsonify
import threading
import time
from typing import Dict, List, Optional, Union, Any

class WhatsAppIntegration:
    def __init__(self):
        # Check for Twilio credentials
        self.twilio_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        self.twilio_token = os.environ.get('TWILIO_AUTH_TOKEN')
        self.twilio_phone = os.environ.get('TWILIO_PHONE_NUMBER')
        
        # Initialize Twilio client if credentials are available
        if self.twilio_sid and self.twilio_token:
            self.client = Client(self.twilio_sid, self.twilio_token)
        else:
            self.client = None
            print("Warning: Twilio credentials not found. WhatsApp via Twilio will not be available.")
        
        # Message templates for consistent communication
        self.message_templates = {
            'wait_time': "Jayanagar Clinic Update: Your estimated wait time is {wait_time:.0f} minutes.",
            'queue_position': "Jayanagar Clinic Update: Your current queue position is {queue_position}. Estimated wait time: {wait_time:.0f} minutes.",
            'appointment_reminder': "Reminder: Your appointment with {doctor_name} is scheduled for {appointment_time}. {location_info} Reply '1' to confirm, '2' to reschedule, or '3' to cancel.",
            'prescription_ready': "Your prescription from Dr. {doctor_name} is ready for pickup at {pharmacy_name}. {additional_info}",
            'test_results': "Your test results for {test_name} are now available. Please log in to the patient portal or contact the clinic for details.",
            'follow_up': "This is a follow-up regarding your recent visit with Dr. {doctor_name}. How are you feeling? Reply with a number from 1-5 (1=worse, 3=same, 5=better)."
        }
        
        # Message delivery tracking
        self.message_status_tracker = {}
        
        # Response handler callbacks
        self.response_handlers = {}
        
        # Webhook server status
        self.webhook_server_running = False
    
    def format_phone_for_whatsapp(self, patient_phone: str) -> str:
        """Format phone number for WhatsApp"""
        if not patient_phone.startswith('whatsapp:'):
            return f"whatsapp:{patient_phone}"
        return patient_phone
    
    def get_message_from_template(self, template_name: str, **kwargs) -> str:
        """Get message content from template"""
        if template_name not in self.message_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        try:
            return self.message_templates[template_name].format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter {e} for template '{template_name}'")
    
    def track_message_status(self, message_sid: str, patient_phone: str, template_name: str, data: Dict[str, Any]) -> None:
        """Track message delivery status"""
        self.message_status_tracker[message_sid] = {
            'patient_phone': patient_phone,
            'template_name': template_name,
            'data': data,
            'status': 'sent',
            'sent_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'delivery_timestamp': None,
            'read_timestamp': None
        }
    
    def update_message_status(self, message_sid: str, status: str) -> bool:
        """Update message delivery status"""
        if message_sid not in self.message_status_tracker:
            return False
        
        self.message_status_tracker[message_sid]['status'] = status
        
        if status == 'delivered':
            self.message_status_tracker[message_sid]['delivery_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elif status == 'read':
            self.message_status_tracker[message_sid]['read_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return True
    
    def send_wait_time_update_via_twilio(self, patient_phone, wait_time, queue_position=None):
        """Send wait time update via Twilio WhatsApp"""
        if not self.client:
            raise ValueError("Twilio client not initialized. Check your credentials.")
        
        # Format phone number for WhatsApp
        whatsapp_phone = self.format_phone_for_whatsapp(patient_phone)
        
        # Prepare message content using templates
        template_name = 'queue_position' if queue_position else 'wait_time'
        template_data = {'wait_time': wait_time}
        
        if queue_position:
            template_data['queue_position'] = queue_position
        
        message = self.get_message_from_template(template_name, **template_data)
        
        try:
            # Send WhatsApp message via Twilio
            response = self.client.messages.create(
                from_=f"whatsapp:{self.twilio_phone}",
                body=message,
                to=whatsapp_phone,
                status_callback=f"{os.environ.get('WEBHOOK_BASE_URL', 'https://example.com')}/whatsapp/status"
            )
            
            # Track message status
            self.track_message_status(response.sid, patient_phone, template_name, template_data)
            
            return {
                'status': 'success',
                'message_sid': response.sid,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'channel': 'whatsapp_twilio'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'channel': 'whatsapp_twilio'
            }
    
    def send_wait_time_update_via_pywhatkit(self, patient_phone, wait_time, queue_position=None, scheduled_time=None):
        """Send wait time update via PyWhatKit (alternative method)"""
        # Clean phone number (remove + and any spaces)
        clean_phone = patient_phone.replace('+', '').replace(' ', '')
        
        # Prepare message content using templates
        template_name = 'queue_position' if queue_position else 'wait_time'
        template_data = {'wait_time': wait_time}
        
        if queue_position:
            template_data['queue_position'] = queue_position
        
        message = self.get_message_from_template(template_name, **template_data)
        
        try:
            # If scheduled time is provided, send at that time, otherwise send now
            if scheduled_time:
                hour = scheduled_time.hour
                minute = scheduled_time.minute + 1  # Add 1 minute to ensure it's in the future
            else:
                now = datetime.now()
                hour = now.hour
                minute = now.minute + 1  # Add 1 minute to ensure it's in the future
                
                # Adjust if minute exceeds 59
                if minute > 59:
                    hour = (hour + 1) % 24
                    minute = minute % 60
            
            # Send WhatsApp message via pywhatkit
            pywhatkit.sendwhatmsg(clean_phone, message, hour, minute, wait_time=15, tab_close=True)
            
            # Generate a pseudo message ID for tracking
            message_id = f"pywhatkit_{int(time.time())}_{patient_phone}"
            
            # Track message status (limited tracking for PyWhatKit)
            self.track_message_status(message_id, patient_phone, template_name, template_data)
            
            return {
                'status': 'success',
                'message_sid': message_id,  # Using pseudo message ID
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'channel': 'whatsapp_pywhatkit',
                'scheduled_time': f"{hour}:{minute}"
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'channel': 'whatsapp_pywhatkit'
            }
    
    def register_response_handler(self, message_sid: str, callback_function) -> None:
        """Register a callback function to handle responses to a specific message"""
        self.response_handlers[message_sid] = callback_function
    
    def send_appointment_reminder(self, patient_phone, doctor_name, appointment_time, clinic_location=None, appointment_id=None):
        """Send appointment reminder via WhatsApp"""
        # Format phone number for WhatsApp if using Twilio
        if self.client:
            whatsapp_phone = self.format_phone_for_whatsapp(patient_phone)
                
            # Prepare message content using template
            location_info = f"Please arrive at {clinic_location}." if clinic_location else ""
            template_data = {
                'doctor_name': doctor_name,
                'appointment_time': appointment_time,
                'location_info': location_info
            }
            
            message = self.get_message_from_template('appointment_reminder', **template_data)
            
            try:
                # Send WhatsApp message via Twilio
                response = self.client.messages.create(
                    from_=f"whatsapp:{self.twilio_phone}",
                    body=message,
                    to=whatsapp_phone,
                    status_callback=f"{os.environ.get('WEBHOOK_BASE_URL', 'https://example.com')}/whatsapp/status"
                )
                
                # Track message status
                self.track_message_status(response.sid, patient_phone, 'appointment_reminder', template_data)
                
                # Register a response handler if appointment_id is provided
                if appointment_id:
                    def appointment_response_handler(response_text, patient_phone):
                        response_text = response_text.strip()
                        if response_text == '1':
                            return {'action': 'confirm_appointment', 'appointment_id': appointment_id}
                        elif response_text == '2':
                            return {'action': 'reschedule_appointment', 'appointment_id': appointment_id}
                        elif response_text == '3':
                            return {'action': 'cancel_appointment', 'appointment_id': appointment_id}
                        else:
                            return {'action': 'unknown_response', 'appointment_id': appointment_id, 'response': response_text}
                    
                    self.register_response_handler(response.sid, appointment_response_handler)
                
                return {
                    'status': 'success',
                    'message_sid': response.sid,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'channel': 'whatsapp_twilio'
                }
            except Exception as e:
                # Fall back to pywhatkit if Twilio fails
                return self.send_appointment_reminder_via_pywhatkit(patient_phone, doctor_name, appointment_time, clinic_location, appointment_id)
        else:
            # Use pywhatkit if Twilio is not available
            return self.send_appointment_reminder_via_pywhatkit(patient_phone, doctor_name, appointment_time, clinic_location, appointment_id)
    
    def send_appointment_reminder_via_pywhatkit(self, patient_phone, doctor_name, appointment_time, clinic_location=None, appointment_id=None):
        """Send appointment reminder via PyWhatKit (alternative method)"""
        # Clean phone number (remove + and any spaces)
        clean_phone = patient_phone.replace('+', '').replace(' ', '')
        
        # Prepare message content using template
        location_info = f"Please arrive at {clinic_location}." if clinic_location else ""
        template_data = {
            'doctor_name': doctor_name,
            'appointment_time': appointment_time,
            'location_info': location_info
        }
        
        message = self.get_message_from_template('appointment_reminder', **template_data)
        
        try:
            now = datetime.now()
            hour = now.hour
            minute = now.minute + 1  # Add 1 minute to ensure it's in the future
            
            # Adjust if minute exceeds 59
            if minute > 59:
                hour = (hour + 1) % 24
                minute = minute % 60
            
            # Send WhatsApp message via pywhatkit
            pywhatkit.sendwhatmsg(clean_phone, message, hour, minute, wait_time=15, tab_close=True)
            
            # Generate a pseudo message ID for tracking
            message_id = f"pywhatkit_{int(time.time())}_{patient_phone}"
            
            # Track message status (limited tracking for PyWhatKit)
            self.track_message_status(message_id, patient_phone, 'appointment_reminder', template_data)
            
            # Register a response handler if appointment_id is provided
            if appointment_id:
                def appointment_response_handler(response_text, patient_phone):
                    response_text = response_text.strip()
                    if response_text == '1':
                        return {'action': 'confirm_appointment', 'appointment_id': appointment_id}
                    elif response_text == '2':
                        return {'action': 'reschedule_appointment', 'appointment_id': appointment_id}
                    elif response_text == '3':
                        return {'action': 'cancel_appointment', 'appointment_id': appointment_id}
                    else:
                        return {'action': 'unknown_response', 'appointment_id': appointment_id, 'response': response_text}
                
                self.register_response_handler(message_id, appointment_response_handler)
            
            return {
                'status': 'success',
                'message_sid': message_id,  # Using pseudo message ID
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'channel': 'whatsapp_pywhatkit',
                'scheduled_time': f"{hour}:{minute}"
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'channel': 'whatsapp_pywhatkit'
            }
    
    def send_queue_position_notification(self, patient_phone, queue_position, wait_time, doctor_name=None):
        """Send queue position update via WhatsApp"""
        # Determine if we should use Twilio or PyWhatKit
        if self.client:
            return self.send_wait_time_update_via_twilio(patient_phone, wait_time, queue_position)
        else:
            return self.send_wait_time_update_via_pywhatkit(patient_phone, wait_time, queue_position)
    
    def start_webhook_server(self, host='0.0.0.0', port=5001):
        """Start a Flask webhook server to handle incoming WhatsApp messages and status updates"""
        if self.webhook_server_running:
            print("Webhook server is already running")
            return False
        
        app = Flask('whatsapp_webhook')
        
        @app.route('/whatsapp/status', methods=['POST'])
        def handle_status_update():
            """Handle message status updates from Twilio"""
            message_sid = request.values.get('MessageSid')
            message_status = request.values.get('MessageStatus')
            
            if not message_sid or not message_status:
                return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
            
            # Update message status in tracker
            success = self.update_message_status(message_sid, message_status)
            
            if success:
                return jsonify({'status': 'success', 'message': f'Status updated to {message_status}'})
            else:
                return jsonify({'status': 'error', 'message': 'Message not found in tracker'}), 404
        
        @app.route('/whatsapp/webhook', methods=['POST'])
        def handle_incoming_message():
            """Handle incoming WhatsApp messages"""
            # Extract message details from the request
            from_number = request.values.get('From')
            body = request.values.get('Body')
            message_sid = request.values.get('MessageSid')
            
            if not from_number or not body:
                return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
            
            # Clean the phone number for internal use
            patient_phone = from_number
            if patient_phone.startswith('whatsapp:'):
                patient_phone = patient_phone[9:]  # Remove 'whatsapp:' prefix
            
            # Check if this is a response to a message we're tracking
            # We need to find which message this is responding to
            response_processed = False
            
            # Look for the most recent message sent to this number
            matching_sids = []
            for sid, data in self.message_status_tracker.items():
                if data['patient_phone'] == patient_phone:
                    matching_sids.append((sid, data['sent_timestamp']))
            
            # Sort by timestamp (most recent first)
            matching_sids.sort(key=lambda x: x[1], reverse=True)
            
            # Process the response with the appropriate handler if available
            if matching_sids:
                most_recent_sid = matching_sids[0][0]
                if most_recent_sid in self.response_handlers:
                    handler = self.response_handlers[most_recent_sid]
                    result = handler(body, patient_phone)
                    response_processed = True
                    
                    # Log the response
                    print(f"Processed response from {patient_phone}: {body} -> {result}")
            
            if response_processed:
                return jsonify({'status': 'success', 'message': 'Response processed'})
            else:
                # Handle as a new message if not a response
                print(f"Received message from {patient_phone}: {body}")
                return jsonify({'status': 'success', 'message': 'Message received'})
        
        def run_flask_app():
            app.run(host=host, port=port, debug=False, use_reloader=False)
        
        # Start the Flask app in a separate thread
        thread = threading.Thread(target=run_flask_app)
        thread.daemon = True  # Thread will exit when the main program exits
        thread.start()
        
        self.webhook_server_running = True
        print(f"Webhook server started at http://{host}:{port}")
        return True
    
    def stop_webhook_server(self):
        """Stop the webhook server if it's running"""
        if not self.webhook_server_running:
            print("Webhook server is not running")
            return False
        
        # In a real implementation, we would need to properly shut down the Flask server
        # This is a simplified version that just changes the flag
        self.webhook_server_running = False
        print("Webhook server stopped")
        return True
    
    def get_message_status_analytics(self):
        """Get analytics on message delivery and read status"""
        total_messages = len(self.message_status_tracker)
        if total_messages == 0:
            return {
                'total_messages': 0,
                'delivery_rate': 0,
                'read_rate': 0,
                'average_delivery_time': 0,
                'status_counts': {}
            }
        
        status_counts = {}
        delivered_count = 0
        read_count = 0
        delivery_times = []
        
        for message_id, data in self.message_status_tracker.items():
            status = data['status']
            
            # Count by status
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
            
            # Count delivered and read messages
            if status in ['delivered', 'read']:
                delivered_count += 1
            if status == 'read':
                read_count += 1
            
            # Calculate delivery time if available
            if data['sent_timestamp'] and data['delivery_timestamp']:
                sent_time = datetime.strptime(data['sent_timestamp'], '%Y-%m-%d %H:%M:%S')
                delivery_time = datetime.strptime(data['delivery_timestamp'], '%Y-%m-%d %H:%M:%S')
                delivery_seconds = (delivery_time - sent_time).total_seconds()
                delivery_times.append(delivery_seconds)
        
        # Calculate rates and averages
        delivery_rate = delivered_count / total_messages if total_messages > 0 else 0
        read_rate = read_count / total_messages if total_messages > 0 else 0
        avg_delivery_time = sum(delivery_times) / len(delivery_times) if delivery_times else 0
        
        return {
            'total_messages': total_messages,
            'delivery_rate': delivery_rate,
            'read_rate': read_rate,
            'average_delivery_time': avg_delivery_time,
            'status_counts': status_counts
        }

# Example usage
if __name__ == "__main__":
    # Initialize WhatsApp integration
    whatsapp = WhatsAppIntegration()
    
    # Example: Send wait time update
    result = whatsapp.send_queue_position_notification(
        patient_phone="+1234567890",  # Replace with actual phone number
        queue_position=3,
        wait_time=25,
        doctor_name="Dr. Sharma"
    )
    
    print(f"Notification result: {result['status']}")