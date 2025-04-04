import os
import requests
import json
from datetime import datetime, timedelta
from twilio.rest import Client
import pywhatkit
from flask import Flask, request, jsonify
import threading
import time
import random
from typing import Dict, List, Optional, Union, Any
from whatsapp_integration import WhatsAppIntegration

class EnhancedWhatsAppIntegration(WhatsAppIntegration):
    """Enhanced WhatsApp integration with automated appointment reminders and feedback collection"""
    
    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        # Additional message templates for enhanced features
        self.message_templates.update({
            'appointment_confirmation': "Your appointment with Dr. {doctor_name} has been confirmed for {appointment_time}. We look forward to seeing you at {clinic_name}.",
            'appointment_reminder_day_before': "Reminder: Your appointment with Dr. {doctor_name} is tomorrow at {appointment_time}. Please arrive 15 minutes early to complete any paperwork. Reply '1' to confirm, '2' to reschedule, or '3' to cancel.",
            'appointment_reminder_hour_before': "Your appointment with Dr. {doctor_name} is in 1 hour at {appointment_time}. Current estimated wait time is {wait_time} minutes. Reply 'W' for wait time updates.",
            'feedback_request': "Thank you for visiting Dr. {doctor_name} today. We value your feedback. On a scale of 1-5 (5 being excellent), how would you rate your experience? Reply with a number.",
            'feedback_followup': "Thank you for your rating of {rating}/5. {followup_message} Is there anything specific you'd like to share about your experience? Reply with your comments or 'N' to skip.",
            'prescription_reminder': "Reminder: Please take your {medication_name} {dosage} as prescribed by Dr. {doctor_name}. Reply '1' to confirm you've taken it or '2' if you have questions.",
            'test_reminder': "Reminder: Your {test_name} is scheduled for {test_date} at {test_location}. Please remember to {preparation_instructions}. Reply '1' to confirm."
        })
        
        # Feedback followup messages based on rating
        self.feedback_followup_messages = {
            1: "We're sorry to hear about your experience. We take your feedback seriously and will work to improve.",
            2: "We apologize that your experience wasn't better. We appreciate your feedback and will address your concerns.",
            3: "Thank you for your feedback. We're constantly working to improve our services.",
            4: "We're glad you had a good experience. Thank you for your feedback.",
            5: "We're thrilled you had an excellent experience! Thank you for your positive feedback."
        }
        
        # Scheduled message queue
        self.scheduled_messages = []
        
        # Feedback data storage
        self.patient_feedback = {}
        
        # Start scheduler thread
        self.scheduler_running = False
        self.start_scheduler()
    
    def start_scheduler(self):
        """Start the scheduler thread for sending scheduled messages"""
        if not self.scheduler_running:
            self.scheduler_running = True
            scheduler_thread = threading.Thread(target=self._scheduler_loop)
            scheduler_thread.daemon = True
            scheduler_thread.start()
    
    def _scheduler_loop(self):
        """Background thread to process scheduled messages"""
        while self.scheduler_running:
            now = datetime.now()
            
            # Check for messages that need to be sent
            messages_to_send = [msg for msg in self.scheduled_messages 
                               if msg['scheduled_time'] <= now and not msg['sent']]
            
            for message in messages_to_send:
                try:
                    # Send the message
                    if message['channel'] == 'twilio':
                        result = self.send_wait_time_update_via_twilio(
                            message['patient_phone'],
                            message.get('wait_time', 0),
                            message.get('queue_position', None)
                        )
                    elif message['channel'] == 'pywhatkit':
                        result = self.send_wait_time_update_via_pywhatkit(
                            message['patient_phone'],
                            message.get('wait_time', 0),
                            message.get('queue_position', None)
                        )
                    else:
                        # Use custom message template
                        result = self.send_custom_message(
                            message['patient_phone'],
                            message['template_name'],
                            **message['template_data']
                        )
                    
                    # Mark as sent
                    message['sent'] = True
                    message['result'] = result
                    
                    # Register callback for response if needed
                    if message.get('response_handler'):
                        if result.get('message_sid'):
                            self.register_response_handler(
                                result['message_sid'],
                                message['response_handler']
                            )
                except Exception as e:
                    message['error'] = str(e)
                    message['sent'] = True  # Mark as sent to avoid retrying indefinitely
            
            # Clean up old messages (keep for 24 hours for reporting)
            cutoff_time = now - timedelta(hours=24)
            self.scheduled_messages = [
                msg for msg in self.scheduled_messages 
                if not msg['sent'] or msg.get('scheduled_time', now) > cutoff_time
            ]
            
            # Sleep for a short time
            time.sleep(10)
    
    def schedule_message(self, patient_phone, scheduled_time, template_name, template_data, 
                        channel='twilio', response_handler=None):
        """Schedule a message to be sent at a specific time"""
        message = {
            'patient_phone': patient_phone,
            'scheduled_time': scheduled_time,
            'template_name': template_name,
            'template_data': template_data,
            'channel': channel,
            'sent': False,
            'created_at': datetime.now(),
            'response_handler': response_handler
        }
        
        self.scheduled_messages.append(message)
        return {
            'status': 'scheduled',
            'message_id': f"scheduled_{len(self.scheduled_messages)}",
            'scheduled_time': scheduled_time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def send_custom_message(self, patient_phone, template_name, **kwargs):
        """Send a custom message using a template"""
        if not self.client:
            raise ValueError("Twilio client not initialized. Check your credentials.")
        
        # Format phone number for WhatsApp
        whatsapp_phone = self.format_phone_for_whatsapp(patient_phone)
        
        # Get message from template
        message = self.get_message_from_template(template_name, **kwargs)
        
        try:
            # Send WhatsApp message via Twilio
            response = self.client.messages.create(
                from_=f"whatsapp:{self.twilio_phone}",
                body=message,
                to=whatsapp_phone,
                status_callback=f"{os.environ.get('WEBHOOK_BASE_URL', 'https://example.com')}/whatsapp/status"
            )
            
            # Track message status
            self.track_message_status(response.sid, patient_phone, template_name, kwargs)
            
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
    
    def schedule_appointment_reminders(self, patient_phone, doctor_name, appointment_time, 
                                     clinic_name, appointment_id=None):
        """Schedule a series of appointment reminders"""
        # Convert appointment_time to datetime if it's a string
        if isinstance(appointment_time, str):
            appointment_time = datetime.strptime(appointment_time, '%Y-%m-%d %H:%M:%S')
        
        # Template data
        template_data = {
            'doctor_name': doctor_name,
            'appointment_time': appointment_time.strftime('%I:%M %p on %A, %B %d'),
            'clinic_name': clinic_name
        }
        
        # Schedule day-before reminder (24 hours before)
        day_before = appointment_time - timedelta(days=1)
        day_before = day_before.replace(hour=10, minute=0)  # Send at 10 AM
        
        self.schedule_message(
            patient_phone,
            day_before,
            'appointment_reminder_day_before',
            template_data,
            response_handler=self._handle_appointment_confirmation
        )
        
        # Schedule hour-before reminder
        hour_before = appointment_time - timedelta(hours=1)
        
        # Add estimated wait time to hour-before reminder
        template_data_with_wait = template_data.copy()
        template_data_with_wait['wait_time'] = '15-20'  # This would be dynamically calculated in production
        
        self.schedule_message(
            patient_phone,
            hour_before,
            'appointment_reminder_hour_before',
            template_data_with_wait
        )
        
        return {
            'status': 'scheduled',
            'appointment_id': appointment_id or f"appt_{int(time.time())}",
            'reminders': [
                {'type': 'day_before', 'scheduled_time': day_before.strftime('%Y-%m-%d %H:%M:%S')},
                {'type': 'hour_before', 'scheduled_time': hour_before.strftime('%Y-%m-%d %H:%M:%S')}
            ]
        }
    
    def _handle_appointment_confirmation(self, message_sid, from_number, body):
        """Handle responses to appointment confirmation messages"""
        response = body.strip().lower()
        
        if response == '1':
            # Confirmed
            reply = "Thank you for confirming your appointment. We look forward to seeing you."
        elif response == '2':
            # Reschedule
            reply = "We'll help you reschedule your appointment. Please call our office at (080) 2222-3333 during business hours."
        elif response == '3':
            # Cancel
            reply = "We've received your cancellation request. If this was a mistake, please call our office at (080) 2222-3333."
        else:
            # Invalid response
            reply = "We didn't recognize your response. Please reply with '1' to confirm, '2' to reschedule, or '3' to cancel your appointment."
        
        # Send reply
        self.client.messages.create(
            from_=f"whatsapp:{self.twilio_phone}",
            body=reply,
            to=from_number
        )
        
        return {
            'status': 'processed',
            'response': response,
            'action': ['confirmed', 'reschedule', 'cancelled', 'invalid'][int(response) if response in ['1', '2', '3'] else 3]
        }
    
    def schedule_feedback_request(self, patient_phone, doctor_name, delay=timedelta(hours=2)):
        """Schedule a feedback request after appointment"""
        # Schedule feedback request
        scheduled_time = datetime.now() + delay
        
        template_data = {
            'doctor_name': doctor_name
        }
        
        self.schedule_message(
            patient_phone,
            scheduled_time,
            'feedback_request',
            template_data,
            response_handler=self._handle_feedback_response
        )
        
        return {
            'status': 'scheduled',
            'feedback_id': f"feedback_{int(time.time())}",
            'scheduled_time': scheduled_time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _handle_feedback_response(self, message_sid, from_number, body):
        """Handle responses to feedback request messages"""
        try:
            rating = int(body.strip())
            if 1 <= rating <= 5:
                # Valid rating
                patient_id = self._get_patient_id_from_phone(from_number)
                
                # Store feedback
                self.patient_feedback[patient_id] = {
                    'rating': rating,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'comments': None
                }
                
                # Send followup message
                followup_message = self.feedback_followup_messages.get(rating, "Thank you for your feedback.")
                
                template_data = {
                    'rating': rating,
                    'followup_message': followup_message
                }
                
                # Send immediately
                self.send_custom_message(
                    from_number,
                    'feedback_followup',
                    **template_data
                )
                
                return {
                    'status': 'processed',
                    'rating': rating,
                    'patient_id': patient_id
                }
            else:
                # Invalid rating
                reply = "Please rate your experience on a scale of 1-5, with 5 being excellent."
                self.client.messages.create(
                    from_=f"whatsapp:{self.twilio_phone}",
                    body=reply,
                    to=from_number
                )
                return {'status': 'invalid_rating'}
        except ValueError:
            # Not a number
            reply = "Please rate your experience on a scale of 1-5, with 5 being excellent."
            self.client.messages.create(
                from_=f"whatsapp:{self.twilio_phone}",
                body=reply,
                to=from_number
            )
            return {'status': 'invalid_format'}
    
    def _get_patient_id_from_phone(self, phone_number):
        """Get patient ID from phone number (in a real system, this would query a database)"""
        # For demo purposes, generate a consistent ID based on phone number
        return f"P{abs(hash(phone_number)) % 10000}"
    
    def schedule_prescription_reminder(self, patient_phone, medication_name, dosage, doctor_name, 
                                     schedule, duration_days):
        """Schedule medication reminders"""
        # Parse schedule (e.g., "every 8 hours", "twice daily")
        times_per_day = 1
        if "twice" in schedule.lower() or "2 times" in schedule.lower():
            times_per_day = 2
        elif "three times" in schedule.lower() or "3 times" in schedule.lower():
            times_per_day = 3
        elif "four times" in schedule.lower() or "4 times" in schedule.lower():
            times_per_day = 4
        elif "every 8 hours" in schedule.lower():
            times_per_day = 3
        elif "every 6 hours" in schedule.lower():
            times_per_day = 4
        elif "every 12 hours" in schedule.lower():
            times_per_day = 2
        
        # Generate reminder times
        reminders = []
        start_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)  # Start at 8 AM today
        
        for day in range(duration_days):
            for i in range(times_per_day):
                # Distribute reminders throughout the day
                hour_offset = (16 // times_per_day) * i
                reminder_time = start_date + timedelta(days=day, hours=hour_offset)
                
                # Skip reminders in the past
                if reminder_time < datetime.now():
                    continue
                
                template_data = {
                    'medication_name': medication_name,
                    'dosage': dosage,
                    'doctor_name': doctor_name
                }
                
                # Schedule reminder
                result = self.schedule_message(
                    patient_phone,
                    reminder_time,
                    'prescription_reminder',
                    template_data,
                    response_handler=self._handle_medication_confirmation
                )
                
                reminders.append({
                    'scheduled_time': reminder_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'message_id': result['message_id']
                })
        
        return {
            'status': 'scheduled',
            'medication': medication_name,
            'dosage': dosage,
            'schedule': schedule,
            'duration_days': duration_days,
            'reminders': reminders
        }
    
    def _handle_medication_confirmation(self, message_sid, from_number, body):
        """Handle responses to medication reminders"""
        response = body.strip().lower()
        
        if response == '1':
            # Confirmed
            reply = "Thank you for confirming. Stay healthy!"
        elif response == '2':
            # Questions
            reply = "If you have questions about your medication, please call our pharmacy at (080) 3333-4444 during business hours."
        else:
            # Invalid response
            reply = "We didn't recognize your response. Please reply with '1' to confirm you've taken your medication or '2' if you have questions."
        
        # Send reply
        self.client.messages.create(
            from_=f"whatsapp:{self.twilio_phone}",
            body=reply,
            to=from_number
        )
        
        return {
            'status': 'processed',
            'response': response,
            'action': 'confirmed' if response == '1' else 'questions' if response == '2' else 'invalid'
        }
    
    def get_feedback_analytics(self):
        """Get analytics on patient feedback"""
        if not self.patient_feedback:
            return {
                'average_rating': 0,
                'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'total_feedback': 0
            }
        
        # Calculate average rating
        ratings = [feedback['rating'] for feedback in self.patient_feedback.values()]
        avg_rating = sum(ratings) / len(ratings)
        
        # Calculate rating distribution
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            distribution[rating] += 1
        
        return {
            'average_rating': round(avg_rating, 2),
            'rating_distribution': distribution,
            'total_feedback': len(ratings)
        }

# Example usage
if __name__ == "__main__":
    # Initialize enhanced WhatsApp integration
    enhanced_whatsapp = EnhancedWhatsAppIntegration()
    
    # Example: Schedule appointment reminders
    result = enhanced_whatsapp.schedule_appointment_reminders(
        "+919876543210",
        "Dr. Sharma",
        datetime.now() + timedelta(days=2, hours=3),
        "Jayanagar Specialty Clinic"
    )
    
    print("Scheduled appointment reminders:", result)
    
    # Example: Schedule feedback request
    result = enhanced_whatsapp.schedule_feedback_request(
        "+919876543210",
        "Dr. Sharma",
        delay=timedelta(minutes=1)  # For testing, normally would be hours
    )
    
    print("Scheduled feedback request:", result)
    
    # Keep the script running to allow scheduler to process
    print("Running scheduler (press Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped")