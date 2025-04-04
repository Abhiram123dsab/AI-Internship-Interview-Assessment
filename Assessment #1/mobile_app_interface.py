import os
import json
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from datetime import datetime, timedelta
import random
from enhanced_clinic_system import predict_wait_time_enhanced
from advanced_ml_models import predict_wait_time_advanced
from enhanced_whatsapp_integration import EnhancedWhatsAppIntegration

class MobileAppInterface:
    """Mobile app interface for patients to access clinic services"""
    
    def __init__(self, app=None, socketio=None):
        # Initialize with Flask app if provided
        self.app = app or Flask(__name__)
        self.socketio = socketio or SocketIO(self.app)
        
        # Initialize WhatsApp integration
        self.whatsapp = EnhancedWhatsAppIntegration()
        
        # Register routes
        self.register_routes()
        
        # Register socket events
        self.register_socket_events()
    
    def register_routes(self):
        """Register API routes for mobile app"""
        # Mobile app home page
        @self.app.route('/mobile')
        def mobile_home():
            return render_template('mobile_app.html')
        
        # API endpoint for patient queue status
        @self.app.route('/api/mobile/queue/<int:patient_id>')
        def get_mobile_queue(patient_id):
            # In a real app, this would query a database
            # For demo, we'll generate sample data
            position = random.randint(1, 5)
            wait_time = position * 8 + random.randint(-5, 5)
            
            return jsonify({
                'patient_id': patient_id,
                'position': position,
                'wait_time': wait_time,
                'status': 'In Queue' if position > 1 else 'Ready',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'doctor_name': f"Dr. {random.choice(['Sharma', 'Patel', 'Reddy', 'Kumar', 'Singh'])}",
                'appointment_time': (datetime.now() + timedelta(minutes=wait_time)).strftime('%I:%M %p')
            })
        
        # API endpoint for upcoming appointments
        @self.app.route('/api/mobile/appointments/<int:patient_id>')
        def get_mobile_appointments(patient_id):
            # In a real app, this would query a database
            # For demo, we'll generate sample data
            appointments = []
            
            # Generate 1-3 upcoming appointments
            for i in range(random.randint(1, 3)):
                appointment_date = datetime.now() + timedelta(days=random.randint(1, 14))
                appointment_time = appointment_date.replace(
                    hour=random.randint(9, 20),
                    minute=random.choice([0, 15, 30, 45])
                )
                
                appointments.append({
                    'appointment_id': f"A{random.randint(1000, 9999)}",
                    'doctor_name': f"Dr. {random.choice(['Sharma', 'Patel', 'Reddy', 'Kumar', 'Singh'])}",
                    'specialty': random.choice(['Cardiology', 'Orthopedics', 'Pediatrics', 'Dermatology', 'General Medicine']),
                    'date': appointment_time.strftime('%Y-%m-%d'),
                    'time': appointment_time.strftime('%I:%M %p'),
                    'clinic_name': 'Jayanagar Specialty Clinic',
                    'status': random.choice(['Confirmed', 'Pending', 'Rescheduled'])
                })
            
            return jsonify({
                'patient_id': patient_id,
                'appointments': appointments,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # API endpoint for medical records
        @self.app.route('/api/mobile/records/<int:patient_id>')
        def get_mobile_records(patient_id):
            # In a real app, this would query a database with proper authentication
            # For demo, we'll generate sample data
            records = []
            
            # Generate 2-5 past visits
            for i in range(random.randint(2, 5)):
                visit_date = datetime.now() - timedelta(days=random.randint(10, 365))
                
                records.append({
                    'visit_id': f"V{random.randint(1000, 9999)}",
                    'date': visit_date.strftime('%Y-%m-%d'),
                    'doctor_name': f"Dr. {random.choice(['Sharma', 'Patel', 'Reddy', 'Kumar', 'Singh'])}",
                    'diagnosis': random.choice(['Regular checkup', 'Fever', 'Allergies', 'Hypertension', 'Diabetes follow-up']),
                    'prescription': bool(random.randint(0, 1)),
                    'lab_results': bool(random.randint(0, 1)),
                    'follow_up': bool(random.randint(0, 1))
                })
            
            return jsonify({
                'patient_id': patient_id,
                'records': sorted(records, key=lambda x: x['date'], reverse=True),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # API endpoint for notifications
        @self.app.route('/api/mobile/notifications/<int:patient_id>')
        def get_mobile_notifications(patient_id):
            # In a real app, this would query a database
            # For demo, we'll generate sample data
            notifications = []
            
            # Generate 3-7 notifications
            notification_types = [
                'appointment_reminder',
                'wait_time_update',
                'prescription_ready',
                'lab_results_ready',
                'feedback_request',
                'doctor_message'
            ]
            
            for i in range(random.randint(3, 7)):
                notification_time = datetime.now() - timedelta(hours=random.randint(1, 72))
                notification_type = random.choice(notification_types)
                
                if notification_type == 'appointment_reminder':
                    message = "Reminder: You have an appointment tomorrow at 10:30 AM with Dr. Sharma."
                elif notification_type == 'wait_time_update':
                    message = "Your estimated wait time has been updated to 15 minutes."
                elif notification_type == 'prescription_ready':
                    message = "Your prescription is ready for pickup at the pharmacy."
                elif notification_type == 'lab_results_ready':
                    message = "Your lab results are now available. Please check the app."
                elif notification_type == 'feedback_request':
                    message = "Please rate your recent visit with Dr. Patel."
                else:  # doctor_message
                    message = "Dr. Kumar has sent you a message regarding your treatment plan."
                
                notifications.append({
                    'notification_id': f"N{random.randint(1000, 9999)}",
                    'type': notification_type,
                    'message': message,
                    'time': notification_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'read': bool(random.randint(0, 1))
                })
            
            return jsonify({
                'patient_id': patient_id,
                'notifications': sorted(notifications, key=lambda x: x['time'], reverse=True),
                'unread_count': sum(1 for n in notifications if not n['read']),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # API endpoint for voice assistant integration
        @self.app.route('/api/mobile/voice-query', methods=['POST'])
        def process_voice_query():
            data = request.json
            query = data.get('query', '')
            patient_id = data.get('patient_id')
            
            # Process natural language query
            # In a real app, this would use NLP to understand the query
            response = self._process_voice_query(query, patient_id)
            
            return jsonify(response)
        
        # API endpoint to enable push notifications
        @self.app.route('/api/mobile/register-device', methods=['POST'])
        def register_device():
            data = request.json
            patient_id = data.get('patient_id')
            device_token = data.get('device_token')
            platform = data.get('platform', 'android')  # or 'ios'
            
            # In a real app, this would store the device token in a database
            # For demo, we'll just return success
            return jsonify({
                'status': 'success',
                'message': f"Device registered for patient {patient_id}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    def register_socket_events(self):
        """Register socket events for real-time updates"""
        @self.socketio.on('connect', namespace='/mobile')
        def handle_mobile_connect():
            print('Mobile client connected')
        
        @self.socketio.on('disconnect', namespace='/mobile')
        def handle_mobile_disconnect():
            print('Mobile client disconnected')
        
        @self.socketio.on('join_patient_room', namespace='/mobile')
        def handle_join_room(data):
            patient_id = data.get('patient_id')
            if patient_id:
                # In a real app, this would join a room specific to the patient
                print(f"Patient {patient_id} joined their room")
        
        @self.socketio.on('request_wait_time_update', namespace='/mobile')
        def handle_wait_time_request(data):
            patient_id = data.get('patient_id')
            if patient_id:
                # In a real app, this would fetch the latest wait time
                position = random.randint(1, 5)
                wait_time = position * 8 + random.randint(-5, 5)
                
                self.socketio.emit('wait_time_update', {
                    'patient_id': patient_id,
                    'position': position,
                    'wait_time': wait_time,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, namespace='/mobile', room=request.sid)
    
    def _process_voice_query(self, query, patient_id):
        """Process natural language voice query"""
        query = query.lower()
        
        # Simple keyword matching for demo
        if 'wait' in query or 'time' in query or 'how long' in query:
            # Query about wait time
            position = random.randint(1, 5)
            wait_time = position * 8 + random.randint(-5, 5)
            
            return {
                'type': 'wait_time',
                'message': f"Your current wait time is approximately {wait_time} minutes. You are position {position} in the queue.",
                'data': {
                    'wait_time': wait_time,
                    'position': position
                }
            }
        elif 'appointment' in query or 'schedule' in query or 'when' in query:
            # Query about appointments
            return {
                'type': 'appointment',
                'message': "Your next appointment is scheduled for tomorrow at 10:30 AM with Dr. Sharma.",
                'data': {
                    'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'time': '10:30 AM',
                    'doctor': 'Dr. Sharma'
                }
            }
        elif 'prescription' in query or 'medicine' in query or 'medication' in query:
            # Query about prescriptions
            return {
                'type': 'prescription',
                'message': "Your prescription for Amoxicillin was issued yesterday and is ready for pickup at the pharmacy.",
                'data': {
                    'medication': 'Amoxicillin',
                    'status': 'ready',
                    'pharmacy': 'Clinic Pharmacy'
                }
            }
        elif 'doctor' in query or 'physician' in query:
            # Query about doctor
            return {
                'type': 'doctor',
                'message': "You are scheduled to see Dr. Sharma, who is a cardiologist with 15 years of experience.",
                'data': {
                    'doctor': 'Dr. Sharma',
                    'specialty': 'Cardiology',
                    'experience': '15 years'
                }
            }
        else:
            # General query
            return {
                'type': 'general',
                'message': "I'm sorry, I couldn't understand your query. You can ask about wait times, appointments, prescriptions, or your doctor.",
                'data': {}
            }
    
    def send_push_notification(self, patient_id, title, message, data=None):
        """Send push notification to patient's mobile device"""
        # In a real app, this would use Firebase Cloud Messaging or Apple Push Notification Service
        # For demo, we'll just print the notification
        print(f"Push notification to patient {patient_id}: {title} - {message}")
        
        # Emit socket event for real-time update
        self.socketio.emit('push_notification', {
            'patient_id': patient_id,
            'title': title,
            'message': message,
            'data': data or {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, namespace='/mobile')
        
        return {
            'status': 'sent',
            'patient_id': patient_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the mobile app interface server"""
        self.socketio.run(self.app, host=host, port=port, debug=debug)

# Create mobile app template
def create_mobile_app_template(app):
    """Create HTML template for mobile app interface"""
    os.makedirs('templates', exist_ok=True)
    
    mobile_app_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Clinic Mobile App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <script src="https://cdn.socket.io/4.0.1/socket.io.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding-bottom: 70px; /* Space for bottom nav */
        }
        
        .app-header {
            background-color: #0d6efd;
            color: white;
            padding: 15px;
            text-align: center;
            position: relative;
        }
        
        .app-title {
            margin: 0;
            font-size: 1.2rem;
        }
        
        .notification-badge {
            position: absolute;
            top: 10px;
            right: 15px;
            background-color: #dc3545;
            color: white;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
        }
        
        .queue-card {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .queue-position {
            font-size: 3rem;
            font-weight: bold;
            color: #0d6efd;
            text-align: center;
            margin: 10px 0;
        }
        
        .wait-time {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
        
        .status-badge {
            font-size: 1rem;
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
        }
        
        .status-in-queue {
            background-color: #ffc107;
            color: #212529;
        }
        
        .status-ready {
            background-color: #198754;
            color: white;
        }
        
        .appointment-card {
            background-color: white;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .appointment-date {
            font-size: 0.9rem;
            color: #6c757d;
        }
        
        .appointment-doctor {
            font-weight: bold;
            margin: 5px 0;
        }
        
        .appointment-specialty {
            font-size: 0.9rem;
            color: #0d6efd;
        }
        
        .bottom-nav {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            display: flex;
            justify-content: space-around;
            padding: 10px 0;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
        
        .nav-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #6c757d;
            text-decoration: none;
            font-size: 0.8rem;
        }
        
        .nav-item.active {
            color: #0d6efd;
        }
        
        .nav-icon {
            font-size: 1.5rem;
            margin-bottom: 2px;
        }
        
        .section {
            display: none;
            padding: 15px;
        }
        
        .section.active {
            display: block;
        }
        
        .voice-button {
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: #0d6efd;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            border: none;
            font-size: 1.5rem;
        }
        
        .voice-button:active {
            background-color: #0b5ed7;
        }
        
        .voice-response {
            background-color: #e9ecef;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            display: none;
        }
        
        .notification-item {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            position: relative;
        }
        
        .notification-unread {
            border-left: 4px solid #0d6efd;
        }
        
        .notification-time {
            font-size: 0.8rem;
            color: #6c757d;
        }
        
        .notification-message {
            margin-top: 5px;
        }
        
        .record-item {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .record-date {
            font-size: 0.8rem;
            color: #6c757d;
        }
        
        .record-doctor {
            font-weight: bold;
            margin: 5px 0;
        }
        
        .record-diagnosis {
            margin-top: 5px;
        }
        
        .record-badges {
            margin-top: 10px;
        }
        
        .record-badge {
            font-size: 0.8rem;
            padding: 3px 8px;
            border-radius: 10px;
            display: inline-block;
            margin-right: 5px;
            background-color: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="app-header">
        <h1 class="app-title">Jayanagar Clinic</h1>
        <div class="notification-badge" id="notification-badge">0</div>
    </div>
    
    <!-- Queue Section -->
    <div class="section active" id="queue-section">
        <div class="queue-card">
            <h2 class="text-center mb-3">Your Queue Status</h2>
            <div class="text-center mb-3">
                <span class="status-badge" id="queue-status">In Queue</span>
            </div>
            <div class="queue-position" id="queue-position">3</div>
            <p class="text-center">Your Position</p>
            <div class="wait-time" id="wait-time">25</div>
            <p class="text-center">Minutes Estimated Wait</p>
            <div class="d-flex justify-content-between align-items-center mt-4">
                <div>
                    <p class="mb-1">Doctor: <span id="doctor-name">Dr. Sharma</span></p>
                    <p class="mb-0">Appointment: <span id="appointment-time">3:30 PM</span></p>
                </div>
                <button class="btn btn-primary" id="refresh-queue">Refresh</button>
            </div>
        </div>
        
        <div class="voice-response" id="voice-response">
            <p id="voice-response-text"></p>
        </div>
    </div>
    
    <!-- Appointments Section -->
    <div class="section" id="appointments-section">
        <h2 class="mb-3">Your Appointments</h2>
        <div id="appointments-container">
            <!-- Appointments will be loaded here -->
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Records Section -->
    <div class="section" id="records-section">
        <h2 class="mb-3">Medical Records</h2>
        <div id="records-container">
            <!-- Records will be loaded here -->
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Notifications Section -->
    <div class="section" id="notifications-section">
        <h2 class="mb-3">Notifications</h2>
        <div id="notifications-container">
            <!-- Notifications will be loaded here -->
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Voice Assistant Button -->
    <button class="voice-button" id="voice-button">
        <i class="bi bi-mic-fill"></i>
    </button>
    
    <!-- Bottom Navigation -->
    <div class="bottom-nav">
        <a href="#" class="nav-item active" data-section="queue-section">
            <i class="bi bi-clock nav-icon"></i>
            <span>Queue</span>
        </a>
        <a href="#" class="nav-item" data-section="appointments-section">
            <i class="bi bi-calendar-check nav-icon"></i>
            <span>Appointments</span>
        </a>
        <a href="#" class="nav-item" data-section="records-section">
            <i class="bi bi-file-medical nav-icon"></i>
            <span>Records</span>
        </a>
        <a href="#" class="nav-item" data-section="notifications-section">
            <i class="bi bi-bell nav-icon"></i>
            <span>Notifications</span>
        </a>
    </div>
    
    <script>
        // Simulate patient ID (would come from login in a real app)
        const patientId = 1234;
        
        // Initialize Socket.IO
        const socket = io('/mobile');
        
        // Join patient room
        socket.emit('join_patient_room', { patient_id: patientId });
        
        // Listen for wait time updates
        socket.on('wait_time_update', function(data) {
            if (data.patient_id === patientId) {
                updateQueueInfo(data);
            }
        });
        
        // Listen for push notifications
        socket.on('push_notification', function(data) {
            if (data.patient_id === patientId) {
                // Update notification badge
                const badge = document.getElementById('notification-badge');
                badge.textContent = parseInt(badge.textContent) + 1;
                badge.classList.remove('d-none');
                
                // Show notification
                if ('Notification' in window && Notification.permission === 'granted') {
                    new Notification(data.title, { body: data.message });
                }
            }
        });
        
        // Request notification permission
        if ('Notification' in window) {
            Notification.requestPermission();
        }
        
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Update active nav item
                document.querySelectorAll('.nav-item').forEach(navItem => {
                    navItem.classList.remove('active');
                });
                this.classList.add('active');
                
                // Show active section
                const sectionId = this.getAttribute('data-section');
                document.querySelectorAll('.section').forEach(section => {
                    section.classList.remove('active');
                });
                document.getElementById(sectionId).classList.add('active');
                
                // Load data for the section if needed
                if (sectionId === 'appointments-section') {
                    loadAppointments();
                } else if (sectionId === 'records-section') {
                    loadRecords();
                } else if (sectionId === 'notifications-section') {
                    loadNotifications();
                }
            });
        });
        
        // Load queue info on page load
        window.addEventListener('load', function() {
            loadQueueInfo();
            
            // Register device for push notifications
            registerDevice();
        });
        
        // Refresh queue button
        document.getElementById('refresh-queue').addEventListener('click', function() {
            loadQueueInfo();
        });
        
        // Voice assistant button
        document.getElementById('voice-button').addEventListener('click', function() {
            if ('webkitSpeechRecognition' in window) {
                const recognition = new webkitSpeechRecognition();
                recognition.lang = 'en-US';
                
                recognition.onstart = function() {
                    document.getElementById('voice-button').innerHTML = '<i class="bi bi-mic"></i>';
                };
                
                recognition.onresult = function(event) {
                    const transcript = event.results[0][0].transcript;
                    processVoiceQuery(transcript);
                };
                
                recognition.onend = function() {
                    document.getElementById('voice-button').innerHTML = '<i class="bi bi-mic-fill"></i>';
                };
                
                recognition.start();
            } else {
                alert('Voice recognition is not supported in your browser.');
            }
        });
        
        // Load queue info
        function loadQueueInfo() {
            fetch(`/api/mobile/queue/${pat