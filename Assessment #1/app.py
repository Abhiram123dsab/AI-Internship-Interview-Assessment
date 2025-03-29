import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from datetime import datetime, timedelta
import random
import json
from enhanced_clinic_system import predict_wait_time_enhanced, optimize_schedule_enhanced, EnhancedClinicMonitor, load_data
from whatsapp_integration import WhatsAppIntegration

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'clinic-wait-time-secret'
socketio = SocketIO(app)

# Load data
df = load_data()

# Initialize clinic monitor
clinic_monitor = EnhancedClinicMonitor()

# Initialize WhatsApp integration
whatsapp_integration = WhatsAppIntegration()

# Update clinic metrics initially
clinic_monitor.update_metrics(df)

# Routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/patient-portal')
def patient_portal():
    return render_template('patient_portal.html')

@app.route('/api/metrics')
def get_metrics():
    return jsonify(clinic_monitor.get_current_status())

@app.route('/api/predict-wait-time', methods=['POST'])
def predict_wait_time_api():
    data = request.json
    doctor_id = data.get('doctor_id')
    scheduled_time = data.get('scheduled_time')
    
    if not doctor_id or not scheduled_time:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        scheduled_time = pd.to_datetime(scheduled_time)
        wait_time = predict_wait_time_enhanced(doctor_id, scheduled_time)
        return jsonify({
            'doctor_id': doctor_id,
            'scheduled_time': scheduled_time.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_wait_time': float(wait_time),
            'unit': 'minutes'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-schedule', methods=['POST'])
def optimize_schedule_api():
    data = request.json
    appointments = data.get('appointments')
    
    if not appointments:
        return jsonify({'error': 'No appointments provided'}), 400
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(appointments)
        optimized_df = optimize_schedule_enhanced(df)
        
        # Convert back to dict for JSON response
        result = optimized_df.to_dict(orient='records')
        
        # Calculate metrics
        original_wait = (pd.to_datetime(df['actual_time']) - pd.to_datetime(df['scheduled_time'])).dt.total_seconds().mean() / 60
        optimized_wait = (pd.to_datetime(optimized_df['actual_time']) - pd.to_datetime(optimized_df['scheduled_time'])).dt.total_seconds().mean() / 60
        reduction_percent = ((original_wait - optimized_wait) / original_wait * 100) if original_wait > 0 else 0
        
        return jsonify({
            'optimized_appointments': result,
            'metrics': {
                'original_avg_wait': original_wait,
                'optimized_avg_wait': optimized_wait,
                'reduction_percent': reduction_percent
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient-queue/<int:patient_id>')
def get_patient_queue(patient_id):
    # In a real app, this would query a database
    # For demo, we'll generate sample data
    position = random.randint(1, 5)
    wait_time = position * 8 + random.randint(-5, 5)
    
    return jsonify({
        'patient_id': patient_id,
        'position': position,
        'wait_time': wait_time,
        'status': 'In Queue' if position > 1 else 'Ready',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('metrics_update', clinic_monitor.get_current_status())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Simulate real-time updates
def simulate_clinic_activity():
    """Simulate clinic activity for demonstration"""
    global df
    
    # Add some random new appointments
    num_new = random.randint(1, 3)
    for _ in range(num_new):
        new_appointment = {
            'doctor_id': random.randint(1, 15),
            'scheduled_time': (datetime.now() + timedelta(minutes=random.randint(30, 180))).strftime('%Y-%m-%d %H:%M:%S'),
            'actual_time': (datetime.now() + timedelta(minutes=random.randint(60, 240))).strftime('%Y-%m-%d %H:%M:%S'),
            'patient_id': random.randint(1000, 9999)
        }
        df = pd.concat([df, pd.DataFrame([new_appointment])], ignore_index=True)
    
    # Update metrics
    clinic_monitor.update_metrics(df)
    
    # Emit queue updates for random patients
    for _ in range(random.randint(1, 3)):
        patient_id = random.choice(df['patient_id'].tolist())
        position = random.randint(1, 5)
        wait_time = position * 8 + random.randint(-5, 5)
        
        socketio.emit('queue_update', {
            'patient_id': f"P-{patient_id}",
            'position': position,
            'wait_time': wait_time,
            'status': 'In Queue' if position > 1 else 'Ready'
        })

# Schedule periodic updates
@socketio.on('start_simulation')
def start_simulation():
    def update_loop():
        while True:
            simulate_clinic_activity()
            socketio.sleep(30)  # Update every 30 seconds
    
    socketio.start_background_task(update_loop)

# WhatsApp webhook routes
@app.route('/whatsapp/webhook', methods=['POST'])
def whatsapp_webhook():
    """Forward incoming WhatsApp messages to the WhatsApp integration"""
    # This route is just a proxy to the webhook server in the WhatsApp integration
    # In a production environment, you would handle this directly in the main app
    return jsonify({'status': 'success', 'message': 'Webhook received'})

@app.route('/whatsapp/status', methods=['POST'])
def whatsapp_status():
    """Forward message status updates to the WhatsApp integration"""
    # Extract message details from the request
    message_sid = request.values.get('MessageSid')
    message_status = request.values.get('MessageStatus')
    
    if not message_sid or not message_status:
        return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
    
    # Update message status in tracker
    success = whatsapp_integration.update_message_status(message_sid, message_status)
    
    if success:
        return jsonify({'status': 'success', 'message': f'Status updated to {message_status}'})
    else:
        return jsonify({'status': 'error', 'message': 'Message not found in tracker'}), 404

@app.route('/api/whatsapp/analytics', methods=['GET'])
def whatsapp_analytics():
    """Get analytics on WhatsApp message delivery and read status"""
    analytics = whatsapp_integration.get_message_status_analytics()
    return jsonify(analytics)

@app.route('/api/send-queue-update', methods=['POST'])
def send_queue_update():
    """Send queue position update to a patient via WhatsApp"""
    data = request.json
    patient_phone = data.get('patient_phone')
    queue_position = data.get('queue_position')
    wait_time = data.get('wait_time')
    doctor_name = data.get('doctor_name')
    
    if not patient_phone or not queue_position or not wait_time:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        result = whatsapp_integration.send_queue_position_notification(
            patient_phone=patient_phone,
            queue_position=queue_position,
            wait_time=wait_time,
            doctor_name=doctor_name
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-appointment-reminder', methods=['POST'])
def send_appointment_reminder():
    """Send appointment reminder to a patient via WhatsApp"""
    data = request.json
    patient_phone = data.get('patient_phone')
    doctor_name = data.get('doctor_name')
    appointment_time = data.get('appointment_time')
    clinic_location = data.get('clinic_location')
    appointment_id = data.get('appointment_id')
    
    if not patient_phone or not doctor_name or not appointment_time:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        result = whatsapp_integration.send_appointment_reminder(
            patient_phone=patient_phone,
            doctor_name=doctor_name,
            appointment_time=appointment_time,
            clinic_location=clinic_location,
            appointment_id=appointment_id
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start simulation on server start
    socketio.start_background_task(start_simulation)
    
    # Start WhatsApp webhook server
    # In a production environment, you would use a proper webhook setup with Twilio
    # For development, we're using a separate server
    whatsapp_integration.start_webhook_server(port=5001)
    
    # Run the app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)