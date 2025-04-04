import unittest
from unittest.mock import patch, MagicMock
import random
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import all necessary modules
from solution import (
    CommunicationManager, 
    ChannelOptimizer, 
    MessageOptimizer, 
    patients, 
    message_templates, 
    select_optimal_channel
)
from advanced_effectiveness_measurement import AdvancedEffectivenessMeasurement
from effectiveness_measurement_integration import EnhancedCommunicationManager

class TestMultiLanguageCommunicationSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.test_patient = {
            "id": 100,
            "name": "Test Patient",
            "language": "Tamil",
            "age": 45,
            "channel": "SMS",
            "preferred_channels": ["SMS", "WhatsApp", "IVR"]
        }
        self.channel_optimizer = ChannelOptimizer()
        self.message_optimizer = MessageOptimizer()
        self.comm_manager = CommunicationManager()
        self.advanced_metrics = AdvancedEffectivenessMeasurement()
        self.enhanced_comm_manager = EnhancedCommunicationManager()

    def test_patient_data_integrity(self):
        """Test that patient data has all required fields"""
        required_fields = ["id", "name", "language", "age", "channel", "preferred_channels"]
        for patient in patients:
            for field in required_fields:
                self.assertIn(field, patient, f"Patient {patient['id']} missing {field}")

    def test_message_templates_structure(self):
        """Test message templates structure and completeness"""
        # Check all message types
        message_types = ["appointment_confirmation", "wait_time", "prescription_reminder"]
        for msg_type in message_types:
            self.assertIn(msg_type, message_templates)

        # Check all languages
        languages = ["Tamil", "Telugu", "Malayalam", "Hindi", "English"]
        for msg_type in message_templates:
            for lang in languages:
                self.assertIn(lang, message_templates[msg_type])
                # Check template variants
                self.assertIn("standard", message_templates[msg_type][lang])
                self.assertIn("elderly", message_templates[msg_type][lang])

    def test_channel_optimization(self):
        """Test channel optimization logic"""
        # Test elderly patient preference
        elderly_patient = self.test_patient.copy()
        elderly_patient["age"] = 70
        channel = self.channel_optimizer.get_optimal_channel(elderly_patient, "appointment_confirmation")
        self.assertEqual(channel, "IVR", "Elderly patients should prefer IVR")

        # Test young patient preference
        young_patient = self.test_patient.copy()
        young_patient["age"] = 25
        channel = self.channel_optimizer.get_optimal_channel(young_patient, "appointment_confirmation")
        self.assertEqual(channel, "WhatsApp", "Young patients should prefer WhatsApp")

        # Test middle-aged patient
        middle_aged_patient = self.test_patient.copy()
        middle_aged_patient["age"] = 45
        channel = self.channel_optimizer.get_optimal_channel(middle_aged_patient, "appointment_confirmation")
        self.assertEqual(channel, "SMS", "Middle-aged patients should use default channel")

    def test_message_sending(self):
        """Test message sending functionality"""
        # Test appointment confirmation
        result = self.comm_manager.send_message(
            self.test_patient,
            "appointment_confirmation",
            date="2024-03-30",
            time="10:00 AM"
        )
        self.assertTrue(result["success"])
        self.assertIn("2024-03-30", result["message"])
        self.assertIn("10:00 AM", result["message"])

        # Test wait time message
        result = self.comm_manager.send_message(
            self.test_patient,
            "wait_time",
            wait_time=15
        )
        self.assertTrue(result["success"])
        self.assertIn("15", result["message"])

        # Test prescription reminder
        result = self.comm_manager.send_message(
            self.test_patient,
            "prescription_reminder",
            medicine="Paracetamol",
            next_time="2:00 PM"
        )
        self.assertTrue(result["success"])
        self.assertIn("Paracetamol", result["message"])
        self.assertIn("2:00 PM", result["message"])

    def test_language_fallback(self):
        """Test language fallback to English for unsupported languages"""
        unsupported_lang_patient = self.test_patient.copy()
        unsupported_lang_patient["language"] = "French"
        
        result = self.comm_manager.send_message(
            unsupported_lang_patient,
            "appointment_confirmation",
            date="2024-03-30",
            time="10:00 AM"
        )
        
        self.assertTrue(result["success"])
        # Should use English template
        self.assertIn("Your appointment", result["message"])

    def test_elderly_template_selection(self):
        """Test elderly template selection for older patients"""
        elderly_patient = self.test_patient.copy()
        elderly_patient["age"] = 70
        elderly_patient["language"] = "English"
        
        result = self.comm_manager.send_message(
            elderly_patient,
            "appointment_confirmation",
            date="2024-03-30",
            time="10:00 AM"
        )
        
        self.assertTrue(result["success"])
        # Should use elderly template
        self.assertIn("medical appointment", result["message"])

    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        # Test positive sentiment
        response_data = self.comm_manager.process_response(
            self.test_patient["id"],
            "Thank you for the reminder, I will be there on time.",
            datetime.now()
        )
        self.assertGreater(response_data["sentiment_score"], 0)
        
        # Test negative sentiment
        response_data = self.comm_manager.process_response(
            self.test_patient["id"],
            "I cannot make it to the appointment, I am very disappointed.",
            datetime.now()
        )
        self.assertLess(response_data["sentiment_score"], 0)
        
        # Test neutral sentiment
        response_data = self.comm_manager.process_response(
            self.test_patient["id"],
            "I received your message.",
            datetime.now()
        )
        self.assertAlmostEqual(response_data["sentiment_score"], 0, delta=0.3)

    def test_system_performance_metrics(self):
        """Test system performance metrics calculation"""
        # Send multiple messages to generate metrics
        for patient in patients[:5]:
            self.comm_manager.send_message(
                patient,
                "appointment_confirmation",
                date="2024-03-30",
                time="10:00 AM"
            )
            
            # Simulate some responses
            if random.random() > 0.5:
                self.comm_manager.process_response(
                    patient["id"],
                    "Thank you for the reminder.",
                    datetime.now() + timedelta(minutes=30)
                )
        
        # Get performance metrics
        metrics = self.comm_manager.get_system_performance()
        
        # Check required metrics
        self.assertIn("delivery_success_rate", metrics)
        self.assertIn("patient_response_rate", metrics)
        self.assertIn("channel_usage", metrics)
        self.assertIn("language_usage", metrics)
        self.assertIn("weighted_effectiveness", metrics)
        self.assertIn("peak_hours", metrics)
        self.assertIn("last_updated", metrics)
        self.assertIn("hourly_performance", metrics)
        self.assertIn("channel_engagement", metrics)

    def test_advanced_effectiveness_measurement(self):
        """Test advanced effectiveness measurement functionality"""
        # Track message delivery
        self.advanced_metrics.track_message_delivery(
            self.test_patient,
            "SMS",
            "appointment_confirmation",
            status="delivered"
        )
        
        # Track patient response
        self.advanced_metrics.track_patient_response(
            self.test_patient,
            "SMS",
            "appointment_confirmation",
            "Thank you for the reminder.",
            response_time=120  # 2 minutes
        )
        
        # Generate comprehensive report
        report = self.advanced_metrics.generate_comprehensive_report()
        
        # Check report structure
        self.assertIn("channel_effectiveness", report)
        self.assertIn("demographic_performance", report)
        self.assertIn("message_effectiveness", report)
        self.assertIn("real_time_metrics", report)
        self.assertIn("summary_metrics", report)
        
        # Check channel effectiveness metrics
        self.assertIn("SMS", report["channel_effectiveness"])
        self.assertIn("delivery_rate", report["channel_effectiveness"]["SMS"])
        self.assertIn("response_rate", report["channel_effectiveness"]["SMS"])
        
        # Check demographic performance metrics
        self.assertIn("language", report["demographic_performance"])
        self.assertIn("age_group", report["demographic_performance"])
        
        # Check real-time metrics
        self.assertIn("channel_health", report["real_time_metrics"])
        self.assertIn("delivery_trends", report["real_time_metrics"])

    def test_enhanced_communication_manager(self):
        """Test enhanced communication manager functionality"""
        # Send message through enhanced manager
        result = self.enhanced_comm_manager.send_message(
            self.test_patient,
            "appointment_confirmation",
            date="2024-03-30",
            time="10:00 AM"
        )
        
        self.assertTrue(result["success"])
        
        # Process response
        self.enhanced_comm_manager.process_response(
            self.test_patient["id"],
            "Thank you for the reminder.",
            datetime.now() + timedelta(minutes=30)
        )
        
        # Get enhanced performance metrics
        enhanced_metrics = self.enhanced_comm_manager.get_enhanced_system_performance()
        
        # Check enhanced metrics structure
        self.assertIn("advanced_metrics", enhanced_metrics)
        self.assertIn("cross_channel_analysis", enhanced_metrics)
        
        # Check advanced metrics components
        self.assertIn("channel_effectiveness", enhanced_metrics["advanced_metrics"])
        self.assertIn("demographic_performance", enhanced_metrics["advanced_metrics"])
        self.assertIn("message_effectiveness", enhanced_metrics["advanced_metrics"])
        self.assertIn("real_time_metrics", enhanced_metrics["advanced_metrics"])
        self.assertIn("summary_metrics", enhanced_metrics["advanced_metrics"])

    def test_channel_health_monitoring(self):
        """Test channel health monitoring functionality"""
        # Get channel health status
        health_status = self.enhanced_comm_manager.get_channel_health_status()
        
        # Check health status for each channel
        for channel in ["SMS", "WhatsApp", "IVR"]:
            self.assertIn(channel, health_status)
            self.assertIn("uptime", health_status[channel])
            self.assertIn("latency", health_status[channel])
            self.assertIn("error_rate", health_status[channel])
            self.assertIn("last_check", health_status[channel])

    def test_delivery_status_tracking(self):
        """Test delivery status tracking functionality"""
        # Track some message deliveries
        self.advanced_metrics.track_message_delivery(
            self.test_patient,
            "SMS",
            "appointment_confirmation",
            status="delivered"
        )
        
        self.advanced_metrics.track_message_delivery(
            self.test_patient,
            "WhatsApp",
            "wait_time",
            status="failed"
        )
        
        # Get delivery status tracking
        delivery_status = self.enhanced_comm_manager.get_delivery_status_tracking()
        
        # Check delivery status for each channel
        for channel in ["SMS", "WhatsApp", "IVR"]:
            self.assertIn(channel, delivery_status)
            self.assertIn("success_rate", delivery_status[channel])
            self.assertIn("failure_rate", delivery_status[channel])
            self.assertIn("total_sent", delivery_status[channel])
            self.assertIn("total_delivered", delivery_status[channel])
            self.assertIn("total_failed", delivery_status[channel])

    def test_response_time_analytics(self):
        """Test response time analytics functionality"""
        # Track some patient responses
        self.advanced_metrics.track_patient_response(
            self.test_patient,
            "SMS",
            "appointment_confirmation",
            "Thank you for the reminder.",
            response_time=120  # 2 minutes
        )
        
        self.advanced_metrics.track_patient_response(
            self.test_patient,
            "SMS",
            "appointment_confirmation",
            "I will be there.",
            response_time=300  # 5 minutes
        )
        
        # Get response time analytics
        response_analytics = self.enhanced_comm_manager.get_response_time_analytics()
        
        # Check response analytics for SMS channel
        self.assertIn("SMS", response_analytics)
        self.assertIn("average", response_analytics["SMS"])
        self.assertIn("minimum", response_analytics["SMS"])
        self.assertIn("maximum", response_analytics["SMS"])
        self.assertIn("median", response_analytics["SMS"])
        self.assertIn("p90", response_analytics["SMS"])
        self.assertIn("sample_size", response_analytics["SMS"])

    def test_optimal_channel_selection_algorithm(self):
        """Test the AI-driven optimal channel selection algorithm"""
        # Test elderly patient
        elderly_patient = self.test_patient.copy()
        elderly_patient["age"] = 70
        channel = select_optimal_channel(elderly_patient, "appointment_confirmation")
        self.assertIn(channel, elderly_patient["preferred_channels"])
        
        # Test young patient
        young_patient = self.test_patient.copy()
        young_patient["age"] = 25
        channel = select_optimal_channel(young_patient, "appointment_confirmation")
        self.assertIn(channel, young_patient["preferred_channels"])
        
        # Test different message types
        channel_wait_time = select_optimal_channel(self.test_patient, "wait_time")
        channel_prescription = select_optimal_channel(self.test_patient, "prescription_reminder")
        self.assertIn(channel_wait_time, self.test_patient["preferred_channels"])
        self.assertIn(channel_prescription, self.test_patient["preferred_channels"])

    @patch('solution.datetime')
    def test_time_based_channel_selection(self, mock_datetime):
        """Test time-based channel selection"""
        # Test business hours
        mock_datetime.now.return_value = datetime(2024, 3, 30, 14, 0)  # 2 PM
        channel_business = select_optimal_channel(self.test_patient, "appointment_confirmation")
        
        # Test after hours
        mock_datetime.now.return_value = datetime(2024, 3, 30, 22, 0)  # 10 PM
        channel_after = select_optimal_channel(self.test_patient, "appointment_confirmation")
        
        self.assertIn(channel_business, self.test_patient["preferred_channels"])
        self.assertIn(channel_after, self.test_patient["preferred_channels"])

if __name__ == "__main__":
    unittest.main()