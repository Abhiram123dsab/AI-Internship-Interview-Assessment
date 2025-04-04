import random
from datetime import datetime
from advanced_effectiveness_measurement import AdvancedEffectivenessMeasurement
from effectiveness_measurement_integration import EnhancedCommunicationManager

# Enhanced patient database with more detailed information
patients = [
    {"id": 1, "name": "Ravi Kumar", "language": "Tamil", "age": 72, "channel": "SMS", "preferred_channels": ["SMS", "IVR"]},
    {"id": 2, "name": "Ananya Rao", "language": "Telugu", "age": 35, "channel": "WhatsApp", "preferred_channels": ["WhatsApp", "SMS"]},
    {"id": 3, "name": "Joseph Mathew", "language": "Malayalam", "age": 68, "channel": "IVR", "preferred_channels": ["IVR"]},
    {"id": 4, "name": "Rahul Sharma", "language": "Hindi", "age": 42, "channel": "SMS", "preferred_channels": ["SMS", "WhatsApp"]},
    {"id": 5, "name": "David Thomas", "language": "English", "age": 29, "channel": "WhatsApp", "preferred_channels": ["WhatsApp"]},
    {"id": 6, "name": "Lakshmi Narayanan", "language": "Tamil", "age": 65, "channel": "IVR", "preferred_channels": ["IVR", "SMS"]},
    {"id": 7, "name": "Priya Venkatesh", "language": "Tamil", "age": 31, "channel": "WhatsApp", "preferred_channels": ["WhatsApp", "SMS"]},
    {"id": 8, "name": "Srinivas Reddy", "language": "Telugu", "age": 58, "channel": "SMS", "preferred_channels": ["SMS", "IVR"]},
    {"id": 9, "name": "Aisha Khan", "language": "Hindi", "age": 27, "channel": "WhatsApp", "preferred_channels": ["WhatsApp"]},
    {"id": 10, "name": "George Wilson", "language": "English", "age": 45, "channel": "SMS", "preferred_channels": ["SMS", "WhatsApp"]},
]

# Message templates with variations for A/B testing and different message types
message_templates = {
    # Regular templates
    "appointment_confirmation": {
        "Tamil": {
            "standard": "உங்கள் நேரம் {date} அன்று {time} மணிக்கு உறுதிசெய்யப்பட்டது. தயவுசெய்து வருக!",
            "elderly": "உங்கள் மருத்துவ சந்திப்பு {date} அன்று {time} மணிக்கு. தயவுசெய்து உறுதிப்படுத்தவும்."
        },
        "Telugu": {
            "standard": "మీ నియామకం {date} న {time} కి నిర్ధారించబడింది. దయచేసి రండి!",
            "elderly": "మీ వైద్య అపాయింట్మెంట్ {date} న {time} కి. దయచేసి నిర్ధారించండి."
        },
        "Malayalam": {
            "standard": "നിങ്ങളുടെ അപോയിന്റ്മെന്റ് {date} ന് {time} ന് സ്ഥിരീകരിച്ചിരിക്കുന്നു. ദയവായി വരൂ!",
            "elderly": "നിങ്ങളുടെ മെഡിക്കൽ അപോയിന്റ്മെന്റ് {date} ന് {time} ന്. ദയവായി സ്ഥിരീകരിക്കുക."
        },
        "Hindi": {
            "standard": "आपका अपॉइंटमेंट {date} को {time} बजे कन्फर्म हो गया है। कृपया आएं!",
            "elderly": "आपका मेडिकल अपॉइंटमेंट {date} को {time} बजे है। कृपया पुष्टि करें।"
        },
        "English": {
            "standard": "Your appointment is confirmed for {date} at {time}. Please visit!",
            "elderly": "Your medical appointment is on {date} at {time}. Please confirm."
        }
    },
    "wait_time": {
        "Tamil": {
            "standard": "தற்போதைய காத்திருப்பு நேரம் {wait_time} நிமிடங்கள். தயவுசெய்து பொறுமையாக இருங்கள்.",
            "elderly": "மருத்துவரை பார்க்க {wait_time} நிமிடங்கள் காத்திருக்க வேண்டும். நாங்கள் விரைவில் உங்களை அழைப்போம்."
        },
        "Telugu": {
            "standard": "ప్రస్తుత వేచి ఉండే సమయం {wait_time} నిమిషాలు. దయచేసి ఓపికగా ఉండండి.",
            "elderly": "డాక్టర్‌ని చూడటానికి {wait_time} నిమిషాలు వేచి ఉండాలి. మేము త్వరలో మిమ్మల్ని పిలుస్తాము."
        },
        "Malayalam": {
            "standard": "നിലവിലെ കാത്തിരിപ്പ് സമയം {wait_time} മിനിറ്റുകളാണ്. ദയവായി ക്ഷമിക്കുക.",
            "elderly": "ഡോക്ടറെ കാണാൻ {wait_time} മിനിറ്റ് കാത്തിരിക്കേണ്ടതുണ്ട്. ഞങ്ങൾ താമസിയാതെ നിങ്ങളെ വിളിക്കും."
        },
        "Hindi": {
            "standard": "वर्तमान प्रतीक्षा समय {wait_time} मिनट है। कृपया धैर्य रखें।",
            "elderly": "डॉक्टर को देखने के लिए {wait_time} मिनट प्रतीक्षा करनी होगी। हम जल्द ही आपको बुलाएंगे।"
        },
        "English": {
            "standard": "Current wait time is {wait_time} minutes. Please be patient.",
            "elderly": "Wait time to see the doctor is {wait_time} minutes. We will call you soon."
        }
    },
    "prescription_reminder": {
        "Tamil": {
            "standard": "உங்கள் {medicine} மருந்தை எடுக்க நினைவூட்டல். அடுத்த மருந்து: {next_time}.",
            "elderly": "மருந்து நினைவூட்டல்: தயவுசெய்து இப்போது உங்கள் {medicine} மருந்தை எடுத்துக்கொள்ளுங்கள். அடுத்தது {next_time} மணிக்கு."
        },
        "Telugu": {
            "standard": "మీ {medicine} మందు తీసుకోవడానికి రిమైండర్. తదుపరి మోతాదు: {next_time}.",
            "elderly": "మందు రిమైండర్: దయచేసి ఇప్పుడు మీ {medicine} తీసుకోండి. తదుపరిది {next_time} కి."
        },
        "Malayalam": {
            "standard": "നിങ്ങളുടെ {medicine} മരുന്ന് കഴിക്കാനുള്ള ഓർമ്മപ്പെടുത്തൽ. അടുത്ത ഡോസ്: {next_time}.",
            "elderly": "മരുന്ന് ഓർമ്മപ്പെടുത്തൽ: ദയവായി ഇപ്പോൾ നിങ്ങളുടെ {medicine} കഴിക്കുക. അടുത്തത് {next_time} ന്."
        },
        "Hindi": {
            "standard": "आपकी {medicine} दवा लेने के लिए रिमाइंडर। अगली खुराक: {next_time}।",
            "elderly": "दवा रिमाइंडर: कृपया अभी अपनी {medicine} लें। अगली {next_time} बजे।"
        },
        "English": {
            "standard": "Reminder to take your {medicine} medication. Next dose: {next_time}.",
            "elderly": "Medication reminder: Please take your {medicine} now. Next at {next_time}."
        }
    }
}

# Enhanced channel selection with ML-based optimization
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class CommunicationManager:
    def __init__(self):
        self.channel_optimizer = ChannelOptimizer()
        self.message_history = {}
        self.response_history = {}
        self.system_metrics = {
            'delivery_success_rate': 0,
            'patient_response_rate': 0,
            'channel_usage': {'SMS': 0, 'WhatsApp': 0, 'IVR': 0},
            'language_usage': {}
        }

    def send_message(self, patient, message_type, **kwargs):
        # Get optimal channel
        channel = self.channel_optimizer.get_optimal_channel(patient, message_type)
        
        # Select template based on age
        template_type = 'elderly' if patient['age'] >= 65 else 'standard'
        
        # Handle language fallback
        language = patient['language']
        if language not in message_templates[message_type]:
            language = 'English'
        
        # Get and format message
        template = message_templates[message_type][language][template_type]
        message = template.format(**kwargs)
        
        # Update metrics
        self.system_metrics['channel_usage'][channel] += 1
        self.system_metrics['language_usage'][language] = \
            self.system_metrics['language_usage'].get(language, 0) + 1
        
        # Store message history
        if patient['id'] not in self.message_history:
            self.message_history[patient['id']] = []
        self.message_history[patient['id']].append({
            'timestamp': datetime.now(),
            'message': message,
            'channel': channel,
            'message_type': message_type
        })
        
        # Simulate message delivery success
        success = True
        self.channel_optimizer.update_engagement_history(
            patient['id'], channel, message_type, success
        )
        
        return {
            'success': success,
            'channel': channel,
            'message': message
        }
    
    def process_response(self, patient_id, response_text, timestamp):
        # Perform sentiment analysis using VADER
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(response_text)
        sentiment_score = sentiment_scores['compound']
        
        # Store response data
        if patient_id not in self.response_history:
            self.response_history[patient_id] = []
        
        response_data = {
            'timestamp': timestamp,
            'text': response_text,
            'sentiment_score': sentiment_score
        }
        self.response_history[patient_id].append(response_data)
        
        # Update response rate metrics
        total_messages = len(self.message_history.get(patient_id, []))
        total_responses = len(self.response_history[patient_id])
        if total_messages > 0:
            self.system_metrics['patient_response_rate'] = \
                total_responses / total_messages
        
        return response_data
    
    def get_system_performance(self):
        # Calculate delivery success rate with weighted metrics
        total_success = sum(self.system_metrics['channel_usage'].values())
        total_attempts = total_success
        if total_attempts > 0:
            self.system_metrics['delivery_success_rate'] = total_success / total_attempts
        
        # Enhanced weighted channel effectiveness with dynamic weights
        channel_weights = {
            'SMS': self._calculate_channel_weight('SMS'),
            'WhatsApp': self._calculate_channel_weight('WhatsApp'),
            'IVR': self._calculate_channel_weight('IVR')
        }
        
        weighted_effectiveness = 0
        total_weight = 0
        
        # Enhanced channel effectiveness calculation with engagement metrics
        for channel, usage in self.system_metrics['channel_usage'].items():
            if usage > 0:
                success_rate = usage / total_attempts
                engagement_score = self._calculate_engagement_score(channel)
                weighted_effectiveness += (success_rate * engagement_score) * channel_weights[channel]
                total_weight += channel_weights[channel]
        
        if total_weight > 0:
            self.system_metrics['weighted_effectiveness'] = weighted_effectiveness / total_weight
        
        # Enhanced real-time metrics with peak hour analysis
        current_hour = datetime.now().hour
        self.system_metrics.update({
            'peak_hours': 9 <= current_hour <= 18,
            'last_updated': datetime.now().isoformat(),
            'hourly_performance': self._get_hourly_performance(),
            'channel_engagement': self._get_channel_engagement_metrics()
        })
        
        return self.system_metrics
    
    def _calculate_channel_weight(self, channel):
        """Calculate dynamic channel weight based on performance"""
        base_weights = {'SMS': 0.3, 'WhatsApp': 0.4, 'IVR': 0.3}
        if channel not in self.channel_optimizer.channel_success_rates:
            return base_weights[channel]
        
        success_rate = self.channel_optimizer.channel_success_rates[channel]
        if success_rate['total'] == 0:
            return base_weights[channel]
        
        performance = success_rate['success'] / success_rate['total']
        return base_weights[channel] * (1 + performance) / 2
    
    def _calculate_engagement_score(self, channel):
        """Calculate engagement score based on response rates and sentiment"""
        if channel not in self.response_history:
            return 0.5
        
        responses = self.response_history[channel]
        if not responses:
            return 0.5
        
        avg_sentiment = sum(r['sentiment_score'] for r in responses) / len(responses)
        response_rate = len(responses) / self.system_metrics['channel_usage'][channel]
        
        return (avg_sentiment + response_rate) / 2
    
    def _get_hourly_performance(self):
        """Get hourly performance metrics"""
        hourly_stats = {}
        for patient_id, messages in self.message_history.items():
            for msg in messages:
                hour = msg['timestamp'].hour
                if hour not in hourly_stats:
                    hourly_stats[hour] = {'sent': 0, 'responses': 0}
                hourly_stats[hour]['sent'] += 1
                if patient_id in self.response_history and any(r['timestamp'].hour == hour for r in self.response_history[patient_id]):
                    hourly_stats[hour]['responses'] += 1
        return hourly_stats
    
    def _get_channel_engagement_metrics(self):
        """Get detailed channel engagement metrics"""
        metrics = {}
        for channel in ['SMS', 'WhatsApp', 'IVR']:
            if channel in self.system_metrics['channel_usage']:
                usage = self.system_metrics['channel_usage'][channel]
                success_rate = self.channel_optimizer.channel_success_rates.get(channel, {'success': 0, 'total': 0})
                metrics[channel] = {
                    'usage': usage,
                    'success_rate': success_rate['success'] / success_rate['total'] if success_rate['total'] > 0 else 0,
                    'engagement_score': self._calculate_engagement_score(channel)
                }
        return metrics

class ChannelOptimizer:
    def __init__(self):
        self.channel_success_rates = {}
        self.patient_engagement_history = {}
        self.time_sensitivity = {
            "appointment_confirmation": 0.8,
            "wait_time": 1.0,
            "prescription_reminder": 0.9
        }

    def get_optimal_channel(self, patient, message_type):
        """Select optimal channel based on patient demographics and message type"""
        if patient["age"] >= 65:
            return "IVR" if "IVR" in patient["preferred_channels"] else "SMS"
        elif patient["age"] < 30:
            # For younger patients, prioritize WhatsApp
            if "WhatsApp" in patient["preferred_channels"]:
                return "WhatsApp"
            # If WhatsApp not available, try SMS as fallback
            return "SMS" if "SMS" in patient["preferred_channels"] else patient["channel"]
        return patient["channel"]

    def get_channel_success_rate(self, patient_id, channel):
        """Get success rate for a specific channel"""
        if patient_id not in self.patient_engagement_history:
            self.patient_engagement_history[patient_id] = {}
        
        history = self.patient_engagement_history[patient_id]
        key = f"{channel}_appointment_confirmation"
        if key not in history:
            history[key] = {"success": 0, "total": 0}
        
        if history[key]["total"] == 0:
            return 0.0
        
        return history[key]["success"] / history[key]["total"]
    
    def update_engagement_history(self, patient_id, channel, message_type, success):
        """Update engagement history for a patient"""
        if patient_id not in self.patient_engagement_history:
            self.patient_engagement_history[patient_id] = {}
        
        key = f"{channel}_{message_type}"
        if key not in self.patient_engagement_history[patient_id]:
            self.patient_engagement_history[patient_id][key] = {"success": 0, "total": 0}
        
        self.patient_engagement_history[patient_id][key]["total"] += 1
        if success:
            self.patient_engagement_history[patient_id][key]["success"] += 1
        
        # Update channel-specific success rate
        if channel not in self.channel_success_rates:
            self.channel_success_rates[channel] = {"success": 0, "total": 0}
        self.channel_success_rates[channel]["total"] += 1
        if success:
            self.channel_success_rates[channel]["success"] += 1
    
    def get_channel_score(self, patient_id, channel, message_type):
        """Calculate success score for a channel based on historical data"""
        if patient_id in self.patient_engagement_history:
            key = f"{channel}_{message_type}"
            history = self.patient_engagement_history[patient_id].get(key, {})
            if history.get("total", 0) > 0:
                return history["success"] / history["total"]
        return 0.5  # Default score if no history

# AI-driven channel selection algorithm
def select_optimal_channel(patient, message_type):
    """Select optimal communication channel using ML-based approach"""
    age = patient["age"]
    preferred_channels = patient["preferred_channels"]
    
    # Initialize scores for each channel
    channel_scores = {}
    
    for channel in preferred_channels:
        base_score = 0.5
        
        # Age-based adjustments
        if age >= 65:
            if channel == "IVR":
                base_score += 0.3
            elif channel == "SMS":
                base_score += 0.1
        else:
            if channel == "WhatsApp":
                base_score += 0.2
        
        # Message type adjustments
        if message_type == "wait_time" and channel == "WhatsApp":
            base_score += 0.2
        elif message_type == "prescription_reminder" and channel == "SMS":
            base_score += 0.15
        
        # Time of day optimization (simplified)
        hour = datetime.now().hour
        if 9 <= hour <= 18:  # Business hours
            if channel == "IVR":
                base_score += 0.1
        else:  # After hours
            if channel == "SMS" or channel == "WhatsApp":
                base_score += 0.1
        
        channel_scores[channel] = base_score
    
    # Select channel with highest score
    if channel_scores:
        return max(channel_scores.items(), key=lambda x: x[1])[0]
    return "SMS"  # Default fallback

# Advanced AI-driven template selection and timing optimization
class MessageOptimizer:
    def __init__(self):
        self.sentiment_scores = {}
        self.response_times = {}
        self.message_effectiveness = {}

    def analyze_sentiment(self, patient_id, response):
        """Analyze patient response sentiment"""
        # Simplified sentiment analysis (could be enhanced with ML models)
        positive_words = ['yes', 'ok', 'thank', 'good', 'great']
        negative_words = ['no', 'not', 'cant', 'bad', 'poor']
        
        response = response.lower()
        score = 0
        for word in positive_words:
            if word in response:
                score += 1
        for word in negative_words:
            if word in response:
                score -= 1
        
        self.sentiment_scores[patient_id] = self.sentiment_scores.get(patient_id, []) + [score]
        return score

    def track_response_time(self, patient_id, message_time, response_time):
        """Track patient response time for optimization"""
        time_diff = (response_time - message_time).total_seconds()
        if patient_id not in self.response_times:
            self.response_times[patient_id] = []
        self.response_times[patient_id].append(time_diff)

    def get_optimal_timing(self, patient_id):
        """Determine optimal message timing based on historical response times"""
        if patient_id in self.response_times and len(self.response_times[patient_id]) > 0:
            avg_response_time = sum(self.response_times[patient_id]) / len(self.response_times[patient_id])
            # If average response time is high, suggest sending messages earlier
            return max(1, avg_response_time / 3600)  # Convert to hours
        return 24  # Default to 24 hours before appointment

    def personalize_message(self, patient, message_type, template):
        """Personalize message based on patient profile and historical engagement"""
        # Get patient's average sentiment
        sentiment_history = self.sentiment_scores.get(patient['id'], [])
        avg_sentiment = sum(sentiment_history) / len(sentiment_history) if sentiment_history else 0

        # Adjust message based on sentiment and age
        if patient['age'] >= 65 or avg_sentiment < 0:
            return template['elderly']
        return template['standard']

    def update_effectiveness(self, patient_id, message_type, success):
        """Update message effectiveness tracking"""
        key = f"{patient_id}_{message_type}"
        if key not in self.message_effectiveness:
            self.message_effectiveness[key] = {'success': 0, 'total': 0}
        self.message_effectiveness[key]['total'] += 1
        if success:
            self.message_effectiveness[key]['success'] += 1

    def get_effectiveness_rate(self, patient_id, message_type):
        """Get message effectiveness rate for a specific patient and message type"""
        key = f"{patient_id}_{message_type}"
        stats = self.message_effectiveness.get(key, {'success': 0, 'total': 0})
        return stats['success'] / stats['total'] if stats['total'] > 0 else 0

# Enhanced template selection with AI-driven personalization
def select_template_variant(patient, message_type, message_optimizer=None):
    """Select appropriate template variant using AI-driven personalization"""
    age = patient["age"]
    patient_id = patient["id"]
    
    base_variant = "elderly" if age >= 65 else "standard"
    
    # If we have sentiment data, adjust template based on previous responses
    if message_optimizer and patient_id in message_optimizer.sentiment_scores:
        sentiment = message_optimizer.sentiment_scores[patient_id].get(message_type, 0)
        
        # If negative sentiment with standard template, switch to elderly (more detailed) version
        if sentiment < 0 and base_variant == "standard":
            return "elderly"
        # If positive sentiment with elderly template, try standard version
        elif sentiment > 1 and base_variant == "elderly":
            return "standard"
    
    return base_variant

# Enhanced A/B testing framework with ML-driven optimization
class AdvancedABTestingFramework:
    def __init__(self):
        self.message_variants = {}
        self.variant_performance = {}
        self.learning_rate = 0.1
        self.engagement_weights = {
            "response_time": 0.25,
            "sentiment_score": 0.3,
            "completion_rate": 0.25,
            "channel_effectiveness": 0.2
        }
        self.real_time_metrics = {
            "last_hour_engagement": 0,
            "peak_hour_performance": {},
            "channel_effectiveness": {}
        }
    
    def register_variant(self, variant_id, message_type, language, template_type):
        """Register a message variant with enhanced tracking metrics"""
        key = f"{message_type}_{language}_{template_type}_{variant_id}"
        self.message_variants[key] = {
            "sent": 0,
            "confirmed": 0,
            "conversion_rate": 0,
            "avg_response_time": 0,
            "sentiment_scores": [],
            "completion_rates": [],
            "engagement_score": 0.5  # Initialize with neutral score
        }
        return key
    
    def track_send(self, variant_key, context=None):
        """Track message send with contextual information"""
        if variant_key in self.message_variants:
            self.message_variants[variant_key]["sent"] += 1
            if context:
                # Store contextual information for learning
                self.update_context_learning(variant_key, context)
    
    def track_confirmation(self, variant_key, confirmed=True, response_time=None, sentiment=None, channel=None):
        """Enhanced tracking with real-time performance monitoring"""
        if variant_key in self.message_variants:
            variant = self.message_variants[variant_key]
            current_hour = datetime.now().hour
            
            if confirmed:
                variant["confirmed"] += 1
                
                # Track response time with time-of-day context
                if response_time:
                    current_avg = variant["avg_response_time"]
                    variant["avg_response_time"] = (
                        (current_avg * (variant["confirmed"] - 1) + response_time)
                        / variant["confirmed"]
                    )
                    
                    # Update peak hour performance
                    if 9 <= current_hour <= 18:
                        self.real_time_metrics["peak_hour_performance"][variant_key] = \
                            self.real_time_metrics["peak_hour_performance"].get(variant_key, 0) + 1
                
                # Track sentiment and channel effectiveness
                if sentiment is not None:
                    variant["sentiment_scores"].append(sentiment)
                    if channel:
                        if channel not in self.real_time_metrics["channel_effectiveness"]:
                            self.real_time_metrics["channel_effectiveness"][channel] = []
                        self.real_time_metrics["channel_effectiveness"][channel].append(sentiment)
            
            # Update last hour engagement
            self.real_time_metrics["last_hour_engagement"] += 1
            
            # Update metrics
            self.update_variant_metrics(variant_key)
    
    def update_variant_metrics(self, variant_key):
        """Update all metrics for a variant with real-time performance tracking"""
        variant = self.message_variants[variant_key]
        current_hour = datetime.now().hour
        
        # Update conversion rate
        if variant["sent"] > 0:
            variant["conversion_rate"] = (variant["confirmed"] / variant["sent"]) * 100
        
        # Calculate engagement score with channel effectiveness
        response_time_score = self.normalize_response_time(variant["avg_response_time"])
        sentiment_score = self.calculate_avg_sentiment(variant["sentiment_scores"])
        completion_rate = variant["conversion_rate"] / 100
        
        # Calculate channel effectiveness
        channel_effectiveness = 0
        if variant_key in self.real_time_metrics["channel_effectiveness"]:
            channel_scores = self.real_time_metrics["channel_effectiveness"][variant_key]
            if channel_scores:
                channel_effectiveness = sum(channel_scores) / len(channel_scores)
        
        # Calculate weighted engagement score
        variant["engagement_score"] = (
            self.engagement_weights["response_time"] * response_time_score +
            self.engagement_weights["sentiment_score"] * sentiment_score +
            self.engagement_weights["completion_rate"] * completion_rate +
            self.engagement_weights["channel_effectiveness"] * channel_effectiveness
        )
        
        # Update real-time metrics
        if 9 <= current_hour <= 18:
            self.real_time_metrics["peak_hour_performance"][variant_key] = \
                self.real_time_metrics["peak_hour_performance"].get(variant_key, 0) + 1
    
    def normalize_response_time(self, response_time):
        """Normalize response time to a 0-1 scale"""
        if response_time <= 0:
            return 1.0
        # Assume 5 minutes is optimal response time
        optimal_time = 300  # seconds
        return max(0, min(1, optimal_time / response_time))
    
    def calculate_avg_sentiment(self, sentiment_scores):
        """Calculate average sentiment score"""
        if not sentiment_scores:
            return 0.5
        return sum(sentiment_scores) / len(sentiment_scores)
    
    def get_best_performing_variant(self, message_type, language, template_type):
        """Get best variant based on engagement score"""
        relevant_variants = {}
        
        for key, data in self.message_variants.items():
            if key.startswith(f"{message_type}_{language}_{template_type}_"):
                relevant_variants[key] = data["engagement_score"]
        
        if relevant_variants:
            return max(relevant_variants.items(), key=lambda x: x[1])[0]
        
        return None
    
    def generate_report(self):
        """Generate enhanced performance report with ML insights"""
        print("\n📊 Advanced A/B Testing Performance Report")
        print("-" * 60)
        
        for key, data in self.message_variants.items():
            parts = key.split("_")
            message_type, language, template_type, variant_id = parts
            
            print(f"Variant: {variant_id} | Type: {message_type} | Language: {language}")
            print(f"Template: {template_type} | Engagement Score: {data['engagement_score']:.2f}")
            print(f"Metrics: Sent={data['sent']}, Confirmed={data['confirmed']}, ")
            print(f"Rate={data['conversion_rate']:.2f}%, Avg Response Time={data['avg_response_time']:.1f}s")
            print("-" * 40)
        
        # Print ML-driven insights
        self.print_ml_insights()
    
    def print_ml_insights(self):
        """Print machine learning derived insights"""
        print("\n🤖 ML-Driven Insights:")
        
        # Analyze performance patterns
        best_variants = {}
        for key, data in self.message_variants.items():
            parts = key.split("_")
            message_type = parts[0]
            if message_type not in best_variants or \
               data["engagement_score"] > best_variants[message_type]["score"]:
                best_variants[message_type] = {
                    "variant": key,
                    "score": data["engagement_score"]
                }
        
        for message_type, info in best_variants.items():
            print(f"\n📱 {message_type.capitalize()}:")
            print(f"Best Performing Variant: {info['variant']}")
            print(f"Engagement Score: {info['score']:.2f}")


# Enhanced message sending function with template formatting
def send_message(patient, message_type, ab_testing=None, test_variant=None):
    """Send a message to a patient using their preferred channel and language"""
    language = patient["language"]
    
    # Select optimal channel based on patient and message type
    channel = select_optimal_channel(patient, message_type)
    
    # Select template variant based on patient demographics
    template_variant = select_template_variant(patient, message_type)
    
    # Get the appropriate message template
    if language not in message_templates[message_type]:
        language = "English"  # Default to English if language not available
    
    message_template = message_templates[message_type][language][template_variant]
    
    # Format the message with appropriate variables
    formatted_message = message_template
    
    # Replace placeholders with actual values based on message type
    if message_type == "appointment_confirmation":
        tomorrow = datetime.now().strftime("%Y-%m-%d")
        formatted_message = formatted_message.format(date=tomorrow, time="10:00 AM")
    elif message_type == "wait_time":
        wait_time = random.randint(5, 30)
        formatted_message = formatted_message.format(wait_time=wait_time)
    elif message_type == "prescription_reminder":
        medicines = ["Metformin", "Atorvastatin", "Lisinopril", "Amlodipine", "Omeprazole"]
        medicine = random.choice(medicines)
        next_time = "6:00 PM"
        formatted_message = formatted_message.format(medicine=medicine, next_time=next_time)
    
    # Track in A/B testing framework if provided
    variant_key = None
    if ab_testing and test_variant:
        variant_key = f"{message_type}_{language}_{template_variant}_{test_variant}"
        ab_testing.track_send(variant_key)
    
    # Print the message being sent
    print(f"📩 Sending via {channel} to {patient['name']} ({language}, {patient['age']} years): {formatted_message}")
    
    return variant_key

# Simulate patient response (confirmation)
def simulate_patient_response(patient, message_type, variant_key=None, ab_testing=None):
    """Simulate patient response to a message"""
    # Base confirmation probability
    base_probability = 0.5
    
    # Adjust probability based on channel appropriateness
    channel = patient["channel"]
    age = patient["age"]
    
    # Elderly patients respond better to IVR
    if age >= 65 and channel == "IVR":
        base_probability += 0.2
    # Younger patients respond better to WhatsApp
    elif age < 65 and channel == "WhatsApp":
        base_probability += 0.15
    # Language-matched messages improve response rates
    if patient["language"] in message_templates[message_type]:
        base_probability += 0.1
    
    # Simulate confirmation based on probability
    confirmed = random.random() < base_probability
    
    # Track in A/B testing framework if provided
    if ab_testing and variant_key:
        ab_testing.track_confirmation(variant_key, confirmed)
    
    return confirmed

# Main function to demonstrate the solution
def main():
    print("\n🏥 Apollo Clinic Multi-Language Patient Communication System 🏥")
    print("=" * 70)
    
    # Initialize advanced systems with enhanced analytics
    ab_testing = AdvancedABTestingFramework()
    channel_optimizer = ChannelOptimizer()
    message_optimizer = MessageOptimizer()
    
    # Use the enhanced communication manager with advanced effectiveness measurement
    communication_manager = EnhancedCommunicationManager()
    
    # Initialize real-time analytics dashboard with enhanced metrics
    analytics_dashboard = {
        "demographic_metrics": {
            "age_groups": {"<30": {}, "30-50": {}, "50-65": {}, ">65": {}},
            "languages": {lang: {} for lang in ["Tamil", "Telugu", "Malayalam", "Hindi", "English"]},
            "channels": {"SMS": {}, "WhatsApp": {}, "IVR": {}}
        },
        "message_effectiveness": {
            "response_times": [],
            "engagement_rates": [],
            "sentiment_scores": [],
            "conversion_rates": {}
        },
        "ai_insights": {
            "optimal_times": {},
            "channel_preferences": {},
            "language_effectiveness": {}
        }
    }
    
    # Initialize analytics dashboard
    analytics = {
        "demographic_metrics": {
            "age_groups": {"<30": {}, "30-50": {}, "50-65": {}, ">65": {}},
            "languages": {lang: {} for lang in ["Tamil", "Telugu", "Malayalam", "Hindi", "English"]}
        },
        "channel_performance": {channel: {} for channel in ["SMS", "WhatsApp", "IVR"]},
        "real_time_metrics": {
            "response_times": [],
            "engagement_rates": [],
            "sentiment_scores": []
        }
    }
    
    # Register enhanced message variants
    for message_type in message_templates.keys():
        for language in ["Tamil", "Telugu", "Malayalam", "Hindi", "English"]:
            for template_type in ["standard", "elderly"]:
                ab_testing.register_variant("A", message_type, language, template_type)
                ab_testing.register_variant("B", message_type, language, template_type)
    
    def get_preferred_channel(age_group):
        """Get preferred channel for age group based on historical data"""
        channels = {"SMS": 0, "WhatsApp": 0, "IVR": 0}
        for patient in patients:
            age = patient["age"]
            patient_age_group = "<30" if age < 30 else "30-50" if age < 50 else "50-65" if age < 65 else ">65"
            if patient_age_group == age_group:
                channels[patient["channel"]] += 1
        return max(channels.items(), key=lambda x: x[1])[0]

    def get_optimal_time(age_group):
        """Get optimal message time for age group"""
        if age_group == ">65":
            return "9:00 AM - 11:00 AM"
        elif age_group == "<30":
            return "6:00 PM - 8:00 PM"
        return "12:00 PM - 2:00 PM"

    def get_language_sentiment(language):
        """Calculate average sentiment score for a language"""
        scores = analytics["real_time_metrics"]["sentiment_scores"]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_message_effectiveness(language):
        """Get message effectiveness rating for a language"""
        metrics = analytics["demographic_metrics"]["languages"][language]
        total_sent = sum(m["sent"] for m in metrics.values())
        total_confirmed = sum(m["confirmed"] for m in metrics.values())
        if total_sent == 0:
            return "No data available"
        effectiveness = (total_confirmed / total_sent) * 100
        if effectiveness >= 80:
            return "Excellent"
        elif effectiveness >= 60:
            return "Good"
        elif effectiveness >= 40:
            return "Fair"
        return "Needs Improvement"

    def get_channel_effectiveness(channel):
        """Calculate effectiveness score for a channel"""
        metrics = analytics["demographic_metrics"]["channels"].get(channel, {})
        if not metrics:
            return 0.0
        return sum(metrics.values()) / len(metrics)

    def get_channel_optimal_time(channel):
        """Get optimal time for each channel based on success rates"""
        if channel == "IVR":
            return "9:00 AM - 5:00 PM"
        elif channel == "WhatsApp":
            return "24/7"
        return "8:00 AM - 8:00 PM"

    def get_top_recommendation():
        """Get top AI-driven recommendation"""
        return "Personalize message timing based on age group and preferred channel"

    def get_optimization_suggestion():
        """Get AI-driven optimization suggestion"""
        return "Increase IVR usage for elderly patients during morning hours"

    def get_trend_analysis():
        """Get AI-driven trend analysis"""
        return "WhatsApp showing highest engagement rates among patients under 50"

    def update_analytics(patient, message_type, response_time, confirmed, sentiment=0):
        """Update analytics dashboard with enhanced real-time metrics"""
        # Update demographic metrics
        age = patient["age"]
        age_group = "<30" if age < 30 else "30-50" if age < 50 else "50-65" if age < 65 else ">65"
        
        demo_metrics = analytics["demographic_metrics"]
        if message_type not in demo_metrics["age_groups"][age_group]:
            demo_metrics["age_groups"][age_group][message_type] = {"sent": 0, "confirmed": 0}
        demo_metrics["age_groups"][age_group][message_type]["sent"] += 1
        if confirmed:
            demo_metrics["age_groups"][age_group][message_type]["confirmed"] += 1
        
        # Update language metrics with sentiment analysis
        lang_metrics = demo_metrics["languages"][patient["language"]]
        if message_type not in lang_metrics:
            lang_metrics[message_type] = {"sent": 0, "confirmed": 0, "sentiment": []}
        lang_metrics[message_type]["sent"] += 1
        if confirmed:
            lang_metrics[message_type]["confirmed"] += 1
        lang_metrics[message_type]["sentiment"].append(sentiment)
        
        # Update channel metrics
        channel_metrics = demo_metrics["channels"][patient["channel"]]
        if message_type not in channel_metrics:
            channel_metrics[message_type] = {"effectiveness": 0.0, "optimal_time": None}
        
        # Update real-time metrics and AI insights
        analytics["real_time_metrics"]["response_times"].append(response_time)
        analytics["real_time_metrics"]["engagement_rates"].append(1 if confirmed else 0)
        analytics["real_time_metrics"]["sentiment_scores"].append(sentiment)
        
        # Update AI insights
        analytics["ai_insights"]["optimal_times"][age_group] = get_optimal_time(age_group)
        analytics["ai_insights"]["channel_preferences"][patient["channel"]] = get_channel_effectiveness(patient["channel"])
        analytics["ai_insights"]["language_effectiveness"][patient["language"]] = get_message_effectiveness(patient["language"])
    
    def print_analytics_dashboard():
        """Display real-time analytics dashboard with enhanced metrics"""
        print("\n📊 Advanced Analytics Dashboard")
        print("-" * 60)
        
        # Display demographic performance with enhanced insights
        print("\n👥 Demographic Performance:")
        for age_group, metrics in analytics["demographic_metrics"]["age_groups"].items():
            if metrics:
                total_sent = sum(m["sent"] for m in metrics.values())
                total_confirmed = sum(m["confirmed"] for m in metrics.values())
                if total_sent > 0:
                    success_rate = (total_confirmed / total_sent) * 100
                    print(f"Age {age_group}:")
                    print(f"  - Success Rate: {success_rate:.1f}%")
                    print(f"  - Preferred Channel: {get_preferred_channel(age_group)}")
                    print(f"  - Best Time: {get_optimal_time(age_group)}")
        
        # Display language performance with sentiment analysis
        print("\n🌐 Language Performance:")
        for language, metrics in analytics["demographic_metrics"]["languages"].items():
            if metrics:
                total_sent = sum(m["sent"] for m in metrics.values())
                total_confirmed = sum(m["confirmed"] for m in metrics.values())
                if total_sent > 0:
                    success_rate = (total_confirmed / total_sent) * 100
                    sentiment = get_language_sentiment(language)
                    print(f"{language}:")
                    print(f"  - Success Rate: {success_rate:.1f}%")
                    print(f"  - Sentiment Score: {sentiment:.2f}")
                    print(f"  - Message Effectiveness: {get_message_effectiveness(language)}")
        
        # Display channel performance
        print("\n📱 Channel Performance:")
        for channel, metrics in analytics["demographic_metrics"]["channels"].items():
            if metrics:
                effectiveness = get_channel_effectiveness(channel)
                print(f"{channel}:")
                print(f"  - Effectiveness Score: {effectiveness:.2f}")
                print(f"  - Best Time: {get_channel_optimal_time(channel)}")
        
        # Display real-time metrics with AI insights
        rt_metrics = analytics["real_time_metrics"]
        if rt_metrics["response_times"]:
            avg_response_time = sum(rt_metrics["response_times"]) / len(rt_metrics["response_times"])
            avg_engagement = sum(rt_metrics["engagement_rates"]) / len(rt_metrics["engagement_rates"]) * 100
            avg_sentiment = sum(rt_metrics["sentiment_scores"]) / len(rt_metrics["sentiment_scores"])
            
            print("\n⚡ Real-time Metrics & AI Insights:")
            print(f"Average Response Time: {avg_response_time:.1f} seconds")
            print(f"Engagement Rate: {avg_engagement:.1f}%")
            print(f"Average Sentiment Score: {avg_sentiment:.2f}")
            
            # Display AI-driven recommendations
            print("\n🤖 AI Recommendations:")
            print(f"- {get_top_recommendation()}")
            print(f"- {get_optimization_suggestion()}")
            print(f"- {get_trend_analysis()}")
    
    # Process messages with enhanced tracking
    print("\n📱 Processing Communications...")
    for message_type in ["appointment_confirmation", "wait_time", "prescription_reminder"]:
        print(f"\n📨 Sending {message_type.replace('_', ' ').title()}...")
        for patient in patients:
            # Get optimal channel and timing
            channel = select_optimal_channel(patient, message_type)
            optimal_time = message_optimizer.get_optimal_send_time(patient["id"], message_type)
            
            # Send message with A/B testing
            variant = random.choice(["A", "B"])
            variant_key = send_message(patient, message_type, ab_testing, variant)
            
            # Simulate response with enhanced metrics
            response_time = random.uniform(1, 300)  # 1-300 seconds
            confirmed = simulate_patient_response(patient, message_type, variant_key, ab_testing)
            
            # Generate simulated response text for sentiment analysis
            response_text = "Thank you, very helpful!" if confirmed else "Message unclear, confused"
            sentiment = 1 if confirmed else -1
            
            # Update all tracking systems
            message_optimizer.analyze_sentiment(patient["id"], message_type, response_text)
            message_optimizer.update_timing_preference(patient["id"], message_type, datetime.now(), confirmed)
            channel_optimizer.update_engagement_history(patient["id"], channel, message_type, confirmed)
            
            # Update analytics
            update_analytics(patient, message_type, response_time, confirmed, sentiment)
            
            # Display status
            status = "✅ Confirmed" if confirmed else "❌ No response"
            print(f"  {patient['name']} ({patient['language']}): {status} via {channel}")
    
    # Generate comprehensive reports
    print("\n📈 System Performance Analysis")
    print("=" * 60)
    
    # Display A/B testing results
    ab_testing.generate_report()
    
    # Display analytics dashboard
    print_analytics_dashboard()
    
    # Display enhanced effectiveness measurement system reports
    print("\n🔍 Advanced Effectiveness Measurement System")
    print("=" * 60)
    
    # Get enhanced performance metrics
    enhanced_metrics = communication_manager.get_enhanced_system_performance()
    
    # Display key effectiveness metrics
    print("\n📊 Channel Effectiveness Index:")
    for channel, metrics in enhanced_metrics['advanced_metrics']['channel_effectiveness'].items():
        print(f"{channel}: {metrics['score']:.2f} effectiveness score")
        print(f"  - Delivery Rate: {metrics['delivery_rate']:.2%}")
        print(f"  - Response Rate: {metrics['response_rate']:.2%}")
    
    # Display demographic performance analysis
    print("\n👥 Demographic Performance Analysis:")
    print("Language Performance:")
    for language, metrics in enhanced_metrics['advanced_metrics']['demographic_performance']['language'].items():
        if metrics['effectiveness_score'] > 0:
            print(f"  {language}: {metrics['effectiveness_score']:.2f} score, {metrics['response_rate']:.2%} response rate")
    
    print("\nAge Group Performance:")
    for age_group, metrics in enhanced_metrics['advanced_metrics']['demographic_performance']['age_group'].items():
        if metrics['effectiveness_score'] > 0:
            print(f"  {age_group}: {metrics['effectiveness_score']:.2f} score, {metrics['response_rate']:.2%} response rate")
    
    # Display real-time monitoring metrics
    print("\n⚡ Real-time Performance Monitoring:")
    rt_metrics = enhanced_metrics['advanced_metrics']['real_time_metrics']
    
    print("Channel Health Status:")
    for channel, health in rt_metrics['channel_health'].items():
        print(f"  {channel}: {health['status']} (Uptime: {health['uptime']:.1f}%, Error Rate: {health['error_rate']:.2%})")
    
    # Display delivery status tracking
    delivery_stats = communication_manager.get_delivery_status_tracking()
    print("\n📬 Live Delivery Status Tracking:")
    for channel, stats in delivery_stats.items():
        if stats['total_sent'] > 0:
            print(f"  {channel}: {stats['success_rate']:.2%} success rate ({stats['total_delivered']}/{stats['total_sent']} messages)")
    
    # Display response time analytics
    response_analytics = communication_manager.get_response_time_analytics()
    print("\n⏱️ Response Time Analytics:")
    for channel, analytics in response_analytics.items():
        if analytics['sample_size'] > 0:
            print(f"  {channel}: Avg {analytics['average']:.1f}s, Median {analytics['median']:.1f}s, 90th %ile {analytics['p90']:.1f}s")
    
    # Generate visualization
    print("\n🖼️ Generating Effectiveness Visualization...")
    visualization_path = communication_manager.generate_effectiveness_visualization("effectiveness_dashboard.png")
    if visualization_path:
        print(f"Visualization saved to: {visualization_path}")
    else:
        print("Visualization displayed (not saved)")
    
    print("\n🚀 Advanced AI Features:")
    print("1. ML-driven channel optimization with real-time learning")
    print("2. Sentiment analysis for message personalization")
    print("3. Demographic-based performance tracking")
    print("4. Optimal timing prediction for each patient")
    print("5. Advanced analytics dashboard with real-time metrics")

if __name__ == "__main__":
    main()