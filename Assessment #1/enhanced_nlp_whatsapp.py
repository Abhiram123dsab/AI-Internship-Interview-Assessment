import tensorflow as tf
from tensorflow.keras import layers, models
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from datetime import datetime
import json
from transformers import TFBertModel, BertTokenizer
from sklearn.preprocessing import LabelEncoder

class EnhancedNLPWhatsApp:
    """Enhanced WhatsApp integration with NLP capabilities for automated response handling"""
    
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('vader_lexicon')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize models
        self.sentiment_model = self._build_sentiment_model()
        self.intent_model = self._build_intent_model()
        
        # Initialize VADER sentiment analyzer
        self.vader = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        
        # Response templates with sentiment awareness
        self.response_templates = {
            'positive': {
                'feedback': "Thank you for your positive feedback! We're glad you had a great experience.",
                'appointment': "Great to hear from you! We're looking forward to your appointment.",
                'general': "Thank you for your message! We're happy to help."
            },
            'neutral': {
                'feedback': "Thank you for your feedback. Your input helps us improve.",
                'appointment': "Thank you for confirming your appointment.",
                'general': "Thank you for your message. How can we assist you?"
            },
            'negative': {
                'feedback': "We apologize for any inconvenience. Your feedback is important to us.",
                'appointment': "We understand your concern. Let us help you with your appointment.",
                'general': "We apologize for any issues. Please let us know how we can help."
            }
        }
    
    def _build_sentiment_model(self):
        """Build and compile an advanced sentiment analysis model using BERT"""
        # Initialize BERT
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        
        # Input layers
        input_ids = layers.Input(shape=(100,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(100,), dtype=tf.int32, name='attention_mask')
        
        # BERT layer
        bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs[1]
        
        # Classification layers
        x = layers.Dense(32, activation='relu')(pooled_output)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        # Create model
        model = models.Model(
            inputs=[input_ids, attention_mask],
            outputs=outputs
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_text(self, text):
        """Preprocess text for NLP analysis with enhanced features"""
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        
        # Remove stopwords and lemmatize with POS info
        tokens = [
            self.lemmatizer.lemmatize(token, self._get_wordnet_pos(tag))
            for token, tag in pos_tags
            if token not in self.stop_words and token.isalnum()
        ]
        
        return ' '.join(tokens)
    
    def _get_wordnet_pos(self, treebank_tag):
        """Convert POS tag to WordNet POS tag for better lemmatization"""
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN
    
    def _build_intent_model(self):
        """Build and compile an advanced intent recognition model"""
        model = models.Sequential([
            layers.Input(shape=(100,)),
            layers.Embedding(10000, 128),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(8, activation='softmax')  # 8 intent categories
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def analyze_sentiment(self, text):
        """Analyze sentiment using an ensemble of BERT, VADER, TextBlob, and custom features"""
        # VADER sentiment analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # BERT tokenization and prediction
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = tokenizer(text, padding='max_length', truncation=True, max_length=100, return_tensors='tf')
        
        # Get BERT model prediction
        bert_prediction = self.sentiment_model.predict([
            encoded['input_ids'],
            encoded['attention_mask']
        ])
        
        # Combine predictions
        bert_sentiment = np.argmax(bert_prediction)
        
        # Weighted ensemble
        if bert_sentiment == 2:  # BERT predicts positive
            return 'positive' if textblob_sentiment > -0.3 else 'neutral'
        elif bert_sentiment == 0:  # BERT predicts negative
            return 'negative' if textblob_sentiment < 0.3 else 'neutral'
        else:
            return 'neutral' if abs(textblob_sentiment) < 0.5 else ('positive' if textblob_sentiment > 0 else 'negative')
    
    def extract_intent(self, text):
        """Extract user intent from message"""
        text = text.lower()
        
        # Define intent patterns
        intent_patterns = {
            'appointment': ['appointment', 'schedule', 'book', 'visit', 'meet'],
            'feedback': ['feedback', 'rating', 'experience', 'review'],
            'prescription': ['medicine', 'prescription', 'drug', 'medication'],
            'query': ['question', 'help', 'how', 'what', 'when', 'where'],
            'emergency': ['emergency', 'urgent', 'immediately', 'asap']
        }
        
        # Check for each intent
        for intent, patterns in intent_patterns.items():
            if any(pattern in text for pattern in patterns):
                return intent
        
        return 'general'
    
    def generate_response(self, message, patient_context=None):
        """Generate appropriate response based on sentiment and intent"""
        # Analyze message
        sentiment = self.analyze_sentiment(message)
        intent = self.extract_intent(message)
        
        # Get base response template
        base_response = self.response_templates[sentiment].get(
            intent,
            self.response_templates[sentiment]['general']
        )
        
        # Enhance response with context if available
        if patient_context:
            enhanced_response = self._enhance_response_with_context(
                base_response,
                intent,
                patient_context
            )
        else:
            enhanced_response = base_response
        
        return {
            'response_text': enhanced_response,
            'sentiment': sentiment,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        }
    
    def _enhance_response_with_context(self, base_response, intent, context):
        """Enhance response with patient context"""
        if intent == 'appointment':
            if 'next_appointment' in context:
                return f"{base_response} Your next appointment is scheduled for {context['next_appointment']}"
        
        elif intent == 'prescription':
            if 'active_prescriptions' in context:
                return f"{base_response} Your current prescriptions: {', '.join(context['active_prescriptions'])}"
        
        elif intent == 'query':
            if 'common_queries' in context:
                return f"{base_response} You can find more information in our FAQ section."
        
        return base_response
    
    def analyze_conversation_metrics(self, conversation_history):
        """Analyze conversation metrics for insights"""
        metrics = {
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'intent_distribution': {},
            'average_response_time': 0,
            'conversation_quality_score': 0
        }
        
        response_times = []
        prev_timestamp = None
        
        for message in conversation_history:
            # Analyze sentiment and intent
            sentiment = self.analyze_sentiment(message['text'])
            intent = self.extract_intent(message['text'])
            
            # Update distributions
            metrics['sentiment_distribution'][sentiment] += 1
            metrics['intent_distribution'][intent] = \
                metrics['intent_distribution'].get(intent, 0) + 1
            
            # Calculate response time
            if prev_timestamp and message['type'] == 'response':
                response_time = (
                    datetime.fromisoformat(message['timestamp']) -
                    datetime.fromisoformat(prev_timestamp)
                ).total_seconds()
                response_times.append(response_time)
            
            prev_timestamp = message['timestamp']
        
        # Calculate averages and scores
        if response_times:
            metrics['average_response_time'] = sum(response_times) / len(response_times)
        
        # Calculate conversation quality score
        sentiment_score = (
            metrics['sentiment_distribution']['positive'] * 1.0 +
            metrics['sentiment_distribution']['neutral'] * 0.5 +
            metrics['sentiment_distribution']['negative'] * 0.0
        ) / len(conversation_history)
        
        response_time_score = 1.0 if not response_times else \
            max(0, 1 - (sum(response_times) / len(response_times)) / 300)  # 5 minutes benchmark
        
        metrics['conversation_quality_score'] = (
            sentiment_score * 0.7 +
            response_time_score * 0.3
        )
        
        return metrics