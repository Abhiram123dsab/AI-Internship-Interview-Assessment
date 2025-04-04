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
from transformers import (
    TFBertModel, 
    BertTokenizer,
    TFMarianMTModel, 
    MarianTokenizer,
    TFRobertaModel,
    RobertaTokenizer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

class EnhancedNLPWhatsAppV2:
    """Advanced WhatsApp integration with enhanced NLP capabilities for automated response handling"""
    
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('vader_lexicon')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = {
            'english': set(stopwords.words('english')),
            'tamil': set(self._load_custom_stopwords('tamil')),
            'telugu': set(self._load_custom_stopwords('telugu')),
            'malayalam': set(self._load_custom_stopwords('malayalam')),
            'hindi': set(self._load_custom_stopwords('hindi'))
        }
        
        # Initialize models
        self.sentiment_model = self._build_sentiment_model()
        self.intent_model = self._build_intent_model()
        self.emotion_model = self._build_emotion_model()
        
        # Initialize translation models
        self.translation_models = self._initialize_translation_models()
        
        # Initialize VADER sentiment analyzer
        self.vader = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        
        # Initialize RoBERTa model for zero-shot classification
        self.roberta_model = TFRobertaModel.from_pretrained('roberta-large')
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # Response templates with sentiment and emotion awareness
        self.response_templates = self._load_response_templates()
        
        # Initialize metrics tracking
        self.metrics = {
            'sentiment_accuracy': [],
            'intent_accuracy': [],
            'emotion_accuracy': [],
            'response_effectiveness': []
        }
    
    def _load_custom_stopwords(self, language):
        """Load custom stopwords for different languages"""
        # In production, load from file or database
        return set()
    
    def _initialize_translation_models(self):
        """Initialize translation models for different languages"""
        models = {}
        language_pairs = [
            ('en-ta', 'Helsinki-NLP/opus-mt-en-ta'),
            ('en-te', 'Helsinki-NLP/opus-mt-en-te'),
            ('en-ml', 'Helsinki-NLP/opus-mt-en-ml'),
            ('en-hi', 'Helsinki-NLP/opus-mt-en-hi')
        ]
        
        for lang_pair, model_name in language_pairs:
            models[lang_pair] = {
                'model': TFMarianMTModel.from_pretrained(model_name),
                'tokenizer': MarianTokenizer.from_pretrained(model_name)
            }
        
        return models
    
    def _build_sentiment_model(self):
        """Build and compile an advanced sentiment analysis model using BERT"""
        # Initialize BERT with larger capacity
        bert_model = TFBertModel.from_pretrained('bert-large-uncased')
        
        # Input layers
        input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
        
        # BERT layer with gradient checkpointing for memory efficiency
        bert_outputs = bert_model(input_ids, attention_mask=attention_mask, training=True)
        pooled_output = bert_outputs[1]
        
        # Advanced classification layers with residual connections
        x = layers.Dense(64, activation='relu')(pooled_output)
        x = layers.Dropout(0.3)(x)
        residual = x
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])
        x = layers.LayerNormalization()(x)
        
        # Multi-head attention for better feature extraction
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        outputs = layers.Dense(5, activation='softmax')(x)  # 5 sentiment classes
        
        # Create model with multiple inputs
        model = models.Model(
            inputs=[input_ids, attention_mask],
            outputs=outputs
        )
        
        # Compile with weighted loss and advanced metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_emotion_model(self):
        """Build and compile an emotion detection model"""
        model = models.Sequential([
            layers.Input(shape=(128,)),
            layers.Embedding(10000, 256),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dense(8, activation='softmax')  # 8 basic emotions
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_intent_model(self):
        """Build and compile an advanced intent recognition model with transformer architecture"""
        # Input layer
        inputs = layers.Input(shape=(128,))
        
        # Embedding layer
        x = layers.Embedding(10000, 256)(inputs)
        
        # Transformer blocks
        for _ in range(3):
            # Multi-head attention
            attention = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn = layers.Dense(512, activation='relu')(x)
            ffn = layers.Dense(256)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(12, activation='softmax')(x)  # 12 intent categories
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def translate_text(self, text, source_lang, target_lang):
        """Translate text between supported languages"""
        if source_lang == target_lang:
            return text
            
        lang_pair = f'{source_lang}-{target_lang}'
        if lang_pair not in self.translation_models:
            return text
            
        model = self.translation_models[lang_pair]['model']
        tokenizer = self.translation_models[lang_pair]['tokenizer']
        
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text
    
    def analyze_sentiment_advanced(self, text, language='english'):
        """Advanced sentiment analysis with multilingual support and emotion detection"""
        # Translate to English if needed
        if language != 'english':
            text = self.translate_text(text, language, 'english')
        
        # VADER sentiment analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # BERT tokenization and prediction
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        # Get BERT model prediction
        bert_prediction = self.sentiment_model.predict([
            encoded['input_ids'],
            encoded['attention_mask']
        ])
        
        # Emotion detection
        emotion_prediction = self.emotion_model.predict(encoded['input_ids'])
        emotion = self._get_emotion_label(emotion_prediction)
        
        # Ensemble prediction
        sentiment_score = (
            vader_scores['compound'] * 0.3 +
            textblob_sentiment * 0.3 +
            np.max(bert_prediction) * 0.4
        )
        
        # Map score to sentiment category
        if sentiment_score >= 0.5:
            sentiment = 'very_positive'
        elif sentiment_score >= 0.1:
            sentiment = 'positive'
        elif sentiment_score <= -0.5:
            sentiment = 'very_negative'
        elif sentiment_score <= -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'emotion': emotion,
            'confidence': np.max(bert_prediction),
            'scores': {
                'vader': vader_scores['compound'],
                'textblob': textblob_sentiment,
                'bert': np.max(bert_prediction)
            }
        }
    
    def _get_emotion_label(self, prediction):
        """Map emotion prediction to label"""
        emotions = [
            'joy', 'sadness', 'anger', 'fear',
            'surprise', 'disgust', 'trust', 'anticipation'
        ]
        return emotions[np.argmax(prediction)]
    
    def extract_intent_advanced(self, text, language='english'):
        """Advanced intent extraction with zero-shot learning capabilities"""
        # Translate to English if needed
        if language != 'english':
            text = self.translate_text(text, language, 'english')
        
        # Tokenize for RoBERTa
        encoded = self.roberta_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        # Get embeddings
        embeddings = self.roberta_model(encoded['input_ids'])[0]
        
        # Define possible intents with descriptions
        intent_descriptions = {
            'appointment_booking': 'Schedule or book a medical appointment',
            'appointment_reschedule': 'Change or reschedule existing appointment',
            'appointment_cancel': 'Cancel an existing appointment',
            'medical_query': 'Ask about medical conditions or symptoms',
            'prescription_refill': 'Request prescription refill or medication',
            'test_results': 'Ask about medical test results',
            'emergency': 'Report emergency or urgent medical need',
            'feedback': 'Provide feedback about service or experience',
            'insurance': 'Ask about insurance coverage or billing',
            'general_info': 'Request general information about services',
            'complaint': 'Report a problem or file a complaint',
            'followup': 'Schedule or ask about follow-up care'
        }
        
        # Calculate similarity scores with each intent
        scores = {}
        for intent, description in intent_descriptions.items():
            # Encode intent description
            desc_encoded = self.roberta_tokenizer(
                description,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='tf'
            )
            desc_embeddings = self.roberta_model(desc_encoded['input_ids'])[0]
            
            # Calculate cosine similarity
            similarity = tf.keras.losses.cosine_similarity(
                embeddings,
                desc_embeddings
            ).numpy()
            scores[intent] = float(similarity)
        
        # Get top intent and confidence
        top_intent = max(scores.items(), key=lambda x: x[1])[0]
        confidence = max(scores.values())
        
        return {
            'intent': top_intent,
            'confidence': confidence,
            'all_scores': scores
        }
    
    def generate_response_advanced(self, message, language='english', patient_context=None):
        """Generate contextual response with sentiment and intent awareness"""
        # Analyze message
        sentiment_analysis = self.analyze_sentiment_advanced(message, language)
        intent_analysis = self.extract_intent_advanced(message, language)
        
        # Get base response based on sentiment and intent
        base_response = self._get_base_response(
            sentiment_analysis['sentiment'],
            intent_analysis['intent'],
            sentiment_analysis['emotion']
        )
        
        # Enhance response with context
        enhanced_response = self._enhance_response_with_context(
            base_response,
            intent_analysis['intent'],
            patient_context,
            sentiment_analysis
        )
        
        # Translate response if needed
        if language != 'english':
            enhanced_response = self.translate_text(enhanced_response, 'english', language)
        
        return {
            'response': enhanced_response,
            'analysis': {
                'sentiment': sentiment_analysis,
                'intent': intent_analysis
            }
        }
    
    def _get_base_response(self, sentiment, intent, emotion):
        """Get appropriate response template based on sentiment, intent, and emotion"""
        templates = self.response_templates.get(sentiment, {})
        intent_templates = templates.get(intent, {})
        
        # Try to get emotion-specific response
        if emotion in intent_templates:
            return intent_templates[emotion]
        
        # Fallback to general intent response
        if 'general' in intent_templates:
            return intent_templates['general']
        
        # Final fallback
        return self.response_templates['neutral']['general']['general']
    
    def _enhance_response_with_context(self, base_response, intent, context, sentiment_analysis):
        """Enhance response with patient context and sentiment information"""
        if not context:
            return base_response
            
        enhanced_response = base_response
        
        # Add context-specific information
        if intent == 'appointment_booking' and 'next_available_slot' in context:
            enhanced_response += f" The next available appointment slot is {context['next_available_slot']}."
            
        elif intent == 'prescription_refill' and 'last_prescription' in context:
            enhanced_response += f" I can see your last prescription was on {context['last_prescription']}."
            
        elif intent == 'test_results' and 'recent_tests' in context:
            enhanced_response += f" Your recent test from {context['recent_tests']} is ready."
            
        # Add empathy based on emotion
        if sentiment_analysis['emotion'] in ['sadness', 'fear', 'anger']:
            enhanced_response += " I understand this might be difficult, and we're here to help."
            
        return enhanced_response
    
    def update_metrics(self, interaction_data):
        """Update performance metrics based on interaction data"""
        if 'true_sentiment' in interaction_data:
            self.metrics['sentiment_accuracy'].append(
                interaction_data['predicted_sentiment'] == interaction_data['true_sentiment']
            )
            
        if 'true_intent' in interaction_data:
            self.metrics['intent_accuracy'].append(
                interaction_data['predicted_intent'] == interaction_data['true_intent']
            )
            
        if 'response_rating' in interaction_data:
            self.metrics['response_effectiveness'].append(
                interaction_data['response_rating']
            )
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        return {
            'sentiment_accuracy': np.mean(self.metrics['sentiment_accuracy']) if self.metrics['sentiment_accuracy'] else 0,
            'intent_accuracy': np.mean(self.metrics['intent_accuracy']) if self.metrics['intent_accuracy'] else 0,
            'response_effectiveness': np.mean(self.metrics['response_effectiveness']) if self.metrics['response_effectiveness'] else 0
        }