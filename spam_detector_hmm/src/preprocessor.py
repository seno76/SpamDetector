"""
Модуль для предобработки текстов и преобразования в последовательности для HMM
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
import numpy as np

class TextPreprocessor:
    """
    Класс для предобработки текстов и конвертации в последовательности признаков
    """
    
    def __init__(self, feature_type='pos', n_symbols=50):
        """
        Args:
            feature_type: тип признаков ('pos' для частей речи, 'word_clusters' для кластеров слов)
            n_symbols: количество уникальных символов в алфавите наблюдений
        """
        self.feature_type = feature_type
        self.n_symbols = n_symbols
        self.vocab = {}  # Словарь для маппинга признаков в числа
        self.reverse_vocab = {}  # Обратный словарь
        self.is_fitted = False
        
        # Инициализация NLTK компонентов
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Базовая очистка текста
        
        Args:
            text: исходный текст
        Returns:
            очищенный текст
        """
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление специальных символов, оставляя только буквы и пробелы
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Удаление множественных пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Токенизация текста
        
        Args:
            text: текст для токенизации
        Returns:
            список токенов
        """
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        
        # Удаление стоп-слов и коротких токенов
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def extract_pos_features(self, text):
        """
        Извлечение последовательности POS-тегов (частей речи)
        
        Args:
            text: исходный текст
        Returns:
            список POS-тегов
        """
        tokens = self.tokenize(text)
        
        if not tokens:
            return ['UNK']  # Неизвестный токен для пустых текстов
        
        # Получение POS-тегов
        pos_tags = pos_tag(tokens)
        
        # Упрощение тегов (берем первые 2 символа)
        simplified_tags = [tag[:2] for word, tag in pos_tags]
        
        return simplified_tags
    
    def build_vocabulary(self, texts):
        """
        Построение словаря признаков из корпуса текстов
        
        Args:
            texts: список текстов для анализа
        """
        all_features = []
        
        for text in texts:
            if self.feature_type == 'pos':
                features = self.extract_pos_features(text)
            else:
                features = self.tokenize(text)
            
            all_features.extend(features)
        
        # Подсчет частоты признаков
        feature_counts = Counter(all_features)
        
        # Берем N самых частых признаков
        most_common = feature_counts.most_common(self.n_symbols - 1)
        
        # Создание словаря: признак -> число
        self.vocab = {feature: idx for idx, (feature, _) in enumerate(most_common)}
        self.vocab['UNK'] = len(self.vocab)  # Неизвестный символ
        
        # Обратный словарь
        self.reverse_vocab = {idx: feature for feature, idx in self.vocab.items()}
        
        self.is_fitted = True
        
        print(f"✓ Словарь построен: {len(self.vocab)} уникальных признаков")
        print(f"  Топ-10 признаков: {list(self.vocab.keys())[:10]}")
    
    def text_to_sequence(self, text):
        """
        Преобразование текста в последовательность чисел для HMM
        
        Args:
            text: исходный текст
        Returns:
            numpy array с последовательностью индексов
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor не обучен! Вызовите build_vocabulary() сначала.")
        
        if self.feature_type == 'pos':
            features = self.extract_pos_features(text)
        else:
            features = self.tokenize(text)
        
        # Маппинг признаков в числа
        sequence = [
            self.vocab.get(feature, self.vocab['UNK']) 
            for feature in features
        ]
        
        return np.array(sequence, dtype=np.int32)
    
    def texts_to_sequences(self, texts):
        """
        Преобразование списка текстов в последовательности
        
        Args:
            texts: список текстов
        Returns:
            список numpy arrays
        """
        return [self.text_to_sequence(text) for text in texts]
    
    def get_vocabulary_size(self):
        """Возвращает размер словаря"""
        return len(self.vocab) if self.is_fitted else 0
