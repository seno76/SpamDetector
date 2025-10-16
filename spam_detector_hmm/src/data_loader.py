"""
Загрузка и подготовка данных для обучения
"""
import os
import json
from pathlib import Path

class DataLoader:
    """Класс для загрузки текстовых данных"""
    
    @staticmethod
    def load_texts_from_directory(directory):
        """
        Загрузка всех текстовых файлов из директории
        
        Args:
            directory: путь к директории
        Returns:
            список текстов
        """
        texts = []
        directory = Path(directory)
        
        if not directory.exists():
            print(f"⚠ Директория {directory} не существует")
            return texts
        
        for file_path in directory.glob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text.strip():  # Только непустые
                        texts.append(text)
            except Exception as e:
                print(f"⚠ Ошибка чтения {file_path}: {e}")
        
        return texts
    
    @staticmethod
    def load_sample_data(json_path='data/sample_texts.json'):
        """
        Загрузка примеров из JSON файла
        
        Args:
            json_path: путь к JSON файлу
        Returns:
            словарь с данными
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"⚠️ Файл {json_path} не найден")
            return {'natural': [], 'spam': [], 'test': []}
    
    @staticmethod
    def load_train_data():
        """
        Загрузка обучающих данных
        
        Returns:
            tuple (natural_texts, spam_texts)
        """
        natural_texts = DataLoader.load_texts_from_directory('data/raw/natural')
        spam_texts = DataLoader.load_texts_from_directory('data/raw/spam')
        
        print(f"✓ Загружено обычных текстов: {len(natural_texts)}")
        print(f"✓ Загружено спам-текстов: {len(spam_texts)}")
        
        return natural_texts, spam_texts
    
    @staticmethod
    def load_all_data():
        """
        Загрузка всех типов данных
        
        Returns:
            dict: {'human_natural', 'human_spam', 'markov_spam'}
        """
        human_natural = DataLoader.load_texts_from_directory('data/raw/natural')
        human_spam = DataLoader.load_texts_from_directory('data/raw/spam')
        markov_spam = DataLoader.load_texts_from_directory('data/raw/markov_spam')
        
        # Если нет данных, создаем минимальные примеры
        if not human_natural:
            print("⚠️ Нет natural текстов! Используем примеры...")
            human_natural = [
                "Machine learning is a powerful tool for data analysis and pattern recognition.",
                "Climate change refers to long-term shifts in temperatures and weather patterns.",
                "The Internet has revolutionized communication across the globe.",
                "Renewable energy comes from naturally replenished sources.",
                "Python is a high-level programming language known for simplicity."
            ]
        
        if not human_spam:
            print("⚠️ Нет human spam! Используем примеры...")
            human_spam = [
                "Buy cheap pills online pharmacy discount best price now!",
                "Win money casino gambling bonus free spins jackpot today!",
                "Make money fast easy work from home earn cash quick!",
                "Weight loss pills diet supplement fat burner lose weight!",
                "SEO services optimization ranking google first page backlinks!"
            ]
        
        if not markov_spam:
            print("⚠️ Нет markov spam! Он будет создан при необходимости...")
            markov_spam = []
        
        return {
            'human_natural': human_natural,
            'human_spam': human_spam,
            'markov_spam': markov_spam
        }
    
    @staticmethod
    def get_available_datasets():
        """Проверка доступных датасетов"""
        datasets = {}
        
        paths = {
            'human_natural': 'data/raw/natural',
            'human_spam': 'data/raw/spam',
            'markov_spam': 'data/raw/markov_spam'
        }
        
        for name, path in paths.items():
            if Path(path).exists():
                count = len(list(Path(path).glob('*.txt')))
                datasets[name] = count
            else:
                datasets[name] = 0
        
        return datasets
