"""
Улучшенный генератор поискового спама с помощью марковских цепей
"""
import random
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

class MarkovSpamGenerator:
    """Генератор реалистичного поискового спама"""
    
    def __init__(self, order=2, diversity=0.3):
        self.order = order
        self.diversity = diversity
        self.chain = defaultdict(Counter)
        self.start_words = []
        self.vocab = set()
    
    def train_on_corpus(self, texts):
        """Обучение на корпусе текстов"""
        print(f"🔧 Обучение марковской цепи на {len(texts)} текстах...")
        
        for text in texts:
            words = self._tokenize_text(text)
            if len(words) < self.order + 1:
                continue
            
            # Добавляем стартовые слова
            self.start_words.extend(words[:self.order])
            
            # Строим цепь
            for i in range(len(words) - self.order):
                state = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                
                self.chain[state][next_word] += 1
                self.vocab.add(next_word)
        
        print(f"✓ Обучение завершено")
        print(f"  Уникальных состояний: {len(self.chain)}")
        print(f"  Размер словаря: {len(self.vocab)}")
        print(f"  Стартовых слов: {len(self.start_words)}")
    
    def _tokenize_text(self, text):
        """Токенизация текста с сохранением SEO-паттернов"""
        # Сохраняем специальные SEO-символы
        text = re.sub(r'[!?]+', ' ! ', text)  # Восклицания
        text = re.sub(r'[%]+', ' % ', text)  # Проценты
        text = re.sub(r'[$]+', ' $ ', text)  # Доллары
        
        # Токенизация
        words = re.findall(r'\w+|[!%$#@&*()]', text.lower())
        
        # Фильтрация
        words = [word for word in words if len(word) > 1 or word in '!%$#@&*()']
        
        return words
    
    def generate_spam(self, min_length=15, max_length=50, spam_intensity=0.7):
        """Генерация спам-текста"""
        if not self.chain or not self.start_words:
            return "Обучите сначала модель на данных!"
        
        # Выбираем случайное начальное состояние
        start_idx = random.randint(0, len(self.start_words) - self.order)
        state = tuple(self.start_words[start_idx:start_idx + self.order])
        
        result = list(state)
        current_length = len(result)
        
        # Генерация текста
        while current_length < max_length:
            if state in self.chain and random.random() > self.diversity:
                # Выбираем следующее слово по вероятностям
                next_words = list(self.chain[state].keys())
                weights = list(self.chain[state].values())
                
                # Нормализуем веса
                total = sum(weights)
                probabilities = [w/total for w in weights]
                
                next_word = np.random.choice(next_words, p=probabilities)
            else:
                # Случайное слово для разнообразия
                next_word = random.choice(list(self.vocab)) if self.vocab else "spam"
            
            result.append(next_word)
            state = tuple(result[-self.order:])
            current_length += 1
            
            if current_length >= min_length and random.random() < 0.1:
                break
        
        # Пост-обработка для придания спам-вида
        spam_text = self._post_process_text(result, spam_intensity)
        return spam_text
    
    def _post_process_text(self, words, spam_intensity):
        """Добавление спам-паттернов к тексту"""
        text = ' '.join(words)
        
        # SEO-оптимизация
        if random.random() < spam_intensity:
            seo_patterns = [
                " buy now discount best price free shipping",
                " limited time offer click here win prize",
                " cheap affordable quality guaranteed satisfaction",
                " special promotion exclusive deal today only",
                " order now fast delivery money back guarantee"
            ]
            text += random.choice(seo_patterns)
        
        # Добавляем восклицательные знаки
        if random.random() < spam_intensity * 0.5:
            text = text.replace(' ! ', '!!! ')
            if random.random() < 0.3:
                text += "!!!"
        
        # Добавляем CAPS LOCK для emphasis
        if random.random() < spam_intensity * 0.3:
            words = text.split()
            if len(words) > 3:
                cap_word = random.randint(0, len(words) - 1)
                words[cap_word] = words[cap_word].upper()
                text = ' '.join(words)
        
        return text.capitalize()
    
    def generate_dataset(self, n_samples=100, **kwargs):
        """Генерация датасета спам-текстов"""
        print(f"🔄 Генерация {n_samples} спам-текстов...")
        
        samples = []
        for i in range(n_samples):
            spam_text = self.generate_spam(**kwargs)
            samples.append(spam_text)
            
            if (i + 1) % 10 == 0:
                print(f"  Сгенерировано: {i + 1}/{n_samples}", end='\r')
        
        print(f"\n✓ Датасет сгенерирован: {len(samples)} текстов")
        return samples
    
    def save_model(self, path='models/markov_spam_generator.pkl'):
        """Сохранение модели"""
        import joblib
        Path('models').mkdir(exist_ok=True)
        joblib.dump(self, path)
        print(f"✓ Модель сохранена: {path}")
    
    @classmethod
    def load_model(cls, path='models/markov_spam_generator.pkl'):
        """Загрузка модели"""
        import joblib
        return joblib.load(path)

def create_markov_spam_dataset():
    """Создание датасета марковского спама"""
    from data_loader import DataLoader
    
    # Загрузка существующих спам-текстов для обучения
    _, spam_texts = DataLoader.load_train_data()
    
    if not spam_texts:
        print("⚠️ Нет спам-текстов для обучения! Используем примеры...")
        sample_data = DataLoader.load_sample_data()
        spam_texts = sample_data.get('spam', [])
    
    if not spam_texts:
        # Резервные примеры
        spam_texts = [
            "buy cheap pills online pharmacy discount best price",
            "win money casino gambling bonus free spins jackpot",
            "make money fast easy work from home earn cash",
            "weight loss pills diet supplement fat burner quick",
            "SEO services optimization ranking google first page"
        ]
    
    # Создаем и обучаем генератор
    generator = MarkovSpamGenerator(order=2)
    generator.train_on_corpus(spam_texts)
    
    # Генерируем датасет
    markov_spam_texts = generator.generate_dataset(n_samples=100)
    
    # Сохраняем в отдельную папку
    markov_dir = Path('data/raw/markov_spam')
    markov_dir.mkdir(parents=True, exist_ok=True)
    
    for i, text in enumerate(markov_spam_texts):
        with open(markov_dir / f"markov_spam_{i}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"✓ Марковский спам сохранен: {markov_dir}")
    return markov_spam_texts

if __name__ == "__main__":
    create_markov_spam_dataset()
