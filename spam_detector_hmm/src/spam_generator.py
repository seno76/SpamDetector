"""
Генератор поискового спама с помощью марковских цепей
"""
import random
import re
from pathlib import Path

class MarkovSpamGenerator:
    """Генератор поискового спама с помощью марковских цепей"""
    
    def __init__(self, order=2):
        self.order = order
        self.chain = {}
        self.start_states = []
    
    def train_on_keywords(self, keywords_file):
        """Обучаем на SEO-ключевых словах"""
        try:
            with open(keywords_file, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"⚠️ Файл {keywords_file} не найден")
            return False
        
        # Строим марковскую цепь
        for phrase in keywords:
            words = self._clean_and_split(phrase)
            if len(words) < self.order + 1:
                continue
                
            # Добавляем стартовые состояния
            self.start_states.append(tuple(words[:self.order]))
            
            for i in range(len(words) - self.order):
                state = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                
                if state not in self.chain:
                    self.chain[state] = []
                self.chain[state].append(next_word)
        
        print(f"✓ Построена марковская цепь порядка {self.order}")
        print(f"  Уникальных состояний: {len(self.chain)}")
        print(f"  Стартовых состояний: {len(self.start_states)}")
        return True
    
    def _clean_and_split(self, text):
        """Очистка и разделение текста на слова"""
        # Удаляем лишние пробелы и разделяем
        text = re.sub(r'\s+', ' ', text.strip())
        return text.split()
    
    def generate_spam(self, length=50, diversity=0.3):
        """Генерируем поисковый спам"""
        if not self.chain or not self.start_states:
            return "Марковская цепь не обучена"
        
        # Начинаем со случайного состояния
        state = random.choice(self.start_states)
        result = list(state)
        
        for _ in range(length):
            if state in self.chain and random.random() > diversity:
                # Следуем цепи
                next_word = random.choice(self.chain[state])
                result.append(next_word)
                state = tuple(result[-self.order:])
            else:
                # Добавляем разнообразия - случайное слово из цепи
                if self.chain:
                    random_state = random.choice(list(self.chain.keys()))
                    if self.chain[random_state]:
                        next_word = random.choice(self.chain[random_state])
                        result.append(next_word)
                        state = tuple(result[-self.order:])
                    else:
                        break
                else:
                    break
        
        generated_text = " ".join(result)
        
        # Добавляем спам-паттерны для реалистичности
        if random.random() > 0.5:
            spam_patterns = [
                " купить со скидкой акция бесплатно",
                " доставка по всей России скидки",
                " лучшая цена гарантия качества",
                " скидки до 50% распродажа",
                " бесплатная доставка установка"
            ]
            generated_text += random.choice(spam_patterns)
        
        return generated_text
    
    def generate_multiple_spam(self, count=10, min_length=30, max_length=100):
        """Генерация нескольких спам-текстов"""
        texts = []
        for i in range(count):
            length = random.randint(min_length, max_length)
            text = self.generate_spam(length=length)
            texts.append(text)
        return texts

def create_seo_keywords_file():
    """Создание файла с SEO-ключевыми словами"""
    keywords = [
        "купить квартира Москва недорого",
        "заказать пицца доставка быстро",
        "скачать фильм бесплатно без регистрации",
        "онлайн казино бонус за регистрацию",
        "курсы английского языка обучение",
        "ремонт iPhone быстро качественно",
        "туристические путевки горящие туры",
        "кредит наличными срочно выгодно",
        "работа в интернете без вложений",
        "похудение за неделю быстро эффективно",
        "купить автомобиль с пробегом",
        "запись к врачу онлайн",
        "доставка еды рестораны",
        "строительство домов под ключ",
        "обучение программированию с нуля",
        "купить ноутбук недорого Москва",
        "заказ такси недорого быстро",
        "установка окон пластиковые окна",
        "ремонт квартир под ключ",
        "услуги клининга уборка квартир"
    ]
    
    Path('data').mkdir(exist_ok=True)
    with open('data/seo_keywords.txt', 'w', encoding='utf-8') as f:
        for keyword in keywords:
            f.write(keyword + '\n')
    
    print("✓ Создан файл с SEO-ключевыми словами")
    return 'data/seo_keywords.txt'

# УБИРАЕМ код, который выполняется при импорте
# if __name__ == "__main__":
#     # Тестирование генератора
#     generator = MarkovSpamGenerator(order=2)
#     keywords_file = create_seo_keywords_file()
#     generator.train_on_keywords(keywords_file)
#     
#     print("\n🔍 Примеры сгенерированного спама:")
#     for i in range(3):
#         spam_text = generator.generate_spam(length=40)
#         print(f"{i+1}. {spam_text}")