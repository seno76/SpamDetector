"""
Скрипт для скачивания и подготовки данных для обучения
"""
import os
import json
import urllib.request
from pathlib import Path

def create_directories():
    """Создание структуры папок"""
    dirs = ['data/raw/natural', 'data/raw/spam', 'data/processed', 'models']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Структура папок создана")

def create_sample_data():
    """
    Создание примеров данных для тестирования
    В реальном проекте здесь будет загрузка из датасетов
    """
    
    # Примеры обычных текстов (нормальные статьи)
    natural_texts = [
        """Python is a high-level programming language known for its simplicity and readability. 
        It was created by Guido van Rossum and first released in 1991. Python supports multiple 
        programming paradigms including procedural, object-oriented, and functional programming.""",
        
        """Machine learning is a subset of artificial intelligence that provides systems the ability 
        to automatically learn and improve from experience without being explicitly programmed. 
        The process of learning begins with observations or data to look for patterns.""",
        
        """Climate change refers to long-term shifts in temperatures and weather patterns. 
        These shifts may be natural, but since the 1800s, human activities have been the main 
        driver of climate change, primarily due to the burning of fossil fuels.""",
        
        """The Internet has revolutionized communication and commerce. It connects billions of 
        devices worldwide and enables instant communication across vast distances. The World Wide 
        Web, invented in 1989, made the Internet accessible to the general public.""",
        
        """Renewable energy comes from natural sources that are constantly replenished. Solar power, 
        wind energy, and hydroelectric power are examples of renewable energy sources that produce 
        minimal environmental impact compared to fossil fuels."""
    ]
    
    # Примеры спам-текстов (переоптимизированные под SEO, бессвязные)
    spam_texts = [
        """Buy cheap viagra online pharmacy discount pills best price medicine drug store 
        pharmacy online cheap discount viagra cialis levitra buy now best offer limited time 
        pharmacy discount cheap cheap cheap buy buy buy online store medicine pills.""",
        
        """Casino online gambling poker slots blackjack roulette casino casino casino 
        play now win money jackpot bonus free spins online gambling best casino top rated 
        casino play casino games win big money now casino online gambling.""",
        
        """Make money online fast easy work from home earn cash quick rich wealthy 
        millionaire business opportunity make money make money online fast quick easy 
        simple method earn dollars work home business opportunity wealthy rich.""",
        
        """SEO services optimization ranking google first page backlinks traffic visitors 
        SEO expert consultant agency top ranking optimization services google ranking 
        first page position SEO optimization backlinks quality traffic increase ranking.""",
        
        """Weight loss pills diet supplement fat burner lose weight fast quick results 
        diet pills weight loss weight loss diet supplement fat burner pills lose weight 
        diet pills supplement burner fat loss quick fast results amazing."""
    ]
    
    # Сохранение в файлы
    for i, text in enumerate(natural_texts):
        with open(f'data/raw/natural/doc_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(text)
    
    for i, text in enumerate(spam_texts):
        with open(f'data/raw/spam/doc_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(text)
    
    # Создание JSON с примерами для быстрого тестирования
    sample_data = {
        'natural': natural_texts[:2],
        'spam': spam_texts[:2],
        'test': [
            "Artificial intelligence is transforming industries across the globe.",
            "Click here buy cheap pills online pharmacy discount now best price."
        ]
    }
    
    with open('data/sample_texts.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Создано {len(natural_texts)} обычных текстов")
    print(f"✓ Создано {len(spam_texts)} спам-текстов")
    print("✓ Создан файл с примерами: data/sample_texts.json")

if __name__ == "__main__":
    create_directories()
    create_sample_data()
    print("\n✓ Подготовка данных завершена!")
