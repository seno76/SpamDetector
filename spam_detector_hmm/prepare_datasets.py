"""
Автоматическая подготовка всех необходимых датасетов
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from download_data import create_directories, create_sample_data
from download_natural_texts import download_wikipedia_articles, download_project_gutenberg_books
from download_spam_texts import download_sms_spam_collection, download_email_spam, generate_seo_spam
from src.markov_spam_generator import create_markov_spam_dataset
from src.data_loader import DataLoader

def prepare_all_datasets():
    """Подготовка всех типов данных"""
    print("🚀 ПОДГОТОВКА ПОЛНОГО ДАТАСЕТА")
    print("="*50)
    
    # 1. Создаем структуру папок
    create_directories()
    
    # 2. Загружаем обычные тексты
    print("\n📚 ЗАГРУЗКА ОБЫЧНЫХ ТЕКСТОВ")
    print("-"*30)
    
    wiki_count = download_wikipedia_articles()
    gutenberg_count = download_project_gutenberg_books()
    
    # 3. Загружаем человеческий спам
    print("\n📧 ЗАГРУЗКА ЧЕЛОВЕЧЕСКОГО СПАМА")
    print("-"*30)
    
    sms_count = download_sms_spam_collection()
    email_count = download_email_spam()
    seo_count = generate_seo_spam()
    
    # 4. Генерируем марковский спам
    print("\n🤖 ГЕНЕРАЦИЯ МАРКОВСКОГО СПАМА")
    print("-"*30)
    
    markov_texts = create_markov_spam_dataset()
    
    # 5. Создаем примеры если нужно
    if wiki_count + gutenberg_count < 10:
        print("\n📝 СОЗДАНИЕ ПРИМЕРОВ")
        print("-"*30)
        create_sample_data()
    
    # 6. Статистика
    print("\n📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*50)
    
    datasets = DataLoader.get_available_datasets()
    for name, count in datasets.items():
        print(f"   {name}: {count} текстов")
    
    total_texts = sum(datasets.values())
    print(f"   ВСЕГО: {total_texts} текстов")
    
    print("\n✅ ПОДГОТОВКА ЗАВЕРШЕНА!")
    return datasets

if __name__ == "__main__":
    prepare_all_datasets()
