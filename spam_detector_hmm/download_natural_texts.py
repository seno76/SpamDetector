"""
Скачивание обычных текстов (статьи, новости, литература)
"""
import os
import requests
import wikipedia
from pathlib import Path
import time
import random

def download_wikipedia_articles():
    """Скачивание статей из Wikipedia"""
    print("📚 Загрузка статей из Wikipedia...")
    
    # Темы для статей (разные области знаний)
    topics = [
        "Artificial intelligence", "Machine learning", "Python programming",
        "Data science", "Computer vision", "Natural language processing",
        "Deep learning", "Neural networks", "Statistics", "Mathematics",
        "Physics", "Chemistry", "Biology", "History", "Geography",
        "Literature", "Philosophy", "Psychology", "Economics", "Sociology",
        "Astronomy", "Geology", "Medicine", "Engineering", "Technology",
        "Music", "Art", "Architecture", "Sports", "Education",
        "Climate change", "Renewable energy", "Space exploration", "Robotics",
        "Internet", "Cybersecurity", "Blockchain", "Virtual reality",
        "Quantum computing", "Biotechnology", "Nanotechnology", "Genetics",
        "Ecology", "Agriculture", "Transportation", "Communication",
        "Linguistics", "Anthropology", "Political science", "Law"
    ]
    
    natural_dir = Path('data/raw/natural')
    natural_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for topic in topics:
        try:
            # Устанавливаем язык
            wikipedia.set_lang("en")
            
            # Получаем страницу
            page = wikipedia.page(topic)
            content = page.content
            
            # Сохраняем только если текст достаточно большой
            if len(content) > 1000:
                filename = natural_dir / f"wiki_{topic.replace(' ', '_').lower()}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                downloaded += 1
                print(f"  ✓ {topic} ({len(content)} chars)")
                
                # Пауза чтобы не перегружать сервер
                time.sleep(1)
                
            if downloaded >= 50:  # Останавливаемся на 50 статьях
                break
                
        except Exception as e:
            print(f"  ✗ Ошибка с '{topic}': {e}")
            continue
    
    return downloaded

def download_project_gutenberg_books():
    """Скачивание книг из Project Gutenberg"""
    print("\n📖 Загрузка книг из Project Gutenberg...")
    
    # ID книг в Project Gutenberg (классическая литература)
    book_ids = [
        1342,  # Pride and Prejudice
        84,    # Frankenstein
        11,    # Alice's Adventures in Wonderland
        1661,  # The Adventures of Sherlock Holmes
        74,    # The Adventures of Tom Sawyer
        2701,  # Moby Dick
        98,    # A Tale of Two Cities
        76,    # Adventures of Huckleberry Finn
        1260,  # Jane Eyre
        2554,  # Crime and Punishment
        2600,  # War and Peace
        1080,  # A Modest Proposal
        174,   # The Picture of Dorian Gray
        768,   # Wuthering Heights
        203,   # The Souls of Black Folk
        345,   # Dracula
        5200,  # Metamorphosis
        1232,  # The Prince
        1399,  # Anne of Green Gables
        160,   # The Awakening
    ]
    
    natural_dir = Path('data/raw/natural')
    downloaded = 0
    
    for book_id in book_ids:
        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Очищаем текст от заголовков Gutenberg
                text = response.text
                start_markers = ["*** START OF", "***START OF"]
                end_markers = ["*** END OF", "***END OF"]
                
                for marker in start_markers:
                    if marker in text:
                        text = text.split(marker, 1)[1]
                
                for marker in end_markers:
                    if marker in text:
                        text = text.split(marker, 1)[0]
                
                if len(text) > 5000:  # Сохраняем только большие тексты
                    filename = natural_dir / f"gutenberg_{book_id}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    downloaded += 1
                    print(f"  ✓ Книга ID {book_id} ({len(text)} chars)")
                    
                    time.sleep(2)  # Уважаем сервер
                    
        except Exception as e:
            print(f"  ✗ Ошибка с книгой {book_id}: {e}")
            continue
    
    return downloaded

def create_sample_natural_texts():
    """Создание дополнительных примеров если не хватило загруженных"""
    natural_dir = Path('data/raw/natural')
    
    # Примеры качественных текстов на разные темы
    sample_texts = [
        """Machine learning is a method of data analysis that automates analytical model building. 
        It is a branch of artificial intelligence based on the idea that systems can learn from data, 
        identify patterns and make decisions with minimal human intervention.""",
        
        """Renewable energy is energy that is collected from renewable resources that are naturally 
        replenished on a human timescale. It includes sources like sunlight, wind, rain, tides, 
        waves, and geothermal heat.""",
        
        """Climate change refers to long-term shifts in temperatures and weather patterns. 
        These shifts may be natural, but since the 1800s, human activities have been the main 
        driver of climate change, primarily due to the burning of fossil fuels.""",
        
        """The Internet has revolutionized communication and commerce. It connects billions of 
        devices worldwide and enables instant communication across vast distances.""",
        
        """Python is an interpreted high-level general-purpose programming language. Its design 
        philosophy emphasizes code readability with its use of significant indentation."""
    ]
    
    created = 0
    existing_files = len(list(natural_dir.glob("*.txt")))
    
    # Добавляем примеры пока не достигнем 100 файлов
    for i in range(max(0, 100 - existing_files)):
        text = random.choice(sample_texts)
        filename = natural_dir / f"sample_natural_{i}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        created += 1
    
    if created > 0:
        print(f"\n📝 Создано {created} примеров обычных текстов")
    
    return created

if __name__ == "__main__":
    print("🚀 ЗАГРУЗКА ОБЫЧНЫХ ТЕКСТОВ")
    print("=" * 50)
    
    total_downloaded = 0
    
    # Загружаем из Wikipedia
    wiki_count = download_wikipedia_articles()
    total_downloaded += wiki_count
    
    # Загружаем из Project Gutenberg
    gutenberg_count = download_project_gutenberg_books()
    total_downloaded += gutenberg_count
    
    # Добираем до 100 файлов примерами
    sample_count = create_sample_natural_texts()
    total_downloaded += sample_count
    
    # Итоговая статистика
    natural_dir = Path('data/raw/natural')
    final_count = len(list(natural_dir.glob("*.txt")))
    
    print(f"\n✅ ЗАВЕРШЕНО!")
    print(f"📊 Итоговая статистика:")
    print(f"   Wikipedia статей: {wiki_count}")
    print(f"   Gutenberg книг: {gutenberg_count}")
    print(f"   Примеров создано: {sample_count}")
    print(f"   Всего файлов: {final_count}")
    print(f"   Папка: {natural_dir.absolute()}")