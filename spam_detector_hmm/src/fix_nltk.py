"""
Скрипт для загрузки всех необходимых данных NLTK
"""
import nltk

print("🔧 Загрузка необходимых пакетов NLTK...")

# Список всех необходимых ресурсов
resources = [
    'punkt',           # Токенизация предложений (старая версия)
    'punkt_tab',       # Токенизация предложений (новая версия)
    'averaged_perceptron_tagger',  # POS-теггер (старая версия)
    'averaged_perceptron_tagger_eng',  # POS-теггер (новая версия для английского)
    'stopwords',       # Стоп-слова
]

for resource in resources:
    try:
        print(f"\n📦 Загрузка: {resource}...")
        nltk.download(resource, quiet=False)
        print(f"   ✓ {resource} загружен успешно")
    except Exception as e:
        print(f"   ⚠️  Ошибка при загрузке {resource}: {e}")

print("\n✅ Все пакеты NLTK загружены!")
