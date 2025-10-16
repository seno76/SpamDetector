"""
Расширенная версия с поддержкой разных типов спама + интерактивное тестирование
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from preprocessor import TextPreprocessor
from hmm_detector import SpamDetectorHMM
from visualizer import Visualizer
from markov_spam_generator import create_markov_spam_dataset

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
import joblib

# Глобальные переменные для хранения последней обученной модели
LAST_DETECTOR = None
LAST_PREPROCESSOR = None
LAST_SPAM_TYPE = None

def print_header(text):
    """Красивый заголовок"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def analyze_text_interactive(detector, preprocessor, text):
    """Детальный анализ введенного текста (аналогично main.py)"""
    import numpy as np
    from collections import Counter

    print("\n" + "-"*80)
    print("📄 АНАЛИЗ ВАШЕГО ТЕКСТА")
    print("-"*80)
    print(f"Текст (первые 150 символов):")
    print(f"   {text[:150]}{'...' if len(text) > 150 else ''}")
    print("-"*80)

    # Токенизация и статистика
    tokens = preprocessor.tokenize(text)
    pos_tags = preprocessor.extract_pos_features(text)
    seq = preprocessor.text_to_sequence(text)
    unk_idx = preprocessor.vocab.get('UNK', preprocessor.get_vocabulary_size() - 1)
    unk_ratio = float((seq == unk_idx).sum()) / len(seq) if len(seq) > 0 else 0.0

    print(f"\n📊 СТАТИСТИКА ТЕКСТА:")
    print(f"   Длина текста: {len(text)} символов")
    print(f"   Токенов после очистки: {len(tokens)}")
    print(f"   POS-тегов: {len(pos_tags)}")
    print(f"   Длина последовательности: {len(seq)}")
    print(f"   Уникальных признаков: {len(np.unique(seq))}")
    print(f"   Доля UNK-токенов: {unk_ratio:.2%}")
    
    if len(pos_tags) > 0:
        common_pos = Counter(pos_tags).most_common(5)
        print(f"   Топ-5 POS тегов: {common_pos}")

    # Классификация
    result = detector.predict_proba(seq)
    
    print(f"\n🎯 РЕЗУЛЬТАТ КЛАССИФИКАЦИИ:")
    print(f"   {'='*70}")
    print(f"   Предсказание: {result['prediction'].upper()}")
    print(f"   {'='*70}")
    print(f"   Log P(X|Natural): {result['log_prob_natural']:.2f}")
    print(f"   Log P(X|Spam):    {result['log_prob_spam']:.2f}")
    print(f"   Разница:          {abs(result['log_prob_natural'] - result['log_prob_spam']):.2f}")
    print(f"   {'='*70}")
    print(f"   P(Natural|X):     {result['prob_natural']:.4f} ({result['prob_natural']*100:.2f}%)")
    print(f"   P(Spam|X):        {result['prob_spam']:.4f} ({result['prob_spam']*100:.2f}%)")
    print(f"   {'='*70}")

    # Интерпретация результата
    if result['prediction'] == 'spam':
        confidence = result['prob_spam']
        if confidence > 0.9:
            print("\n⚠️  🔴 ВЫСОКАЯ ВЕРОЯТНОСТЬ СПАМА!")
        elif confidence > 0.7:
            print("\n⚠️  🟠 СРЕДНЯЯ ВЕРОЯТНОСТЬ СПАМА")
        else:
            print("\n⚠️  🟡 НИЗКАЯ ВЕРОЯТНОСТЬ СПАМА (граничный случай)")
    else:
        confidence = result['prob_natural']
        if confidence > 0.9:
            print("\n✅ 🟢 ВЫСОКАЯ УВЕРЕННОСТЬ: ОБЫЧНЫЙ ТЕКСТ")
        elif confidence > 0.7:
            print("\n✅ 🟢 СРЕДНЯЯ УВЕРЕННОСТЬ: ОБЫЧНЫЙ ТЕКСТ")
        else:
            print("\n✅ 🟡 НИЗКАЯ УВЕРЕННОСТЬ (граничный случай)")

    # Витерби декодирование
    print(f"\n🔍 АЛГОРИТМ ВИТЕРБИ (декодирование скрытых состояний):")
    
    for model_type in ['natural', 'spam']:
        vit = detector.decode_viterbi(seq, model_type=model_type)
        gamma = detector.get_posteriors(seq, model_type=model_type)
        
        print(f"\n   📈 {model_type.upper()} модель:")
        print(f"      Витерби log-prob: {vit['log_probability']:.2f}")
        print(f"      Использовано состояний: {vit['n_states_used']}/{detector.n_states}")
        
        if gamma.size > 0:
            avg_gamma = gamma.mean(axis=0)
            print(f"      Средние постериоры: {np.round(avg_gamma, 3)}")
        
        if len(vit['states']) > 0:
            print(f"      Путь (первые 30): {vit['states'][:30]}")
            # Распределение по состояниям
            unique, counts = np.unique(vit['states'], return_counts=True)
            print(f"      Распределение состояний:")
            for state, count in zip(unique, counts):
                percentage = (count / len(vit['states'])) * 100
                print(f"         Состояние {state}: {count} раз ({percentage:.1f}%)")

    print("-"*80)

def interactive_testing_mode(detector, preprocessor, spam_type):
    """Интерактивный режим тестирования текстов"""
    print_header(f"ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ - {spam_type}")
    
    print("💡 Введите текст для анализа (или команду):")
    print("   • Просто напишите текст и нажмите Enter")
    print("   • 'exit' или 'quit' — выход")
    print("   • 'example' — тестовые примеры")
    print("   • 'stats' — статистика модели")
    print("   • 'graphs' — показать графики")
    
    while True:
        print("\n" + "="*80)
        user_input = input("📝 Ваш текст: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
            print("👋 Выход из режима тестирования")
            break
        
        if not user_input:
            print("⚠️  Введите текст!")
            continue
        
        if user_input.lower() == 'example':
            test_examples(detector, preprocessor)
            continue
        
        if user_input.lower() == 'stats':
            show_model_stats(detector, preprocessor)
            continue
        
        if user_input.lower() == 'graphs':
            show_all_visualizations(detector, preprocessor)
            continue
        
        # Анализ текста
        try:
            analyze_text_interactive(detector, preprocessor, user_input)
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()

def test_examples(detector, preprocessor):
    """Тестирование на заранее подготовленных примерах"""
    examples = [
        ("Machine learning is a method of data analysis that automates analytical model building.", "natural"),
        ("Buy cheap pills online pharmacy discount best price now click here!!!", "spam"),
        ("Climate change refers to long-term shifts in temperatures and weather patterns.", "natural"),
        ("Win money casino gambling bonus free spins jackpot limited time offer!", "spam"),
        ("Python is an interpreted high-level programming language with dynamic semantics.", "natural")
    ]
    
    print("\n" + "="*80)
    print("ТЕСТОВЫЕ ПРИМЕРЫ")
    print("="*80)
    
    for i, (text, expected) in enumerate(examples, 1):
        print(f"\n{'─'*80}")
        print(f"ПРИМЕР {i} (ожидается: {expected.upper()})")
        print(f"{'─'*80}")
        analyze_text_interactive(detector, preprocessor, text)

def show_model_stats(detector, preprocessor):
    """Показать статистику обученной модели"""
    print("\n" + "="*80)
    print("СТАТИСТИКА МОДЕЛИ")
    print("="*80)
    
    print(f"\n📊 Параметры HMM:")
    print(f"   Количество скрытых состояний: {detector.n_states}")
    print(f"   Размер словаря признаков: {detector.n_features}")
    print(f"   Размер словаря (preprocessor): {preprocessor.get_vocabulary_size()}")
    
    print(f"\n🔤 Топ-10 признаков в словаре:")
    vocab_items = list(preprocessor.vocab.items())[:10]
    for feature, idx in vocab_items:
        print(f"      {feature}: {idx}")
    
    print(f"\n📈 Матрица переходов (Natural):")
    print(detector.get_transition_matrix('natural'))
    
    print(f"\n📈 Матрица переходов (Spam):")
    print(detector.get_transition_matrix('spam'))

def show_all_visualizations(detector, preprocessor):
    """Показать все графики как в main.py"""
    print_header("ВИЗУАЛИЗАЦИЯ МОДЕЛЕЙ")
    
    try:
        print("\n📊 Создание графиков...")
        
        # 1. Матрицы переходов
        print("   • Матрицы переходов...")
        Visualizer.plot_transition_matrices(detector)
        
        # 2. Распределения эмиссий
        print("   • Распределения эмиссий...")
        vocab_size = preprocessor.get_vocabulary_size()
        Visualizer.plot_emission_distributions(detector, top_n=min(10, vocab_size))
        
        # 3. Важность признаков
        print("   • Важность признаков...")
        Visualizer.plot_feature_importance(preprocessor, top_n=20)
        
        print("\n✅ Все графики построены и сохранены!")
        
    except Exception as e:
        print(f"❌ Ошибка визуализации: {e}")
        import traceback
        traceback.print_exc()

def train_model(natural_texts, spam_texts, description):
    """Универсальная функция обучения модели"""
    global LAST_DETECTOR, LAST_PREPROCESSOR, LAST_SPAM_TYPE
    
    print(f"\n🔧 ОБУЧЕНИЕ НА: {description}")
    print("-"*50)
    
    if len(natural_texts) < 2 or len(spam_texts) < 2:
        print("❌ Недостаточно данных для обучения (минимум 2 текста каждого класса)")
        return None, None
    
    # Разделение данных
    natural_train_texts, natural_test_texts = train_test_split(
        natural_texts, test_size=0.2, random_state=42
    )
    spam_train_texts, spam_test_texts = train_test_split(
        spam_texts, test_size=0.2, random_state=42
    )
    
    print(f"📦 Разделение данных:")
    print(f"   Train Natural: {len(natural_train_texts)}")
    print(f"   Test Natural:  {len(natural_test_texts)}")
    print(f"   Train Spam:    {len(spam_train_texts)}")
    print(f"   Test Spam:     {len(spam_test_texts)}")
    
    # Предобработка
    preprocessor = TextPreprocessor(feature_type='pos', n_symbols=50)
    train_texts = natural_train_texts + spam_train_texts
    print(f"\n🔄 Построение словаря на {len(train_texts)} текстах...")
    preprocessor.build_vocabulary(train_texts)
    
    # Преобразование в последовательности
    natural_train = preprocessor.texts_to_sequences(natural_train_texts)
    spam_train = preprocessor.texts_to_sequences(spam_train_texts)
    natural_test = preprocessor.texts_to_sequences(natural_test_texts)
    spam_test = preprocessor.texts_to_sequences(spam_test_texts)
    
    # Обучение детектора
    print(f"\n🧠 Обучение HMM моделей...")
    detector = SpamDetectorHMM(n_states=3, n_iter=100)
    detector.fit(natural_train, spam_train, preprocessor.get_vocabulary_size())
    
    # Тестирование
    print(f"\n🎯 Тестирование на отложенной выборке...")
    test_sequences = natural_test + spam_test
    true_labels = ['natural'] * len(natural_test) + ['spam'] * len(spam_test)
    predictions = detector.predict(test_sequences)
    
    # Метрики
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label='spam', zero_division=0)
    recall = recall_score(true_labels, predictions, pos_label='spam', zero_division=0)
    f1 = f1_score(true_labels, predictions, pos_label='spam', zero_division=0)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ({description}):")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    
    # Classification Report
    print(f"\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ:")
    print(classification_report(true_labels, predictions, 
                               target_names=['Natural', 'Spam'], 
                               zero_division=0))
    
    # Визуализация результатов классификации
    print(f"\n📊 Создание визуализаций...")
    Visualizer.plot_classification_comparison(predictions, true_labels)
    
    # Визуализация моделей
    print(f"   • Матрицы переходов...")
    Visualizer.plot_transition_matrices(detector)
    
    print(f"   • Распределения эмиссий...")
    vocab_size = preprocessor.get_vocabulary_size()
    Visualizer.plot_emission_distributions(detector, top_n=min(10, vocab_size))
    
    # Сохранение модели
    model_path = f'models/{description.replace(" ", "_").lower()}'
    Path(model_path).mkdir(parents=True, exist_ok=True)
    detector.save(model_path)
    
    # Сохраняем препроцессор
    joblib.dump(preprocessor, f'{model_path}/preprocessor.pkl')
    print(f"✓ Модели и препроцессор сохранены в {model_path}")
    
    # Сохраняем как последнюю модель
    LAST_DETECTOR = detector
    LAST_PREPROCESSOR = preprocessor
    LAST_SPAM_TYPE = description
    
    return detector, preprocessor



def load_saved_model(spam_type):
    """Загрузка сохраненной модели"""
    global LAST_DETECTOR, LAST_PREPROCESSOR, LAST_SPAM_TYPE
    
    model_name = spam_type.replace(" ", "_").lower()
    model_path = f'models/{model_name}'
    
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        return None, None
    
    try:
        print(f"📦 Загрузка модели из {model_path}...")
        
        detector = SpamDetectorHMM()
        detector.load(model_path)
        
        preprocessor = joblib.load(f'{model_path}/preprocessor.pkl')
        
        print(f"✅ Модель успешно загружена!")
        print(f"   Тип: {spam_type}")
        print(f"   Состояний: {detector.n_states}")
        print(f"   Размер словаря: {preprocessor.get_vocabulary_size()}")
        
        # Сохраняем как последнюю модель
        LAST_DETECTOR = detector
        LAST_PREPROCESSOR = preprocessor
        LAST_SPAM_TYPE = spam_type
        
        return detector, preprocessor
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def list_available_models():
    """Список доступных сохраненных моделей"""
    models_dir = Path('models')
    if not models_dir.exists():
        return []
    
    available_models = []
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            metadata_file = model_path / 'metadata.pkl'
            if metadata_file.exists():
                available_models.append(model_path.name)
    
    return available_models

def show_menu():
    """Показать главное меню"""
    print_header("ДЕТЕКТОР СПАМА - РАСШИРЕННАЯ ВЕРСИЯ")
    
    if LAST_DETECTOR:
        print(f"🔄 Текущая модель: {LAST_SPAM_TYPE}")
    else:
        print(f"⚪ Модель не загружена")
    
    print("\n🎯 РЕЖИМЫ РАБОТЫ:")
    print("\n📚 ОБУЧЕНИЕ:")
    print("1. 🧠 Обучение на человеческом спаме")
    print("2. 🤖 Обучение на марковском спаме")
    print("3. 🔄 Обучение на смешанном датасете")
    print("4. 📊 Сравнение всех типов")
    
    print("\n🔍 ТЕСТИРОВАНИЕ:")
    print("5. 🎮 Интерактивное тестирование (текущая модель)")
    print("6. 📂 Загрузить сохраненную модель")
    print("7. 🧪 Тест на внешних данных (unseen data)")  # ← НОВОЕ!
    
    print("\n🛠️ УТИЛИТЫ:")
    print("8. 🤖 Генерация марковского спама")
    print("9. 📈 Анализ существующих данных")
    print("10. 🚀 Автоподготовка всех данных")
    print("11. 📝 Создать тестовый датасет")  # ← НОВОЕ!
    print("12. ❌ Выход")
    
    choice = input("\n📋 Ваш выбор (1-12): ").strip()
    return choice


def train_on_human_spam():
    """Обучение на человеческом спаме"""
    print_header("ОБУЧЕНИЕ НА ЧЕЛОВЕЧЕСКОМ СПАМЕ")
    
    datasets = DataLoader.load_all_data()
    
    if not datasets['human_spam']:
        print("❌ Нет человеческого спама! Запустите подготовку данных (опция 9)")
        return None, None
    
    print(f"\n📊 Данные:")
    print(f"   Natural тексты: {len(datasets['human_natural'])}")
    print(f"   Human spam: {len(datasets['human_spam'])}")
    
    result = train_model(datasets['human_natural'], datasets['human_spam'], "Человеческий спам")
    
    if result[0]:
        choice = input("\n💡 Хотите протестировать модель в интерактивном режиме? (y/n): ").strip().lower()
        if choice in ['y', 'yes', 'д', 'да']:
            interactive_testing_mode(result[0], result[1], "Человеческий спам")
    
    return result

def train_on_markov_spam():
    """Обучение на марковском спаме"""
    print_header("ОБУЧЕНИЕ НА МАРКОВСКОМ СПАМЕ")
    
    datasets = DataLoader.load_all_data()
    
    if not datasets['markov_spam']:
        print("⚠️ Марковский спам не найден! Генерируем...")
        create_markov_spam_dataset(50)
        datasets = DataLoader.load_all_data()
    
    if not datasets['markov_spam']:
        print("❌ Не удалось сгенерировать марковский спам!")
        return None, None
    
    print(f"\n📊 Данные:")
    print(f"   Natural тексты: {len(datasets['human_natural'])}")
    print(f"   Markov spam: {len(datasets['markov_spam'])}")
    
    result = train_model(datasets['human_natural'], datasets['markov_spam'], "Марковский спам")
    
    if result[0]:
        choice = input("\n💡 Хотите протестировать модель в интерактивном режиме? (y/n): ").strip().lower()
        if choice in ['y', 'yes', 'д', 'да']:
            interactive_testing_mode(result[0], result[1], "Марковский спам")
    
    return result

def train_on_mixed_dataset():
    """Обучение на смешанном датасете"""
    print_header("ОБУЧЕНИЕ НА СМЕШАННОМ ДАТАСЕТЕ")
    
    datasets = DataLoader.load_all_data()
    
    if not datasets['markov_spam']:
        print("⚠️ Генерируем марковский спам...")
        create_markov_spam_dataset(50)
        datasets = DataLoader.load_all_data()
    
    mixed_spam = datasets['human_spam'] + datasets.get('markov_spam', [])
    
    if not mixed_spam:
        print("❌ Нет спам-данных!")
        return None, None
    
    print(f"\n📊 Смешанный датасет:")
    print(f"   Natural тексты: {len(datasets['human_natural'])}")
    print(f"   Human spam: {len(datasets['human_spam'])}")
    print(f"   Markov spam: {len(datasets.get('markov_spam', []))}")
    print(f"   Общий спам: {len(mixed_spam)}")
    
    result = train_model(datasets['human_natural'], mixed_spam, "Смешанный спам")
    
    if result[0]:
        choice = input("\n💡 Хотите протестировать модель в интерактивном режиме? (y/n): ").strip().lower()
        if choice in ['y', 'yes', 'д', 'да']:
            interactive_testing_mode(result[0], result[1], "Смешанный спам")
    
    return result

def compare_spam_types():
    """Сравнение разных типов спама"""
    print_header("СРАВНЕНИЕ ТИПОВ СПАМА")
    
    datasets = DataLoader.load_all_data()
    results = {}
    
    if datasets['human_spam']:
        detector1, _ = train_model(datasets['human_natural'], datasets['human_spam'], "Человеческий спам")
        if detector1:
            results['human_only'] = detector1
    
    if datasets['markov_spam'] or datasets['human_spam']:
        if not datasets['markov_spam']:
            create_markov_spam_dataset(50)
            datasets = DataLoader.load_all_data()
        
        if datasets['markov_spam']:
            detector2, _ = train_model(datasets['human_natural'], datasets['markov_spam'], "Марковский спам")
            if detector2:
                results['markov_only'] = detector2
    
    if datasets['human_spam'] and datasets['markov_spam']:
        mixed = datasets['human_spam'] + datasets['markov_spam']
        detector3, _ = train_model(datasets['human_natural'], mixed, "Смешанный спам")
        if detector3:
            results['mixed'] = detector3
    
    if results:
        print("\n📊 Визуализация сравнения...")
        try:
            Visualizer.plot_comparison(results)
        except Exception as e:
            print(f"⚠️ Ошибка визуализации: {e}")
    
    return results

def test_on_external_data(detector, preprocessor, spam_type):
    """Тестирование на внешнем (unseen) датасете"""
    print_header(f"ТЕСТИРОВАНИЕ НА ВНЕШНИХ ДАННЫХ - {spam_type}")
    
    test_dir = Path('data/test')
    
    if not test_dir.exists():
        print("❌ Тестовый датасет не найден!")
        print("💡 Запустите: python create_test_dataset.py")
        
        choice = input("\nСоздать тестовый датасет сейчас? (y/n): ").strip().lower()
        if choice in ['y', 'yes', 'д', 'да']:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from spam_detector_hmm.src.create_test_dataset import save_test_dataset
            save_test_dataset()
        else:
            return
    
    # Загрузка тестовых данных
    test_natural_dir = test_dir / 'natural'
    test_spam_dir = test_dir / 'spam'
    
    test_natural_texts = []
    for file_path in test_natural_dir.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            test_natural_texts.append(f.read())
    
    test_spam_texts = []
    for file_path in test_spam_dir.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            test_spam_texts.append(f.read())
    
    print(f"\n📊 Внешний тестовый датасет:")
    print(f"   Natural тексты: {len(test_natural_texts)}")
    print(f"   Spam тексты: {len(test_spam_texts)}")
    
    # Преобразование в последовательности
    natural_test_seqs = preprocessor.texts_to_sequences(test_natural_texts)
    spam_test_seqs = preprocessor.texts_to_sequences(test_spam_texts)
    
    # Предсказания
    test_sequences = natural_test_seqs + spam_test_seqs
    true_labels = ['natural'] * len(natural_test_seqs) + ['spam'] * len(spam_test_seqs)
    
    print(f"\n🔮 Выполнение классификации...")
    predictions = detector.predict(test_sequences)
    
    # Детальный анализ ошибок
    errors = []
    for i, (true, pred, text) in enumerate(zip(true_labels, predictions, test_natural_texts + test_spam_texts)):
        if true != pred:
            errors.append({
                'index': i,
                'true': true,
                'predicted': pred,
                'text': text[:100]
            })
    
    # Метрики
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label='spam', zero_division=0)
    recall = recall_score(true_labels, predictions, pos_label='spam', zero_division=0)
    f1 = f1_score(true_labels, predictions, pos_label='spam', zero_division=0)
    
    print(f"\n{'='*80}")
    print(f"📊 РЕЗУЛЬТАТЫ НА ВНЕШНИХ ДАННЫХ ({spam_type})")
    print(f"{'='*80}")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    print(f"{'='*80}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions, labels=['natural', 'spam'])
    print(f"\n📋 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              Natural  Spam")
    print(f"True Natural:    {cm[0][0]:3d}     {cm[0][1]:3d}")
    print(f"     Spam:       {cm[1][0]:3d}     {cm[1][1]:3d}")
    
    # Детальный отчет
    print(f"\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ:")
    print(classification_report(true_labels, predictions, 
                               target_names=['Natural', 'Spam'], 
                               zero_division=0))
    
    # Анализ ошибок
    if errors:
        print(f"\n❌ ОШИБКИ КЛАССИФИКАЦИИ ({len(errors)}):")
        for error in errors[:5]:  # Показываем первые 5
            print(f"\n   Пример #{error['index']}:")
            print(f"      Истинная метка: {error['true']}")
            print(f"      Предсказание: {error['predicted']}")
            print(f"      Текст: {error['text']}...")
    else:
        print(f"\n✅ НЕТ ОШИБОК! Идеальная классификация!")
    
    # Визуализация
    Visualizer.plot_classification_comparison(predictions, true_labels)
    
    return accuracy, precision, recall, f1


def main():
    """Главная функция с полным меню"""
    global LAST_DETECTOR, LAST_PREPROCESSOR, LAST_SPAM_TYPE
    
    while True:
        choice = show_menu()
        
        try:
            # ==================== ОБУЧЕНИЕ ====================
            if choice == '1':
                # Обучение на человеческом спаме
                try:
                    result = train_on_human_spam()
                    if result and result[0]:
                        print("\n" + "="*80)
                        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
                        print("="*80)
                        print(f"💾 Модель сохранена в: models/человеческий_спам/")
                        print(f"📊 Тип спама: Человеческий")
                        print(f"🔢 Состояний HMM: {result[0].n_states}")
                        print(f"📖 Размер словаря: {result[1].get_vocabulary_size()}")
                    else:
                        print("\n❌ Обучение не удалось. Проверьте данные.")
                        print("💡 Запустите опцию 10 для подготовки данных")
                except Exception as e:
                    print(f"\n❌ Ошибка обучения: {e}")
                    import traceback
                    traceback.print_exc()
                
            elif choice == '2':
                # Обучение на марковском спаме
                try:
                    result = train_on_markov_spam()
                    if result and result[0]:
                        print("\n" + "="*80)
                        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
                        print("="*80)
                        print(f"💾 Модель сохранена в: models/марковский_спам/")
                        print(f"📊 Тип спама: Марковский (сгенерированный)")
                        print(f"🔢 Состояний HMM: {result[0].n_states}")
                        print(f"📖 Размер словаря: {result[1].get_vocabulary_size()}")
                    else:
                        print("\n❌ Обучение не удалось. Проверьте данные.")
                except Exception as e:
                    print(f"\n❌ Ошибка обучения: {e}")
                    import traceback
                    traceback.print_exc()
                
            elif choice == '3':
                # Обучение на смешанном датасете
                try:
                    result = train_on_mixed_dataset()
                    if result and result[0]:
                        print("\n" + "="*80)
                        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
                        print("="*80)
                        print(f"💾 Модель сохранена в: models/смешанный_спам/")
                        print(f"📊 Тип спама: Смешанный (человеческий + марковский)")
                        print(f"🔢 Состояний HMM: {result[0].n_states}")
                        print(f"📖 Размер словаря: {result[1].get_vocabulary_size()}")
                    else:
                        print("\n❌ Обучение не удалось. Проверьте данные.")
                except Exception as e:
                    print(f"\n❌ Ошибка обучения: {e}")
                    import traceback
                    traceback.print_exc()
                
            elif choice == '4':
                # Сравнение всех типов спама
                try:
                    print("⏳ СРАВНЕНИЕ МОДЕЛЕЙ")
                    print("   Это займет время (обучение 3 моделей)...")
                    print("   Подождите...\n")
                    
                    results = compare_spam_types()
                    
                    if results:
                        print("\n" + "="*80)
                        print("✅ СРАВНЕНИЕ ЗАВЕРШЕНО!")
                        print("="*80)
                        print(f"📊 Обучено моделей: {len(results)}")
                        print(f"📈 График сравнения сохранен: model_comparison.png")
                        
                        for name in results.keys():
                            print(f"   • {name}")
                    else:
                        print("\n❌ Не удалось выполнить сравнение")
                        print("💡 Проверьте наличие данных (опция 9)")
                        
                except Exception as e:
                    print(f"\n❌ Ошибка сравнения: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ==================== ТЕСТИРОВАНИЕ ====================
            elif choice == '5':
                # Интерактивное тестирование текущей модели
                if LAST_DETECTOR and LAST_PREPROCESSOR:
                    try:
                        print(f"\n🎮 Запуск интерактивного режима для: {LAST_SPAM_TYPE}")
                        interactive_testing_mode(LAST_DETECTOR, LAST_PREPROCESSOR, LAST_SPAM_TYPE)
                    except KeyboardInterrupt:
                        print("\n⚠️ Тестирование прервано пользователем")
                    except Exception as e:
                        print(f"\n❌ Ошибка тестирования: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("❌ НЕТ ЗАГРУЖЕННОЙ МОДЕЛИ!")
                    print("="*80)
                    print("💡 Доступные действия:")
                    print("   1️⃣ Обучить новую модель (опции 1-3)")
                    print("   2️⃣ Загрузить сохраненную модель (опция 6)")
                    print("="*80)
                
            elif choice == '6':
                # Загрузка сохраненной модели
                print_header("ЗАГРУЗКА СОХРАНЕННОЙ МОДЕЛИ")
                
                try:
                    available = list_available_models()
                    
                    if not available:
                        print("❌ НЕТ СОХРАНЕННЫХ МОДЕЛЕЙ!")
                        print("="*80)
                        print("💡 Сначала обучите модель:")
                        print("   • Опция 1: Человеческий спам")
                        print("   • Опция 2: Марковский спам")
                        print("   • Опция 3: Смешанный датасет")
                        print("="*80)
                    else:
                        print("\n📂 ДОСТУПНЫЕ МОДЕЛИ:")
                        print(f"{'─'*80}")
                        for i, model_name in enumerate(available, 1):
                            # Красивое отображение имени
                            display_name = model_name.replace("_", " ").title()
                            print(f"   {i}. 📦 {display_name}")
                        print(f"{'─'*80}")
                        
                        model_choice = input("\nВыберите номер модели (или Enter для отмены): ").strip()
                        
                        if model_choice.isdigit() and 1 <= int(model_choice) <= len(available):
                            model_name = available[int(model_choice) - 1]
                            spam_type = model_name.replace("_", " ").title()
                            
                            print(f"\n⏳ Загрузка модели: {spam_type}...")
                            result = load_saved_model(spam_type)
                            
                            if result[0]:
                                print("\n" + "="*80)
                                print("✅ МОДЕЛЬ ЗАГРУЖЕНА УСПЕШНО!")
                                print("="*80)
                                print(f"📦 Тип: {spam_type}")
                                print(f"🔢 Состояний: {result[0].n_states}")
                                print(f"📖 Размер словаря: {result[1].get_vocabulary_size()}")
                                print("="*80)
                                
                                choice = input("\n💡 Начать интерактивное тестирование? (y/n): ").strip().lower()
                                if choice in ['y', 'yes', 'д', 'да']:
                                    interactive_testing_mode(result[0], result[1], spam_type)
                        elif model_choice:
                            print("⚠️ Неверный номер модели")
                        else:
                            print("↩️ Отменено")
                        
                except Exception as e:
                    print(f"❌ Ошибка загрузки: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == '7':
                # Тест на внешних данных (unseen data)
                if LAST_DETECTOR and LAST_PREPROCESSOR:
                    try:
                        print(f"\n🧪 Тестирование модели: {LAST_SPAM_TYPE}")
                        print("   На внешнем датасете (unseen data)...")
                        test_on_external_data(LAST_DETECTOR, LAST_PREPROCESSOR, LAST_SPAM_TYPE)
                    except Exception as e:
                        print(f"❌ Ошибка тестирования: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("❌ НЕТ ЗАГРУЖЕННОЙ МОДЕЛИ!")
                    print("="*80)
                    print("💡 Сначала обучите или загрузите модель:")
                    print("   • Опции 1-3: Обучение")
                    print("   • Опция 6: Загрузка сохраненной")
                    print("="*80)
            
            # ==================== УТИЛИТЫ ====================
            elif choice == '8':
                # Генерация марковского спама
                print_header("ГЕНЕРАЦИЯ МАРКОВСКОГО СПАМА")
                
                try:
                    count = input("Сколько текстов сгенерировать? (по умолчанию 100): ").strip()
                    
                    if count:
                        if not count.isdigit():
                            print("⚠️ Неверный ввод! Используем 100 по умолчанию")
                            n = 100
                        else:
                            n = int(count)
                            if n <= 0:
                                print("⚠️ Число должно быть больше 0! Используем 100")
                                n = 100
                            elif n > 1000:
                                print("⚠️ Слишком много! Максимум 1000 за раз")
                                n = 1000
                    else:
                        n = 100
                    
                    print(f"\n⏳ Генерация {n} марковских спам-текстов...")
                    print("   Подождите...\n")
                    
                    create_markov_spam_dataset(n)
                    
                    print("\n" + "="*80)
                    print(f"✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
                    print("="*80)
                    print(f"📊 Создано текстов: {n}")
                    print(f"📂 Папка: data/raw/markov_spam/")
                    print("="*80)
                    
                except ValueError:
                    print("❌ Ошибка! Используем 100 по умолчанию")
                    create_markov_spam_dataset(100)
                except Exception as e:
                    print(f"❌ Ошибка генерации: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == '9':
                # Анализ существующих данных
                print_header("АНАЛИЗ СУЩЕСТВУЮЩИХ ДАННЫХ")
                
                try:
                    datasets = DataLoader.get_available_datasets()
                    
                    print("\n📊 ОБУЧАЮЩИЕ ДАТАСЕТЫ:")
                    print(f"{'─'*80}")
                    print(f"   📂 data/raw/natural/      → {datasets.get('human_natural', 0):4d} файлов")
                    print(f"   📂 data/raw/spam/         → {datasets.get('human_spam', 0):4d} файлов")
                    print(f"   📂 data/raw/markov_spam/  → {datasets.get('markov_spam', 0):4d} файлов")
                    
                    # Проверяем тестовый датасет
                    test_natural_count = 0
                    test_spam_count = 0
                    test_dir = Path('data/test')
                    
                    if test_dir.exists():
                        if (test_dir / 'natural').exists():
                            test_natural_count = len(list((test_dir / 'natural').glob('*.txt')))
                        if (test_dir / 'spam').exists():
                            test_spam_count = len(list((test_dir / 'spam').glob('*.txt')))
                    
                    print(f"{'─'*80}")
                    print("\n🧪 ТЕСТОВЫЕ ДАТАСЕТЫ:")
                    print(f"{'─'*80}")
                    if test_natural_count > 0 or test_spam_count > 0:
                        print(f"   📂 data/test/natural/     → {test_natural_count:4d} файлов")
                        print(f"   📂 data/test/spam/        → {test_spam_count:4d} файлов")
                    else:
                        print("   ❌ Тестовый датасет не создан")
                        print("   💡 Используйте опцию 11 для создания")
                    
                    print(f"{'─'*80}")
                    
                    # Итоги
                    total_train = sum(datasets.values())
                    total_test = test_natural_count + test_spam_count
                    total_all = total_train + total_test
                    
                    print("\n📈 ИТОГО:")
                    print(f"{'─'*80}")
                    print(f"   🎓 Обучающих текстов:  {total_train:4d}")
                    print(f"   🧪 Тестовых текстов:   {total_test:4d}")
                    print(f"   📊 Всего текстов:      {total_all:4d}")
                    print(f"{'─'*80}")
                    
                    # Рекомендации
                    print("\n💡 РЕКОМЕНДАЦИИ:")
                    print(f"{'─'*80}")
                    
                    recommendations = []
                    
                    if total_train < 50:
                        recommendations.append("   ⚠️  Мало обучающих данных (рекомендуется минимум 50)")
                        recommendations.append("       👉 Запустите опцию 10 (Автоподготовка)")
                    else:
                        recommendations.append("   ✅ Достаточно обучающих данных")
                    
                    if total_test < 20:
                        recommendations.append("   ⚠️  Мало тестовых данных (рекомендуется минимум 20)")
                        recommendations.append("       👉 Запустите опцию 11 (Создать тестовый датасет)")
                    else:
                        recommendations.append("   ✅ Достаточно тестовых данных")
                    
                    if datasets.get('markov_spam', 0) == 0:
                        recommendations.append("   💡 Нет марковского спама")
                        recommendations.append("       👉 Запустите опцию 8 (Генерация марковского спама)")
                    
                    for rec in recommendations:
                        print(rec)
                    
                    print(f"{'─'*80}")
                    
                    # Визуализация
                    if total_train > 0:
                        print("\n📊 Создание графика распределения...")
                        try:
                            Visualizer.plot_spam_type_distribution()
                        except Exception as e:
                            print(f"⚠️ Не удалось построить график: {e}")
                
                except Exception as e:
                    print(f"❌ Ошибка анализа данных: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == '10':
                # Автоподготовка всех данных
                print_header("АВТОПОДГОТОВКА ВСЕХ ДАННЫХ")
                
                try:
                    print("🚀 НАЧИНАЕМ АВТОМАТИЧЕСКУЮ ПОДГОТОВКУ ДАТАСЕТОВ")
                    print("="*80)
                    print("   Это включает:")
                    print("   1. 📚 Загрузку статей из Wikipedia")
                    print("   2. 📖 Загрузку книг из Project Gutenberg")
                    print("   3. 📧 Загрузку спам-сообщений")
                    print("   4. 🤖 Генерацию марковского спама")
                    print("\n⏳ Это может занять несколько минут...")
                    print("="*80)
                    
                    confirm = input("\nПродолжить? (y/n): ").strip().lower()
                    
                    if confirm not in ['y', 'yes', 'д', 'да']:
                        print("↩️ Отменено")
                        continue
                    
                    print("\n🔄 Загрузка...\n")
                    
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from prepare_datasets import prepare_all_datasets
                    
                    datasets = prepare_all_datasets()
                    
                    print("\n" + "="*80)
                    print("✅ АВТОПОДГОТОВКА ЗАВЕРШЕНА!")
                    print("="*80)
                    print("📊 Готово к обучению:")
                    for name, count in datasets.items():
                        print(f"   • {name}: {count} текстов")
                    print("="*80)
                    print("💡 Теперь можете запустить обучение (опции 1-3)")
                    
                except ImportError as e:
                    print(f"❌ Не найден модуль prepare_datasets: {e}")
                    print("💡 Убедитесь что файл prepare_datasets.py находится в корне проекта")
                except Exception as e:
                    print(f"❌ Ошибка подготовки данных: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == '11':
                # Создание тестового датасета
                print_header("СОЗДАНИЕ ТЕСТОВОГО ДАТАСЕТА")
                
                try:
                    # Проверяем существующий датасет
                    test_dir = Path('data/test')
                    existing_natural = 0
                    existing_spam = 0
                    
                    if test_dir.exists():
                        if (test_dir / 'natural').exists():
                            existing_natural = len(list((test_dir / 'natural').glob('*.txt')))
                        if (test_dir / 'spam').exists():
                            existing_spam = len(list((test_dir / 'spam').glob('*.txt')))
                    
                    if existing_natural > 0 or existing_spam > 0:
                        print(f"⚠️ Тестовый датасет уже существует:")
                        print(f"   Natural: {existing_natural} файлов")
                        print(f"   Spam: {existing_spam} файлов")
                        
                        overwrite = input("\nПерезаписать? (y/n): ").strip().lower()
                        if overwrite not in ['y', 'yes', 'д', 'да']:
                            print("↩️ Отменено")
                            continue
                    
                    print("\n📝 Создание тестового датасета...")
                    print("   • 20 обычных текстов (natural)")
                    print("   • 20 спам-текстов (spam)")
                    print("\n⏳ Подождите...\n")
                    
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from spam_detector_hmm.src.create_test_dataset import save_test_dataset
                    
                    nat_count, spam_count = save_test_dataset()
                    
                    print("\n" + "="*80)
                    print("✅ ТЕСТОВЫЙ ДАТАСЕТ СОЗДАН!")
                    print("="*80)
                    print(f"📂 Расположение: data/test/")
                    print(f"   • data/test/natural/ → {nat_count} файлов")
                    print(f"   • data/test/spam/    → {spam_count} файлов")
                    print(f"   📊 Всего: {nat_count + spam_count} тестовых текстов")
                    print("="*80)
                    print("💡 Теперь можете использовать:")
                    print("   • Опция 7: Тестирование на внешних данных")
                    print("="*80)
                    
                except ImportError as e:
                    print(f"❌ Не найден модуль create_test_dataset: {e}")
                    print("💡 Убедитесь что файл create_test_dataset.py находится в корне проекта")
                except Exception as e:
                    print(f"❌ Ошибка создания тестового датасета: {e}")
                    import traceback
                    traceback.print_exc()
                
            elif choice == '12':
                # Выход из программы
                print("\n" + "="*80)
                print("👋 СПАСИБО ЗА ИСПОЛЬЗОВАНИЕ ДЕТЕКТОРА СПАМА!")
                print("="*80)
                print("\n📚 Использованные технологии:")
                print("   • Скрытые марковские модели (HMM)")
                print("   • Алгоритм Баума-Велша (обучение EM-алгоритм)")
                print("   • Алгоритм Витерби (декодирование состояний)")
                print("   • POS-теггинг (извлечение признаков)")
                print("   • Марковские цепи (генерация спама)")
                
                print("\n🎓 Разработано для исследований в области:")
                print("   • Обработки естественного языка (NLP)")
                print("   • Машинного обучения")
                print("   • Детекции спама")
                
                print("\n💡 Удачи в ваших исследованиях!")
                print("="*80 + "\n")
                break
                
            else:
                # Неверный выбор
                print("\n" + "="*80)
                print("⚠️ НЕВЕРНЫЙ ВЫБОР!")
                print("="*80)
                print("💡 Доступные опции: 1-12")
                print("   Введите число от 1 до 12 из меню выше")
                print("="*80)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("⚠️ ПРОГРАММА ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ (Ctrl+C)")
            print("="*80)
            
            confirm = input("\nВы действительно хотите выйти? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', 'д', 'да']:
                print("\n👋 До свидания!")
                break
            else:
                print("\n↩️ Продолжаем работу...")
                continue
                
        except Exception as e:
            print("\n" + "="*80)
            print(f"❌ НЕОЖИДАННАЯ ОШИБКА")
            print("="*80)
            print(f"Ошибка: {e}")
            print("\n📋 Детали ошибки:")
            import traceback
            traceback.print_exc()
            print("="*80)
            print("💡 Попробуйте еще раз или сообщите об ошибке")
        
        # Пауза перед следующей итерацией
        input("\n↵ Нажмите Enter для продолжения...")
        print("\n" * 2)  # Пустые строки для читаемости


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Программа завершена пользователем")
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*80)
        print("Программа завершена")
        print("="*80)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
