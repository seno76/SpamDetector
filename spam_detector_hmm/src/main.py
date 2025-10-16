"""
Главный скрипт для обучения и тестирования детектора спама
"""
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from preprocessor import TextPreprocessor
from hmm_detector import SpamDetectorHMM
from visualizer import Visualizer

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_header(text):
    """Красивый заголовок"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def analyze_text_detail(detector, preprocessor, text, true_label):
    """
    Детальный анализ одного текста
    
    Args:
        detector: обученный детектор
        preprocessor: препроцессор
        text: текст для анализа
        true_label: истинная метка
    """
    print("\n" + "-"*70)
    print(f"📄 ТЕКСТ (первые 100 символов):")
    print(f"   {text[:100]}...")
    print(f"   Истинная метка: {true_label.upper()}")
    print("-"*70)
    
    # Преобразуем текст в последовательность
    sequence = preprocessor.text_to_sequence(text)
    print(f"\n📊 Статистика последовательности:")
    print(f"   Длина: {len(sequence)}")
    print(f"   Уникальных символов: {len(np.unique(sequence))}")
    print(f"   Первые 20 символов: {sequence[:20]}")
    
    # Классификация
    result = detector.predict_proba(sequence)
    print(f"\n🎯 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
    print(f"   Предсказание: {result['prediction'].upper()}")
    print(f"   Log P(X|Natural): {result['log_prob_natural']:.2f}")
    print(f"   Log P(X|Spam):    {result['log_prob_spam']:.2f}")
    print(f"   P(Natural|X):     {result['prob_natural']:.4f}")
    print(f"   P(Spam|X):        {result['prob_spam']:.4f}")
    
    # Декодирование Витерби для обеих моделей
    print(f"\n🔍 ДЕКОДИРОВАНИЕ ВИТЕРБИ:")
    
    for model_type in ['natural', 'spam']:
        viterbi_result = detector.decode_viterbi(sequence, model_type)
        states = viterbi_result['states']
        
        print(f"\n   Модель: {model_type.upper()}")
        print(f"   Log-вероятность: {viterbi_result['log_probability']:.2f}")
        print(f"   Используется состояний: {viterbi_result['n_states_used']}/{detector.n_states}")
        print(f"   Последовательность состояний (первые 30): {states[:30]}")
        
        # Статистика по состояниям
        unique, counts = np.unique(states, return_counts=True)
        print(f"   Распределение по состояниям:")
        for state, count in zip(unique, counts):
            percentage = (count / len(states)) * 100
            print(f"      Состояние {state}: {count} раз ({percentage:.1f}%)")
    
    # Правильность классификации
    is_correct = (result['prediction'] == true_label)
    print(f"\n{'✅ ПРАВИЛЬНО' if is_correct else '❌ ОШИБКА'}")
    print("-"*70)

def main():
    """Основная функция"""
    
    print_header("ДЕТЕКТОР ПОИСКОВОГО СПАМА НА ОСНОВЕ HMM")
    print("Скрытые марковские модели: алгоритмы Баума-Велша и Витерби")
    
    # ==================== ШАГ 1: ЗАГРУЗКА ДАННЫХ ====================
    print_header("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
    
    natural_texts, spam_texts = DataLoader.load_train_data()
    
    if len(natural_texts) < 3 or len(spam_texts) < 3:
        print("\n⚠ Недостаточно данных! Используем примеры из sample_texts.json")
        sample_data = DataLoader.load_sample_data()
        natural_texts = sample_data['natural'] * 3  # Дублируем для объёма
        spam_texts = sample_data['spam'] * 3
        test_texts_examples = sample_data['test']
    else:
        test_texts_examples = []
    
    print(f"\n📚 Итого данных:")
    print(f"   Обычных текстов: {len(natural_texts)}")
    print(f"   Спам-текстов: {len(spam_texts)}")
    
    # ==================== ШАГ 2: РАЗДЕЛЕНИЕ ДАННЫХ (ДО ПРЕДОБРАБОТКИ!) ====================
    print_header("ШАГ 2: РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
    
    # КРИТИЧЕСКИ ВАЖНО: Разделяем ПЕРЕД построением словаря!
    natural_train_texts, natural_test_texts = train_test_split(
        natural_texts, test_size=0.2, random_state=42
    )
    spam_train_texts, spam_test_texts = train_test_split(
        spam_texts, test_size=0.2, random_state=42
    )
    
    print(f"\n📦 Размеры текстовых выборок:")
    print(f"   Обучающая выборка:")
    print(f"      Обычные: {len(natural_train_texts)}")
    print(f"      Спам: {len(spam_train_texts)}")
    print(f"   Тестовая выборка:")
    print(f"      Обычные: {len(natural_test_texts)}")
    print(f"      Спам: {len(spam_test_texts)}")
    
    # ==================== ШАГ 3: ПРЕДОБРАБОТКА ====================
    print_header("ШАГ 3: ПРЕДОБРАБОТКА ТЕКСТОВ")
    
    # Создаём препроцессор
    preprocessor = TextPreprocessor(feature_type='pos', n_symbols=50)
    
    # КРИТИЧЕСКИ ВАЖНО: Строим словарь ТОЛЬКО на обучающих данных!
    train_texts = natural_train_texts + spam_train_texts
    print(f"\n📖 Построение словаря на {len(train_texts)} обучающих текстах...")
    preprocessor.build_vocabulary(train_texts)
    
    print(f"\n🔤 Размер словаря: {preprocessor.get_vocabulary_size()}")
    
    # Преобразуем тексты в последовательности
    print(f"\n🔄 Преобразование текстов в последовательности...")
    
    natural_train = preprocessor.texts_to_sequences(natural_train_texts)
    spam_train = preprocessor.texts_to_sequences(spam_train_texts)
    natural_test = preprocessor.texts_to_sequences(natural_test_texts)
    spam_test = preprocessor.texts_to_sequences(spam_test_texts)
    
    # Статистика по последовательностям
    natural_train_lengths = [len(seq) for seq in natural_train]
    spam_train_lengths = [len(seq) for seq in spam_train]
    
    print(f"\n📊 Статистика длин последовательностей:")
    print(f"   Обучающая - Обычные тексты:")
    print(f"      Средняя длина: {np.mean(natural_train_lengths):.1f}")
    print(f"      Мин/Макс: {np.min(natural_train_lengths)}/{np.max(natural_train_lengths)}")
    print(f"   Обучающая - Спам-тексты:")
    print(f"      Средняя длина: {np.mean(spam_train_lengths):.1f}")
    print(f"      Мин/Макс: {np.min(spam_train_lengths)}/{np.max(spam_train_lengths)}")
    
    # Проверка индексов
    print(f"\n🔍 Проверка согласованности индексов:")
    vocab_size = preprocessor.get_vocabulary_size()
    
    all_train = natural_train + spam_train
    all_test = natural_test + spam_test
    
    max_train_idx = max([seq.max() for seq in all_train if len(seq) > 0])
    max_test_idx = max([seq.max() for seq in all_test if len(seq) > 0])
    
    print(f"   Размер словаря: {vocab_size}")
    print(f"   Max индекс в Train: {max_train_idx}")
    print(f"   Max индекс в Test: {max_test_idx}")
    
    if max_train_idx >= vocab_size:
        print(f"   ⚠️  ОШИБКА: Train индексы выходят за границы!")
    if max_test_idx >= vocab_size:
        print(f"   ⚠️  ОШИБКА: Test индексы выходят за границы!")
    
    if max_train_idx < vocab_size and max_test_idx < vocab_size:
        print(f"   ✅ Все индексы в допустимых границах")
    
    # ==================== ШАГ 4: ОБУЧЕНИЕ HMM ====================
    print_header("ШАГ 4: ОБУЧЕНИЕ СКРЫТЫХ МАРКОВСКИХ МОДЕЛЕЙ")
    print("\nАлгоритм: Баума-Велша (EM-алгоритм для HMM)")
    
    # Создаём детектор
    detector = SpamDetectorHMM(n_states=3, n_iter=100, tol=1e-2)
    
    # Обучаем на тренировочных данных
    detector.fit(
        natural_sequences=natural_train,
        spam_sequences=spam_train,
        n_features=vocab_size
    )
    
    # Сохраняем модели
    detector.save('models/')
    
    # ==================== ШАГ 5: АНАЛИЗ МОДЕЛЕЙ ====================
    print_header("ШАГ 5: АНАЛИЗ ОБУЧЕННЫХ МОДЕЛЕЙ")
    
    print("\n📈 Матрица переходов для ОБЫЧНЫХ текстов:")
    print(detector.get_transition_matrix('natural'))
    
    print("\n📈 Матрица переходов для СПАМ текстов:")
    print(detector.get_transition_matrix('spam'))
    
    # Визуализация матриц переходов
    Visualizer.plot_transition_matrices(detector)
    
    # Визуализация распределений эмиссий
    print("\n🎨 Создание визуализаций распределений эмиссий...")
    Visualizer.plot_emission_distributions(detector, top_n=min(10, vocab_size))
    
    # ==================== ШАГ 6: ТЕСТИРОВАНИЕ ====================
    print_header("ШАГ 6: ТЕСТИРОВАНИЕ НА ОТЛОЖЕННОЙ ВЫБОРКЕ")
    
    # Объединяем тестовые последовательности
    test_sequences = natural_test + spam_test
    true_labels = ['natural'] * len(natural_test) + ['spam'] * len(spam_test)
    
    # Предсказания
    print(f"\n🔮 Выполнение предсказаний на {len(test_sequences)} примерах...")
    predictions = detector.predict(test_sequences)
    
    # Метрики
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label='spam', zero_division=0)
    recall = recall_score(true_labels, predictions, pos_label='spam', zero_division=0)
    f1 = f1_score(true_labels, predictions, pos_label='spam', zero_division=0)
    
    print(f"\n📊 МЕТРИКИ КАЧЕСТВА:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Визуализация результатов
    Visualizer.plot_classification_comparison(predictions, true_labels)
    
    # ==================== ШАГ 7: ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРОВ ====================
    print_header("ШАГ 7: ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРОВ (АЛГОРИТМ ВИТЕРБИ)")
    
    # Анализируем по одному примеру каждого класса из тестовой выборки
    if len(natural_test_texts) > 0 and len(spam_test_texts) > 0:
        print("\n🔬 Детальный анализ с использованием алгоритма Витерби:")
        
        analyze_text_detail(
            detector, preprocessor, 
            natural_test_texts[0], 
            'natural'
        )
        
        analyze_text_detail(
            detector, preprocessor,
            spam_test_texts[0],
            'spam'
        )
    
    # ==================== ШАГ 8: ПРИМЕРЫ ИЗ SAMPLE ====================
    if test_texts_examples:
        print_header("ШАГ 8: АНАЛИЗ ДОПОЛНИТЕЛЬНЫХ ПРИМЕРОВ")
        
        for idx, text in enumerate(test_texts_examples, 1):
            print(f"\n{'='*70}")
            print(f"ПРИМЕР {idx}")
            print(f"{'='*70}")
            print(f"Текст: {text}")
            
            sequence = preprocessor.text_to_sequence(text)
            result = detector.predict_proba(sequence)
            
            print(f"\n🎯 Результат:")
            print(f"   Предсказание: {result['prediction'].upper()}")
            print(f"   Уверенность (Natural): {result['prob_natural']:.4f}")
            print(f"   Уверенность (Spam):    {result['prob_spam']:.4f}")
            
            # Витерби для примера
            viterbi_natural = detector.decode_viterbi(sequence, 'natural')
            viterbi_spam = detector.decode_viterbi(sequence, 'spam')
            
            print(f"\n   Витерби (Natural model): log-prob = {viterbi_natural['log_probability']:.2f}")
            print(f"   Витерби (Spam model):    log-prob = {viterbi_spam['log_probability']:.2f}")
    
    # ==================== ЗАВЕРШЕНИЕ ====================
    print_header("✅ АНАЛИЗ ЗАВЕРШЁН")
    
    print("\n📁 Созданные файлы:")
    print("   • models/natural_model.pkl - обученная модель для обычных текстов")
    print("   • models/spam_model.pkl - обученная модель для спама")
    print("   • transition_matrices.png - визуализация матриц переходов")
    print("   • emission_distributions.png - распределения эмиссий")
    print("   • confusion_matrix.png - матрица ошибок")
    
    print("\n💡 Ключевые выводы:")
    print("   1. Алгоритм Баума-Велша успешно обучил две HMM модели")
    print("   2. Модели научились различать паттерны естественного текста и спама")
    print("   3. Алгоритм Витерби позволяет декодировать скрытые состояния")
    print(f"   4. Точность классификации: {accuracy*100:.2f}%")
    
    return detector, preprocessor

if __name__ == "__main__":
    try:
        detector, preprocessor = main()
        
        # Интерактивный режим
        print("\n" + "="*70)
        print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("="*70)
        print("Введите текст для анализа (или 'exit' для выхода):\n")
        
        while True:
            user_input = input("\n📝 Ваш текст: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("\n👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            # Анализ введённого текста
            sequence = preprocessor.text_to_sequence(user_input)
            result = detector.predict_proba(sequence)
            
            print(f"\n{'='*70}")
            print(f"🎯 РЕЗУЛЬТАТ АНАЛИЗА:")
            print(f"{'='*70}")
            print(f"Классификация: {result['prediction'].upper()}")
            print(f"Вероятность (Natural): {result['prob_natural']:.4f}")
            print(f"Вероятность (Spam):    {result['prob_spam']:.4f}")
            print(f"Log-likelihood разница: {abs(result['log_prob_natural'] - result['log_prob_spam']):.2f}")
            
            # Краткая интерпретация
            if result['prediction'] == 'spam':
                confidence = result['prob_spam']
                if confidence > 0.9:
                    print("\n⚠️  ВЫСОКАЯ вероятность спама!")
                elif confidence > 0.7:
                    print("\n⚠️  Средняя вероятность спама")
                else:
                    print("\n⚠️  Низкая вероятность спама (граничный случай)")
            else:
                print("\n✅ Текст выглядит естественным")
            
            print(f"{'='*70}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()