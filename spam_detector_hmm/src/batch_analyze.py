"""
Скрипт для массового анализа текстов и сбора статистики
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from preprocessor import TextPreprocessor
from hmm_detector import SpamDetectorHMM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

class BatchAnalyzer:
    """Класс для массового анализа текстов"""
    
    def __init__(self, detector, preprocessor):
        self.detector = detector
        self.preprocessor = preprocessor
        self.results = []
    
    def analyze_single(self, text, true_label=None):
        """
        Анализ одного текста с полной статистикой
        
        Returns:
            dict с результатами
        """
        seq = self.preprocessor.text_to_sequence(text)
        result = self.detector.predict_proba(seq)
        
        # Дополнительная статистика
        tokens = self.preprocessor.tokenize(text)
        pos_tags = self.preprocessor.extract_pos_features(text)
        unk_idx = self.preprocessor.vocab.get('UNK', self.preprocessor.get_vocabulary_size() - 1)
        unk_ratio = float((seq == unk_idx).sum()) / len(seq) if len(seq) > 0 else 0.0
        
        # Витерби для обеих моделей
        vit_nat = self.detector.decode_viterbi(seq, 'natural')
        vit_spam = self.detector.decode_viterbi(seq, 'spam')
        
        # Постериоры
        gamma_nat = self.detector.get_posteriors(seq, 'natural')
        gamma_spam = self.detector.get_posteriors(seq, 'spam')
        
        analysis = {
            'text_preview': text[:100],
            'text_length': len(text),
            'tokens_count': len(tokens),
            'sequence_length': len(seq),
            'unique_features': len(np.unique(seq)),
            'unk_ratio': unk_ratio,
            'prediction': result['prediction'],
            'prob_natural': result['prob_natural'],
            'prob_spam': result['prob_spam'],
            'log_prob_natural': result['log_prob_natural'],
            'log_prob_spam': result['log_prob_spam'],
            'log_diff': abs(result['log_prob_natural'] - result['log_prob_spam']),
            'viterbi_natural_logprob': vit_nat['log_probability'],
            'viterbi_spam_logprob': vit_spam['log_probability'],
            'viterbi_natural_states_used': vit_nat['n_states_used'],
            'viterbi_spam_states_used': vit_spam['n_states_used'],
            'avg_posterior_natural': gamma_nat.mean(axis=0).tolist() if gamma_nat.size > 0 else [],
            'avg_posterior_spam': gamma_spam.mean(axis=0).tolist() if gamma_spam.size > 0 else [],
            'true_label': true_label,
            'correct': (result['prediction'] == true_label) if true_label else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(analysis)
        return analysis
    
    def analyze_batch(self, texts, true_labels=None, show_progress=True):
        """
        Массовый анализ списка текстов
        
        Args:
            texts: список текстов
            true_labels: список истинных меток (опционально)
            show_progress: показывать прогресс
        """
        if true_labels is None:
            true_labels = [None] * len(texts)
        
        print(f"\n📊 Начало анализа {len(texts)} текстов...")
        
        for i, (text, label) in enumerate(zip(texts, true_labels)):
            if show_progress and (i + 1) % 10 == 0:
                print(f"   Обработано: {i + 1}/{len(texts)}", end='\r')
            
            try:
                self.analyze_single(text, label)
            except Exception as e:
                print(f"\n⚠️  Ошибка при анализе текста {i}: {e}")
        
        if show_progress:
            print(f"\n✓ Анализ завершен: {len(self.results)} текстов")
    
    def get_statistics(self):
        """Получить агрегированную статистику"""
        if not self.results:
            print("⚠️  Нет результатов для анализа")
            return None
        
        df = pd.DataFrame(self.results)
        
        stats = {
            'total_texts': len(df),
            'predictions': {
                'natural': int((df['prediction'] == 'natural').sum()),
                'spam': int((df['prediction'] == 'spam').sum())
            },
            'avg_prob_natural': float(df['prob_natural'].mean()),
            'avg_prob_spam': float(df['prob_spam'].mean()),
            'avg_sequence_length': float(df['sequence_length'].mean()),
            'avg_unk_ratio': float(df['unk_ratio'].mean()),
            'avg_log_diff': float(df['log_diff'].mean())
        }
        
        # Если есть истинные метки - добавляем метрики качества
        if df['true_label'].notna().any():
            valid_df = df[df['true_label'].notna()]
            stats['accuracy'] = float((valid_df['prediction'] == valid_df['true_label']).mean())
            stats['correct_predictions'] = int((valid_df['prediction'] == valid_df['true_label']).sum())
            stats['incorrect_predictions'] = int((valid_df['prediction'] != valid_df['true_label']).sum())
        
        return stats
    
    def export_results(self, filepath='batch_results.json'):
        """Экспорт результатов в JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ Результаты экспортированы: {filepath}")
    
    def export_csv(self, filepath='batch_results.csv'):
        """Экспорт в CSV для Excel"""
        df = pd.DataFrame(self.results)
        # Убираем сложные поля для CSV
        simple_df = df.drop(columns=['avg_posterior_natural', 'avg_posterior_spam'], errors='ignore')
        simple_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✓ Результаты экспортированы: {filepath}")
    
    def plot_statistics(self):
        """Визуализация статистики"""
        if not self.results:
            print("⚠️  Нет результатов для визуализации")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Распределение предсказаний
        pred_counts = df['prediction'].value_counts()
        axes[0, 0].bar(pred_counts.index, pred_counts.values, color=['green', 'red'])
        axes[0, 0].set_title('Распределение предсказаний')
        axes[0, 0].set_ylabel('Количество')
        
        # 2. Гистограмма вероятностей спама
        axes[0, 1].hist(df['prob_spam'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_title('Распределение P(Spam|X)')
        axes[0, 1].set_xlabel('Вероятность')
        axes[0, 1].set_ylabel('Частота')
        
        # 3. Scatter: длина текста vs вероятность спама
        axes[0, 2].scatter(df['sequence_length'], df['prob_spam'], alpha=0.5, c=df['prediction'].map({'natural': 'green', 'spam': 'red'}))
        axes[0, 2].set_title('Длина последовательности vs P(Spam)')
        axes[0, 2].set_xlabel('Длина последовательности')
        axes[0, 2].set_ylabel('P(Spam)')
        
        # 4. UNK ratio
        axes[1, 0].hist(df['unk_ratio'], bins=20, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 0].set_title('Доля UNK-токенов')
        axes[1, 0].set_xlabel('UNK Ratio')
        axes[1, 0].set_ylabel('Частота')
        
        # 5. Log-likelihood разница
        axes[1, 1].hist(df['log_diff'], bins=30, edgecolor='black', alpha=0.7, color='blue')
        axes[1, 1].set_title('Разность лог-правдоподобий')
        axes[1, 1].set_xlabel('|LL_natural - LL_spam|')
        axes[1, 1].set_ylabel('Частота')
        
        # 6. Confusion matrix (если есть истинные метки)
        if df['true_label'].notna().any():
            valid_df = df[df['true_label'].notna()]
            cm = confusion_matrix(valid_df['true_label'], valid_df['prediction'], labels=['natural', 'spam'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2],
                       xticklabels=['Natural', 'Spam'], yticklabels=['Natural', 'Spam'])
            axes[1, 2].set_title('Confusion Matrix')
            axes[1, 2].set_ylabel('True')
            axes[1, 2].set_xlabel('Predicted')
        else:
            axes[1, 2].text(0.5, 0.5, 'Нет истинных меток', ha='center', va='center', fontsize=12)
            axes[1, 2].set_title('Confusion Matrix (N/A)')
        
        plt.tight_layout()
        plt.savefig('batch_statistics.png', dpi=300, bbox_inches='tight')
        print("✓ График сохранён: batch_statistics.png")
        plt.show()

def main():
    """Главная функция для массового анализа"""
    print("="*70)
    print("МАССОВЫЙ АНАЛИЗ ТЕКСТОВ НА СПАМ")
    print("="*70)
    
    # Загрузка обученных моделей
    print("\n📦 Загрузка обученной модели...")
    detector = SpamDetectorHMM()
    detector.load('models/')
    
    preprocessor = TextPreprocessor(feature_type='pos', n_symbols=50)
    
    # Загрузка данных (используем все доступные)
    natural_texts, spam_texts = DataLoader.load_train_data()
    
    # Строим словарь на всех данных (для inference)
    all_texts = natural_texts + spam_texts
    preprocessor.build_vocabulary(all_texts)
    
    # Создаём анализатор
    analyzer = BatchAnalyzer(detector, preprocessor)
    
    # Анализ всех текстов
    texts = natural_texts + spam_texts
    labels = ['natural'] * len(natural_texts) + ['spam'] * len(spam_texts)
    
    analyzer.analyze_batch(texts, labels)
    
    # Статистика
    print("\n" + "="*70)
    print("СТАТИСТИКА")
    print("="*70)
    stats = analyzer.get_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Экспорт
    analyzer.export_results('batch_results.json')
    analyzer.export_csv('batch_results.csv')
    
    # Визуализация
    analyzer.plot_statistics()
    
    # Classification report
    df = pd.DataFrame(analyzer.results)
    if df['true_label'].notna().any():
        valid_df = df[df['true_label'].notna()]
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(valid_df['true_label'], valid_df['prediction'], 
                                   target_names=['Natural', 'Spam'], zero_division=0))
    
    print("\n✅ Анализ завершен!")
    print(f"📁 Файлы:")
    print(f"   • batch_results.json - подробные результаты")
    print(f"   • batch_results.csv - для Excel")
    print(f"   • batch_statistics.png - графики")

if __name__ == "__main__":
    main()
