"""
Детектор спама на основе скрытых марковских моделей (с защитой от NaN)
"""
import numpy as np
from hmmlearn import hmm
import joblib
from pathlib import Path

class SpamDetectorHMM:
    """
    Классификатор спама использующий две HMM:
    - одну для обычных текстов
    - одну для спам-текстов
    """
    
    def __init__(self, n_states=6, n_iter=150, tol=1e-3, random_state=42, smoothing_eps=1e-6):
        """
        Args:
            n_states: количество скрытых состояний в HMM
            n_iter: максимальное число итераций алгоритма Баума-Велша
            tol: порог сходимости для обучения
            random_state: seed для воспроизводимости
            smoothing_eps: эпсилон для сглаживания матриц вероятностей
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.smoothing_eps = float(smoothing_eps)
        
        self.natural_model = None
        self.spam_model = None
        
        self.n_features = None
        self.is_fitted = False
    
    def _prepare_sequences(self, sequences):
        """
        Подготовка последовательностей для hmmlearn
        """
        X = np.concatenate([seq.reshape(-1, 1) for seq in sequences])
        lengths = [len(seq) for seq in sequences]
        return X, lengths

    def _apply_smoothing(self, model):
        """
        Сглаживание параметров HMM: заменяем нули на eps и нормируем построчно.
        Это предотвращает нулевые вероятности, ведущие к -inf и NaN.
        """
        eps = self.smoothing_eps
        
        # startprob_
        sp = model.startprob_.clip(min=eps)
        model.startprob_ = sp / sp.sum()

        # transmat_
        tm = model.transmat_.clip(min=eps)
        tm = tm / tm.sum(axis=1, keepdims=True)
        model.transmat_ = tm

        # emissionprob_
        em = model.emissionprob_.clip(min=eps)
        em = em / em.sum(axis=1, keepdims=True)
        model.emissionprob_ = em

    def fit(self, natural_sequences, spam_sequences, n_features):
        """
        Обучение двух HMM: на обычных текстах и на спаме

        Args:
            natural_sequences: список последовательностей для обычных текстов
            spam_sequences: список последовательностей для спам-текстов
            n_features: общий размер алфавита наблюдений (кол-во уникальных символов)
        """
        self.n_features = int(n_features)

        print(f"\n🔧 Обучение модели для ОБЫЧНЫХ текстов...")
        print(f"   Количество последовательностей: {len(natural_sequences)}")
        print(f"   Состояний: {self.n_states}, Итераций: {self.n_iter}")

        self.natural_model = hmm.CategoricalHMM(
            n_components=self.n_states,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params="ste",  # инициализируем startprob, transmat, emissionprob
            params="ste"
        )
        # Критично: фиксируем размер алфавита
        self.natural_model.n_features = self.n_features

        X_nat, len_nat = self._prepare_sequences(natural_sequences)
        self.natural_model.fit(X_nat, lengths=len_nat)
        # Сглаживание, чтобы избежать нулей
        self._apply_smoothing(self.natural_model)
        nat_ll = self.natural_model.score(X_nat, len_nat)
        print(f"   ✓ Модель обучена. Финальный log-likelihood: {nat_ll:.2f}")
        # print(f"   → emission shape (natural): {self.natural_model.emissionprob_.shape}")

        print(f"\n🔧 Обучение модели для СПАМ текстов...")
        print(f"   Количество последовательностей: {len(spam_sequences)}")

        self.spam_model = hmm.CategoricalHMM(
            n_components=self.n_states,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params="ste",
            params="ste"
        )
        self.spam_model.n_features = self.n_features

        X_spam, len_spam = self._prepare_sequences(spam_sequences)
        self.spam_model.fit(X_spam, lengths=len_spam)
        # Сглаживание
        self._apply_smoothing(self.spam_model)
        spam_ll = self.spam_model.score(X_spam, len_spam)
        print(f"   ✓ Модель обучена. Финальный log-likelihood: {spam_ll:.2f}")
        # print(f"   → emission shape (spam): {self.spam_model.emissionprob_.shape}")

        self.is_fitted = True
        print("\n✓ Обучение завершено успешно!")

    @staticmethod
    def _safe_probs_from_logs(log_a, log_b):
        """
        Безопасное преобразование двух лог-правдоподобий в вероятности классов.
        Обрабатывает случаи -inf, чтобы не получить NaN.
        """
        a_inf = np.isneginf(log_a)
        b_inf = np.isneginf(log_b)

        # Оба -inf → равные вероятности 0.5/0.5
        if a_inf and b_inf:
            return 0.5, 0.5

        # Один -inf → другой класс 1.0
        if a_inf and not b_inf:
            return 0.0, 1.0
        if b_inf and not a_inf:
            return 1.0, 0.0

        # Оба конечные → стабильный softmax через log-sum-exp
        lse = np.logaddexp(log_a, log_b)  # log(exp(a)+exp(b))
        p_a = float(np.exp(log_a - lse))
        p_b = float(np.exp(log_b - lse))

        # Защита от численных артефактов
        if not np.isfinite(p_a) or not np.isfinite(p_b):
            return 0.5, 0.5

        return p_a, p_b
    
    def predict(self, sequences):
        """
        Классификация списка последовательностей
        
        Args:
            sequences: список numpy arrays
        Returns:
            список предсказаний ('natural' или 'spam')
        """
        predictions = []
        for seq in sequences:
            result = self.predict_proba(seq)
            predictions.append(result['prediction'])
        return predictions


    def predict_proba(self, sequence):
        """
        Вычисление вероятностей для одной последовательности

        Args:
            sequence: numpy array с последовательностью (индексы 0..n_features-1)
        Returns:
            dict с log-правдоподобиями и вероятностями классов
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена! Вызовите fit() сначала.")

        if sequence is None or len(sequence) == 0:
            # Пустая последовательность — возврат нейтрального результата
            return {
                'log_prob_natural': float('-inf'),
                'log_prob_spam': float('-inf'),
                'prob_natural': 0.5,
                'prob_spam': 0.5,
                'prediction': 'natural'
            }

        # Защита от индексов вне словаря
        max_idx = int(sequence.max())
        if max_idx >= self.n_features or sequence.min() < 0:
            # Клиппинг или ошибка. Клиппим в UNK (последний индекс).
            seq = sequence.copy()
            unk_idx = self.n_features - 1
            seq[seq >= self.n_features] = unk_idx
            seq[seq < 0] = unk_idx
            X = seq.reshape(-1, 1)
        else:
            X = sequence.reshape(-1, 1)

        # Вычисление лог-правдоподобий
        try:
            log_prob_natural = float(self.natural_model.score(X))
        except Exception:
            log_prob_natural = float('-inf')

        try:
            log_prob_spam = float(self.spam_model.score(X))
        except Exception:
            log_prob_spam = float('-inf')

        # Безопасные вероятности
        prob_natural, prob_spam = self._safe_probs_from_logs(log_prob_natural, log_prob_spam)

        return {
            'log_prob_natural': log_prob_natural,
            'log_prob_spam': log_prob_spam,
            'prob_natural': prob_natural,
            'prob_spam': prob_spam,
            'prediction': 'natural' if prob_natural >= prob_spam else 'spam'
        }

    def decode_viterbi(self, sequence, model_type='natural'):
        """
        Декодирование последовательности скрытых состояний алгоритмом Витерби
        """
        model = self.natural_model if model_type == 'natural' else self.spam_model

        if sequence is None or len(sequence) == 0:
            return {
                'log_probability': float('-inf'),
                'states': np.array([], dtype=int),
                'n_states_used': 0
            }

        X = sequence.reshape(-1, 1)
        try:
            log_prob, states = model.decode(X, algorithm="viterbi")
            return {
                'log_probability': float(log_prob),
                'states': states,
                'n_states_used': len(np.unique(states))
            }
        except Exception:
            return {
                'log_probability': float('-inf'),
                'states': np.zeros(len(sequence), dtype=int),
                'n_states_used': 0
            }

    def get_posteriors(self, sequence, model_type='natural'):
        """
        Постериорные вероятности состояний gamma_t(i) для последовательности.
        Возвращает массив [T, n_states].
        """
        model = self.natural_model if model_type == 'natural' else self.spam_model
        if sequence is None or len(sequence) == 0:
            return np.zeros((0, self.n_states), dtype=float)
        X = sequence.reshape(-1, 1)
        try:
            gamma = model.predict_proba(X)
            return np.asarray(gamma)
        except Exception:
            return np.zeros((len(sequence), self.n_states), dtype=float)

    def get_transition_matrix(self, model_type='natural'):
        model = self.natural_model if model_type == 'natural' else self.spam_model
        return model.transmat_
    
    def get_emission_matrix(self, model_type='natural'):
        model = self.natural_model if model_type == 'natural' else self.spam_model
        return model.emissionprob_
    
    def save(self, path='models/'):
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.natural_model, f'{path}/natural_model.pkl')
        joblib.dump(self.spam_model, f'{path}/spam_model.pkl')
        metadata = {
            'n_states': self.n_states,
            'n_features': self.n_features,
            'is_fitted': self.is_fitted,
            'smoothing_eps': self.smoothing_eps
        }
        joblib.dump(metadata, f'{path}/metadata.pkl')
        print(f"✓ Модели сохранены в {path}")
    
    def load(self, path='models/'):
        self.natural_model = joblib.load(f'{path}/natural_model.pkl')
        self.spam_model = joblib.load(f'{path}/spam_model.pkl')
        metadata = joblib.load(f'{path}/metadata.pkl')
        self.n_states = metadata['n_states']
        self.n_features = metadata['n_features']
        self.is_fitted = metadata['is_fitted']
        self.smoothing_eps = metadata.get('smoothing_eps', 1e-6)
        print(f"✓ Модели загружены из {path}")
