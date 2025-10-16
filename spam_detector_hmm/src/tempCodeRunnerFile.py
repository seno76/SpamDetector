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