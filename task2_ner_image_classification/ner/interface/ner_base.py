from abc import ABC, abstractmethod

class NERInterface(ABC):
    @abstractmethod
    def train(self, train_data, n_iter=50):
        """
        train_data: list of tuples (text, {"entities": [(start, end, label)]})
        """
        pass

    @abstractmethod
    def predict(self, text: str):
        """Return list of detected entities"""
        pass

    @abstractmethod
    def save(self, model_path: str):
        """Save the NER model"""
        pass

    @abstractmethod
    def load(self, model_path: str):
        """Load the NER model"""
        pass
