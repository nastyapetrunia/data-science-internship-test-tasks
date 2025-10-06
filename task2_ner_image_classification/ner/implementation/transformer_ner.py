import spacy
from spacy.training.example import Example
from task2_ner_image_classification.ner.interface.ner_base import NERInterface

class TransformerNER(NERInterface):
    def __init__(self, model_name="google/mobilebert-uncased"):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("transformer", config={"model": {"name": model_name}})
        self.ner = self.nlp.add_pipe("ner")

    def train(self, train_data, n_iter=200, patience=5):
        """
        Train the NER model with early stopping.
        train_data: list of tuples [(text, {"entities": [[start, end, label], ...]}), ...]
        n_iter: maximum number of iterations
        patience: early stopping patience
        """
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        optimizer = self.nlp.initialize()
        best_loss = float("inf")
        counter = 0

        for i in range(n_iter):
            losses = {}
            for text, annotations in train_data:
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                self.nlp.update([example], sgd=optimizer, losses=losses)

            current_loss = losses.get("ner", 0.0)
            print(f"Iteration {i+1}, Losses: {losses}")

            if current_loss < best_loss:
                best_loss = current_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at iteration {i+1} (no improvement for {patience} iters)")
                    break

        print(f"Training finished. Best loss: {best_loss:.6f}")


    def load(self, model_path: str):
        self.nlp = spacy.load(model_path)

    def save(self, model_path: str):
        self.nlp.to_disk(model_path)

    def predict(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
        