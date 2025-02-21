from transformers import AutoModel, AutoTokenizer 
import torch

class StandartModelEntity:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_embeddings(self, text):
        try: 
            if not text: 
                raise ValueError("Input text cannot be empty.")
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.numpy()
        except Exception as e: 
            return e 
        
    def save_model(self, path):
        """Save the model and tokenizer to the specified path."""
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            return str(e)

    def load_model(self, path):
        """Load the model and tokenizer from the specified path."""
        try:
            self.model = AutoModel.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            return str(e)