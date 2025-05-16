from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NLIService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        self.labels = ["contradiction","entailment", "neutral" ]

    def check_contradiction(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
        label = self.labels[predicted_class]
        confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
        return {
            "label": label,
            "confidence": confidence
        }
