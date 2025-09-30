from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Departments
departments = ["Operations", "Finance", "HR", "Engineering", "Safety"]

# Path to your fine-tuned model folder
MODEL_PATH = "./bert_model/muril_model"  # points directly to the folder with config, tokenizer, pytorch_model.bin

# Load tokenizer and model from the same folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def classify_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
    return departments[pred_id], probs[0].tolist()

if __name__ == "__main__":
    paragraph = input("Enter paragraph (English + Malayalam):\n")
    label, scores = classify_text(paragraph)
    print("\nPredicted Department:", label)
    print("Confidence Scores:", dict(zip(departments, scores)))
