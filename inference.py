import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
import torch.nn.functional as F

# ---------------- Paths ----------------
MODEL_DIR = "bert_model/saved_model"  # DistilBERT saved model path

# ---------------- Load Tokenizer ----------------
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)

# ---------------- Load Model ----------------
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ---------------- Load Label Encoder ----------------
with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- Define Keywords per Department ----------------
DEPT_KEYWORDS = {
    "HR": [
        "leave", "recruitment", "policy", "training", "onboarding", "diversity", "employee",
        "payroll", "performance", "appraisal", "benefits", "attendance", "retention",
        "hiring", "staffing", "career", "promotion", "grievance", "resignation", "termination",
        "wellbeing", "orientation", "HRIS", "workforce", "talent"
    ],
    "Finance": [
        "budget", "expenses", "revenue", "income", "depreciation", "account", "finance", "cost",
        "profit", "loss", "balance sheet", "cash flow", "tax", "audit", "investment",
        "funding", "financial", "ledger", "capital", "billing", "invoicing", "accounting",
        "expenditure", "fiscal", "forecast", "capitalization", "ROI", "return on investment",
        "EBITDA", "gross margin", "net margin", "operating income", "EPS", "percentage", "ratio",
        "figures", "valuation", "assets", "liabilities", "equity", "interest", "dividends", "costs",
        "revenue stream", "financial statement", "numbers", "amount", "Rs.", "$", "₹"
    ],
    "Operations": [
        "maintenance", "inspection", "schedule", "operational", "assets",
        "logistics", "fleet", "transport", "supply chain", "procedure",
        "workflow", "efficiency", "monitoring", "standard operating", "downtime", "shift",
        "operations", "quality control", "process", "production", "tracking", "inventory"
    ],
    "Engineering": [
        "technical", "track", "rolling stock", "infrastructure", "engineering", "equipment",
        "design", "project", "construction", "systems", "mechanical", "electrical", "civil",
        "automation", "prototype", "installation", "testing", "commissioning", "specifications",
        "blueprint", "drawings", "repair", "upgrade", "modification", "fabrication"
    ],
    "Safety & Compliance": [
        "safety", "safety leadership", "risk assessment", "incident", "hazard",
        "emergency", "accident", "employee safety", "safety audit", "safety training",
        "compliance", "regulation", "policy compliance", "legal compliance",
        "standard", "audit report", "procedure compliance", "safety compliance",
        "environmental compliance", "regulatory", "statutory", "internal audit"
    ]
}

# ---------------- Inference Function ----------------
def predict_department_with_keywords(text: str):
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze()
    
    # Top 2 model predictions
    top2_probs, top2_idx = torch.topk(probs, k=2)
    top2_labels = [label_encoder.inverse_transform([i])[0] for i in top2_idx.tolist()]
    
    # Keyword matching
    text_lower = text.lower()
    keyword_scores = {}
    for dept, keywords in DEPT_KEYWORDS.items():
        keyword_scores[dept] = sum(1 for kw in keywords if kw.lower() in text_lower)
    
    # Choose department based on keyword count if >0, else top model prediction
    keyword_max_dept = max(keyword_scores, key=lambda k: keyword_scores[k])
    if keyword_scores[keyword_max_dept] > 0:
        final_dept = keyword_max_dept
    else:
        final_dept = top2_labels[0]  # fallback to top model prediction
    
    return {
        "top2_model_preds": list(zip(top2_labels, top2_probs.tolist())),
        "final_department": final_dept
    }

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Enter text to classify (type 'exit' to quit):")
    while True:
        text = input("\n> ")
        if text.lower() in ["exit", "quit"]:
            break
        
        result = predict_department_with_keywords(text)
        print("\nTop 2 Model Predictions:")
        for label, prob in result["top2_model_preds"]:
            print(f"{label}: {prob:.4f}")
        print(f"Final Department (with keyword check): {result['final_department']}")
