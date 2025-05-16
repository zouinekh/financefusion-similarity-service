# 🧠 Semantic Similarity & NLI API Documentation

## ✅ Purpose

This API evaluates the **relationship between two pieces of text** by:

1. Measuring **semantic similarity** using embeddings and cosine similarity.
2. Classifying the **logical relationship** (entailment, contradiction, or neutrality) using a Natural Language Inference (NLI) model.

It is used in the **Finance Fusion** app to determine whether a user's transaction contradicts their financial goals or challenges (e.g., spending at McDonald's vs. a goal to "Avoid fast food").

---

## 🔍 Use Case

In the Finance Fusion app:

1. A user logs a transaction like: `"KFC Bucket Meal"`.
2. The system checks the user's active challenge: `"No fast food this month"`.
3. This service:
   - Compares both texts using embeddings (MiniLM).
   - Uses an NLI model (RoBERTa) to determine if one contradicts the other.
4. If **similarity > 0.75** and **label = contradiction**, a challenge violation is detected.

---

## 🧪 Outputs

- **similarity**: A float (0–1) indicating how similar the two texts are.
- **label**: One of:
  - `"entailment"`: One text logically implies the other.
  - `"neutral"`: No clear logical relationship.
  - `"contradiction"`: One text contradicts the other.
- **confidence**: Confidence score (0–1) for the predicted label.

---

## 🧱 Tech Stack

| Component        | Tool/Library                              |
|------------------|--------------------------------------------|  
| Language         | Python 3                                   |
| Framework        | FastAPI                                    |
| Similarity Model | `all-MiniLM-L6-v2` (SentenceTransformers)  |
| NLI Model        | `roberta-large-mnli` (Facebook AI)         |
| API Server       | Uvicorn                                    |
| Embedding Metric | Cosine Similarity                          |

---

## 📁 Project Structure
```bash
embedding_service/
├── main.py # FastAPI app entry point
├── app/
│ ├── api.py # Defines routes and endpoints
│ ├── models.py # Pydantic models for request and response
│ ├── services.py #  NLI logic
│ ├── embeddingService.py # Embedding  logic
│ └── config.py # Reserved for future config needs
└── requirements.txt # Python dependencies
```

---

## 📮 API Reference

### POST `/api/similarity`

**Request Body:**

```json
{
  "text1": "Avoid fast food",
  "text2": "McDonald's Big Mac"
}
```
Response:


```json
{
  "similarity": 0.9064,
  "label": "contradiction",
  "confidence": 0.9868
}
```
📊 Interpreting Results
Similarity 	Meaning
0.80 – 1.00	  ===> Highly similar (likely related)
0.60 – 0.79	 ===> Possibly related
0.40 – 0.59	 ===> Loosely related
< 0.40	 ===> Not related

Label	Meaning
entailment	 ===> The second text logically follows from the first
neutral	 ===> No clear logical connection
contradiction	 ===> The second text conflicts with the first (e.g., violates a challenge)

⚙️ How to Run Locally
1. Activate the virtual environment:
```bash
source embedding-env/bin/activate
# 2. Start the FastAPI server
uvicorn main:app --reload
```
🔗 Useful Model Links
all-MiniLM-L6-v2 (Embeddings) : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

all-mpnet-base-v2 (Alternative Embeddings) https://huggingface.co/sentence-transformers/all-mpnet-base-v2

roberta-large-mnli (NLI Model) https://huggingface.co/FacebookAI/roberta-large-mnli