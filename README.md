# Toxic Comment Detection using BERT 🚫

This project aims to detect toxic comments using a fine-tuned BERT model. It leverages deep learning and transformer-based NLP techniques to classify user-generated comments as toxic or non-toxic, helping moderate harmful content online.

---

## 📁 Dataset

The dataset used is in JSON format (`z639_assignment1_training.json`), consisting of multiple votes per comment determining its toxicity. A majority voting mechanism is applied to assign the final label (`is_toxic`).

---

## 🧪 Project Workflow

### 1. Data Preprocessing
- Loaded and converted JSON data to a DataFrame.
- Applied "majority voting" to assign toxicity labels (`True` for toxic, `False` otherwise).

### 2. Tokenization
- Used HuggingFace's `BertTokenizer` to tokenize the comment texts.
- Converted data into PyTorch-compatible datasets.

### 3. Model Setup
- Loaded `BertForSequenceClassification` with binary classification head.
- Defined a custom PyTorch dataset and dataloaders for training and evaluation.

### 4. Training Configuration
- Used HuggingFace’s `Trainer` and `TrainingArguments`.
- Model trained with:
  - Adam optimizer
  - Batch size = 16
  - Epochs = 3 (example)
  - Evaluation strategy: steps
  
### 5. Model Evaluation
- Accuracy: ~75%
- Confusion Matrix:
  - True Negatives: 440
  - True Positives: 159
  - False Positives: 150
  - False Negatives: 51
- **Insights:**
  - High precision for non-toxic comments (few false alarms).
  - High recall for toxic comments (most toxic content detected).

---

## 🛠️ Tech Stack

- Python 
- PyTorch
- HuggingFace Transformers (`BERT`)
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## 📌 Key Highlights

- Custom preprocessing using majority voting strategy
- PyTorch DataLoader integration
- BERT fine-tuning for binary classification
- Evaluation using confusion matrix and accuracy

---

## 📄 License

This project is for academic and educational purposes only.

---

## 🙋‍♀️ Author

**Vishwajyothi Reshmi**  
Master's Student | Data Science  
