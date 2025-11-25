# TextMorph-Advanced-Text-Summarization-and-Paraphrasing

## 🚀 Overview
TextMorph is a full‑stack AI-powered web application that provides **smart text summarization, paraphrasing, evaluation, model comparison, dataset analysis, and admin-level analytics**.  
The system includes both **User Dashboard** and **Admin Dashboard**, with JWT authentication, history tracking, feedback system, and dataset evaluation using ROUGE, BERTScore, BLEU, and custom scoring.

---

## ✨ Key Features

### 🔹 User Features
- **AI Text Summarization** (Small / Medium / Large models)
- **AI Paraphrasing** (Simple / Standard / Creative modes)
- **Document Upload (PDF, DOCX, TXT, CSV)**
- **Model Evaluation with Metric Scores**
- **Dataset Evaluation** with automatic scoring
- **Summaries & Paraphrases History**
- **Feedback System** (Thumbs up/down + comments)
- **Profile Management**
- **ROUGE Score Calculation**
- **Readability & Complexity Analysis**

---

## 🔹 Admin Features
- **View & Edit All Summaries**
- **View & Edit All Paraphrases**
- **Moderate & Delete User Outputs**
- **Usage Statistics Dashboard**
  - User counts  
  - Total tasks  
  - Feedback distribution pie-chart  
  - Top models  
  - Top users  
  - Task breakdown (summaries vs paraphrases)
- **Advanced Feedback Analytics**
- **Search filtering for summaries/paraphrases**

---

## 🧠 Machine Learning Models
TextMorph supports multi-model operations:

### **Summarization Models**
- **T5‑Small / Base / Large**
- **BART**
- **PEGASUS**
- Custom fine‑tuned summarization models

### **Paraphrasing Models**
- T5 paraphrase models  
- Custom multi‑task T5 models (via `train_multi_task.py`)

### **Readability & Scoring**
- ROUGE (ROUGE‑1, ROUGE‑2, ROUGE‑L)
- BLEU
- Readability metrics via `textstat`
- Evaluation pipeline via `evaluate_models.py`

---

## 📂 Project Structure

```
Frontend/
 ├── app.py
 ├── user.py
 ├── admin.py
Backend/
 ├── main.py
 ├── account.py
 ├── summarization_routes.py
 ├── paraphrase_routes.py
 ├── translation_routes.py
 ├── feedback_routes.py
 ├── dataset.py
 ├── history.py
 ├── admin_view.py
 ├── admin_view_usage.py
 ├── preprocess.py
 ├── preprocess_all.py
 ├── evaluate_models.py
 ├── train_multi_task.py
 ├── database.py
 ├── otp_utils.py
schema.sql
model_registry.json
requirements.txt
README.md
```

---

## 🛢️ Database Schema Highlights

### Users, History, Admins, Feedback, Rouge scores, etc.

✔ Users  
✔ OTP verification  
✔ Summaries  
✔ Paraphrases  
✔ ROUGE scores  
✔ Feedback  
✔ Admins  
✔ Usage stats  
✔ File uploads  

(Full SQL in `schema.sql`)

---

## ⚙️ Installation

### **1️⃣ Clone the Repository**
```
git clone https://github.com/ajithreddych/TextMorph-Advanced-Text-Summarization-and-Paraphrasing.git
cd TextMorph
```

### **2️⃣ Create Virtual Environment**
```
python3 -m venv venv
source venv/bin/activate
```

### **3️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### **Start Backend**
```
cd Backend
python main.py
```

### **Start Frontend**
```
cd Frontend
streamlit run app.py
```

---

## 📊 Dataset Evaluation

Upload a CSV → Select model → Click Evaluate →  
System automatically generates metrics:

- ROUGE‑1 / ROUGE‑2 / ROUGE‑L  
- Readability  
- Compression ratio  
- Semantic similarity  
- Model comparison charts  

---

## 🔐 Authentication System
- JWT tokens  
- Login / Signup with OTP verification  
- Forgot password with OTP  
- Admin verification via JWT  

---

## 🧮 Feedback Pipeline
- User can like/dislike each summary/paraphrase  
- Add comments  
- Stored in DB  
- Admin sees statistics in interactive charts  

---

## 📈 Admin Usage Dashboard Includes:
- Pie chart of thumbs up vs thumbs down  
- User feedback table  
- Most active users  
- Most used models  
- Top tasks  
- Raw downloadable CSV  

---

## 🛠️ Tech Stack

### **Frontend**
- Streamlit  
- Plotly  
- Pandas  

### **Backend**
- Flask  
- Flask-CORS  
- Flask-JWT-Extended  
- MySQL  

### **ML/NLP**
- Transformers  
- Torch  
- ROUGE  
- BLEU  
- NLTK  
- Textstat  

---

## 📄 Requirements

(Already included in your provided `requirements.txt`)  
All packages from frontend, backend, preprocessing, ML, evaluation, admin dashboards, and dataset modules are accounted for.

---

## 🤝 Contribution Guidelines
- Fork repo  
- Create a branch  
- Commit changes  
- Submit pull request  

---

## 📧 Contact
For any issues or collaboration ideas:

**Ajith Reddy Ch**  
📩 ajithreddychittireddy@gmail.com  

---

## ⭐ If you found this project useful, please star the repository!  
