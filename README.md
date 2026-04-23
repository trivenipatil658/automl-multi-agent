# 🧬 AutoML Studio

An end-to-end **multi-agent AutoML pipeline** built with LangGraph, Streamlit, and Groq LLM. Upload a CSV, train multiple models, get visualizations, give feedback, and let the system improve itself — all in one UI.

---

## ✨ Features

- **Multi-Agent Pipeline** — Specialized agents for data analysis, feature engineering, model selection, hyperparameter tuning, evaluation, and critique
- **Automated Model Training** — Trains Logistic Regression, Random Forest, and SVM with GridSearchCV in parallel
- **Human Feedback Loop** — Rate results, write suggestions, and trigger LLM-guided retraining
- **LLM-Driven Improvement** — Uses Groq (LLaMA 3.3 70B) to interpret your feedback and pick the best retraining strategy
- **Visualizations** — Confusion Matrix and ROC Curve, regenerated after model improvement
- **PDF Report Export** — Download a full report after training or improvement
- **Clean Streamlit UI** — Step-by-step progress bar, metric cards, and responsive layout

---

## 🏗️ Architecture

```
automl-multi-agent/
├── agents/                  # LangGraph agent nodes
│   ├── data_analyst.py      # Analyses dataset structure
│   ├── feature_engineer.py  # Suggests feature transformations
│   ├── model_selector.py    # Recommends model types
│   ├── hyperparameter_tuner.py
│   ├── evaluator.py
│   └── critic.py            # Reviews overall pipeline quality
├── core/
│   ├── graph.py             # LangGraph state machine
│   └── ml_pipeline.py       # Training, evaluation, improvement logic
├── ui/
│   └── app.py               # Streamlit frontend
├── utils/
│   ├── feedback_store.py    # Persist & query human feedback
│   ├── llm.py               # Groq LLM client
│   └── report_generator.py  # PDF report generation
├── main.py
└── requirements.txt
```

---

## 🔄 Pipeline Flow

```
Upload CSV
    ↓
Dataset Preview  →  Configuration
    ↓
Train Models (LR · RF · SVM)
    ↓
Model Comparison + Metrics
    ↓
Visualizations (Confusion Matrix · ROC Curve)
    ↓
Human Feedback  →  LLM interprets suggestion
    ↓
Improved Model  →  Updated Visualizations
    ↓
Export PDF Report
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/automl-multi-agent.git
cd automl-multi-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run ui/app.py
```

---

## 🤖 Feedback-Driven Improvement

When you rate results as 👎 **Inaccurate**, you can type a suggestion like:

> *"Try gradient boosting"* or *"Improve recall"* or *"Use deeper trees"*

The system sends your suggestion to the LLM which picks one of these strategies:

| Strategy | Model Used |
|---|---|
| `boost_trees` | Random Forest (wider search) |
| `increase_depth` | Random Forest (deeper trees) |
| `try_gradient_boosting` | GradientBoostingClassifier |
| `increase_regularization` | Logistic Regression (high C penalty) |

Models with consistently bad feedback are **automatically skipped** in future runs.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| LLM | Groq · LLaMA 3.3 70B |
| ML | scikit-learn |
| Data | pandas · numpy |
| Visualization | matplotlib |
| Report | PDF generation |
| Config | python-dotenv |

---

## 📋 Requirements

- Python 3.9+
- Groq API key (free tier works)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
