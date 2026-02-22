# El Matador — News Credibility Analyzer

El Matador is an AI-powered news credibility analysis tool that helps users evaluate the trustworthiness of news articles. It detects misinformation patterns, analyzes emotional language, highlights suspicious claims, and produces an overall credibility score, all through an interactive Streamlit web interface.

---

## Features

- **Credibility Scoring** — Assigns a trustworthiness score to news articles using a trained ML model
- **Emotional Tone Analysis** — Detects emotionally charged or manipulative language that may signal bias
- **Claim Highlighting** — Identifies and highlights specific claims within articles that warrant scrutiny
- **Pattern Detection** — Flags common misinformation patterns and rhetorical techniques
- **Interactive Web UI** — Clean, browser-based interface built with Streamlit

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Web UI | Streamlit |
| ML Model | Scikit-learn (trained via `train_model.py`) |
| Frontend Assets | HTML, CSS, JavaScript (in `/templates` and `/static`) |
| Config | `.streamlit/config.toml` |

---

## Project Structure

```
El_Matador/
├── streamlit_app.py         # Main application entry point
├── credibility_analyzer.py  # Core credibility scoring logic
├── emotional_analyzer.py    # Emotional language detection
├── claim_highlighter.py     # Claim identification and highlighting
├── pattern_detector.py      # Misinformation pattern detection
├── train_model.py           # ML model training script
├── utils.py                 # Shared utility functions
├── models/                  # Saved ML model files
├── static/                  # CSS, JS, and other static assets
├── templates/               # HTML templates
├── .streamlit/              # Streamlit configuration
│   └── config.toml
├── .kiro/specs/             # Project specs (news-credibility-analysis)
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

##  Setup & Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/parthz-13/El_Matador.git
cd El_Matador
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (First-Time Setup)

If the `models/` directory is empty or no pre-trained model is present, run:

```bash
python train_model.py
```

This will generate and save the ML model used for credibility analysis.

### 5. Run the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. Open the app in your browser after running the Streamlit command above.
2. Paste the text of a news article into the input field.
3. Click **Analyze** to run the full credibility pipeline.
4. Review the results:
   - Overall **credibility score**
   - **Emotional tone** breakdown
   - **Highlighted claims** within the article
   - Detected **misinformation patterns**

---

