# 🏗️ Architecture - Détection Proactive de Fraude Bancaire

**Documentation technique complète du système**

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture système](#architecture-système)
3. [Pipeline ML](#pipeline-ml)
4. [Application Streamlit](#application-streamlit)
5. [Intégration IA Générative](#intégration-ia-générative)
6. [Stack technologique](#stack-technologique)
7. [Flux de données](#flux-de-données)
8. [Performance & Scalabilité](#performance--scalabilité)
9. [Sécurité](#sécurité)

---

## Vue d'ensemble

### Objectif architectural
Créer une **solution modulaire et déployable** combinant :
- 🤖 Pipeline ML robuste (entraînement + prédiction)
- 🧠 IA générative pour explainability (Gemini API)
- 📊 Application web interactive (Streamlit)
- 🔐 Gestion sécurisée des données et API keys

### Principes de design
- ✅ **Séparation des préoccupations** - ML, UI, API distinctes
- ✅ **Modularité** - Composants indépendants et réutilisables
- ✅ **Scalabilité** - Architecture prête pour volume production
- ✅ **Robustesse** - Gestion d'erreurs, fallbacks, logging
- ✅ **Maintenabilité** - Code documenté, tests, CI/CD ready

---

## Architecture système

### Diagram macro

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTÈME COMPLET                             │
└─────────────────────────────────────────────────────────────────────┘

    LAYER 1: DATA & TRAINING
    ───────────────────────────────────────────────────────────────
    ┌─────────────────────────────────────────────────────────────┐
    │  Jupyter Notebook (fraud_detection_final.ipynb)            │
    │  • Load data (creditcard.csv)                              │
    │  • EDA & Feature Analysis                                  │
    │  • Preprocessing (StandardScaler)                          │
    │  • SMOTE (rééquilibre)                                     │
    │  • Train 3 models (RF, XGB, LGBM)                         │
    │  • Evaluation & Selection                                  │
    │  • Save artifacts (model.pkl, scaler.pkl)                │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    LAYER 2: MODELS & INFERENCE
    ───────────────────────────────────────────────────────────────
    ┌─────────────────────────────────────────────────────────────┐
    │  XGBoost Model (production-ready)                           │
    │  • Saved as: models/xgboost_model.pkl                     │
    │  • Input: 30 features (V1-V28, Time, Amount)             │
    │  • Output: Fraud probability [0-1]                        │
    │  • ROC-AUC: 0.9725                                        │
    │  • Rappel: 84.46%                                         │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    LAYER 3: APPLICATION & UI
    ───────────────────────────────────────────────────────────────
    ┌─────────────────────────────────────────────────────────────┐
    │  Streamlit App (app.py)                                    │
    │  ┌─────────┬─────────┬─────────┬─────────┬────────┬──────┐│
    │  │Dashboard│ Alertes │Analyse  │Synthèse │Scénarios│Exports││
    │  │Onglet 1 │Onglet 2 │Onglet 3 │Onglet 4 │Onglet 5│Onglet6││
    │  └─────────┴─────────┴─────────┴─────────┴────────┴──────┘│
    │  • Load model + scaler                                     │
    │  • Display real-time metrics                               │
    │  • Filter & analyze transactions                           │
    │  • Call Gemini API for explanations                       │
    │  • Export to Excel                                         │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    LAYER 4: AI EXPLANATION
    ───────────────────────────────────────────────────────────────
    ┌─────────────────────────────────────────────────────────────┐
    │  Google Gemini 2.5 Flash API                              │
    │  • Receive transaction data + ML score                     │
    │  • Generate textual explanations                           │
    │  • Produce recommendations (BLOCK/VERIFY/MONITOR)         │
    │  • Create synthetic fraud scenarios                        │
    │  • Generate daily risk summaries                           │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    OUTPUT: ANALYST DECISION
    ───────────────────────────────────────────────────────────────
    • Dashboard view with alerts
    • Gemini explanations for each transaction
    • Excel exports for reporting
    • Recommendations for actions
```

---

## Pipeline ML

### Phase 1: Data Loading & EDA

```python
# Input: creditcard.csv (284,807 rows × 31 columns)
#
# Variables:
# ├─ V1-V28: PCA components (anonymized)
# ├─ Time: Seconds since first transaction
# ├─ Amount: Transaction amount (USD)
# └─ Class: Target (0=legitimate, 1=fraud)

# Output: Understanding of data distribution
# ├─ Class balance: 99.83% vs 0.17%
# ├─ Amount statistics: mean 88.35, median varies
# ├─ Correlations: V11, V4, V2 → strongest with fraud
# └─ Missing values: None (perfect data)
```

### Phase 2: Preprocessing

```python
# StandardScaler normalization
# ├─ Input: Amount [0, 25691.16], Time [0, 172800]
# └─ Output: Amount, Time ~ N(0, 1)
#
# Train/Test split (stratified)
# ├─ Train: 199,364 (70%) - maintain 0.17% fraud ratio
# └─ Test: 85,443 (30%) - maintain 0.17% fraud ratio
#
# SMOTE (on training set only)
# ├─ Before: 199,020 legitimate vs 344 fraud
# ├─ After: 199,020 legitimate vs 199,020 synthetic fraud
# └─ k_neighbors=5, sampling_strategy=1.0
```

### Phase 3: Model Training

```python
# Random Forest
# ├─ n_estimators=100
# ├─ max_depth=15
# ├─ ROC-AUC: 0.9690
# └─ Rappel: 78.38%

# XGBoost ⭐ (SELECTED)
# ├─ n_estimators=200
# ├─ max_depth=5
# ├─ learning_rate=0.01
# ├─ ROC-AUC: 0.9725 (BEST)
# └─ Rappel: 84.46% (BEST)

# LightGBM
# ├─ n_estimators=150
# ├─ max_depth=7
# ├─ ROC-AUC: 0.9636
# └─ Rappel: 83.11%
```

### Phase 4: Evaluation

```python
# Metrics computed on test set (85,443 transactions):
#
# Confusion Matrix
# ├─ TP: 125 (fraudes correctement détectées)
# ├─ FP: 28,000 (fausses alertes)
# ├─ FN: 23 (fraudes manquées)
# └─ TN: 57,295 (légitimes correctement classées)
#
# Derived metrics
# ├─ Accuracy: 99.63% (global correctness)
# ├─ Precision: 29.98% (TP / (TP+FP))
# ├─ Rappel: 84.46% (TP / (TP+FN)) ← KEY METRIC
# ├─ F1-score: 0.4467
# └─ ROC-AUC: 0.9725 (area under ROC curve)
#
# Feature Importance
# ├─ V11: 25%
# ├─ V4: 18%
# ├─ V2: 15%
# ├─ V14: 12%
# └─ V12: 10%
```

### Phase 5: Model Serialization

```python
# Save to disk for production
pickle.dump(xgb_model, open('models/xgboost_model.pkl', 'wb'))
pickle.dump(scaler, open('models/standard_scaler.pkl', 'wb'))
json.dump(model_metadata, open('models/model_info.json', 'w'))
```

---

## Application Streamlit

### Architecture modulaire

```
app/
├── app.py                    # Main entry point
├── config.py                 # Configuration centralized
├── utils.py                  # Utility functions
└── components/
    ├── dashboard.py          # Tab 1: KPIs & overview
    ├── alerts.py             # Tab 2: Real-time alerts
    ├── analysis.py           # Tab 3: Detailed analysis
    ├── gemini_integration.py # Tab 4: AI summaries
    ├── scenarios.py          # Tab 5: Synthetic scenarios
    └── exports.py            # Tab 6: Excel exports
```

### Tab Architecture

```
TAB 1: DASHBOARD
├─ Metrics
│  ├─ Total transactions
│  ├─ Number of alerts
│  ├─ Total amount
│  └─ At-risk amount
└─ Visualizations
   ├─ Amount distribution (histogram)
   ├─ Fraud probability distribution
   ├─ Alerts by city
   └─ Activity by hour

TAB 2: REAL-TIME ALERTS
├─ Filters
│  ├─ Risk level (CRITICAL/HIGH/MEDIUM)
│  ├─ City
│  └─ Merchant type
└─ Results table
   ├─ Transaction ID
   ├─ Amount
   ├─ Time
   ├─ Fraud probability
   ├─ [Gemini Analysis] button
   └─ [Details] button

TAB 3: DETAILED ANALYSIS
├─ Amount boxplots (all vs alerts)
├─ Amount by merchant type
└─ City × Risk heatmap

TAB 4: AI SYNTHESIS
├─ [Generate Global Summary] button
└─ Gemini response
   ├─ Risk overview
   ├─ Key patterns
   ├─ Operational recommendations
   └─ Strategic insights

TAB 5: SYNTHETIC SCENARIOS
├─ Slider: number of scenarios (1-10)
├─ [Generate] button
└─ Results
   ├─ Scenario 1 (high amount)
   ├─ Scenario 2 (sequence)
   └─ [Download CSV] button

TAB 6: EXPORTS
├─ Alert table
└─ [Export to Excel] button
```

### State Management

```python
# Session state for interactivity
st.session_state.model          # Cached XGBoost model
st.session_state.scaler         # Cached StandardScaler
st.session_state.transactions   # Loaded transactions
st.session_state.predictions    # ML predictions
st.session_state.alerts         # Filtered alerts
st.session_state.gemini_cache   # Cached Gemini responses
```

---

## Intégration IA Générative

### Pipeline Gemini

```
Transaction Data
    ↓
Prompt Engineering
    ├─ System message: "You are a fraud detection expert"
    ├─ User message: "Analyze this transaction..."
    ├─ Constraints: 2-3 sentences analysis
    ├─ Format: Recommendation + Signals
    └─ Temperature: 0.7 (balanced creativity)
    ↓
API Call
    └─ google.generativeai.GenerativeModel('gemini-2.5-flash')
    ↓
Response Processing
    ├─ Extract recommendation (BLOCK/VERIFY/MONITOR)
    ├─ Extract signals (key indicators)
    ├─ Format for display
    └─ Cache for identical inputs
    ↓
Display to Analyst
    └─ Rich formatted text with recommendations
```

### Example Prompt

```python
system_prompt = """You are an expert in bank fraud detection. 
Your role is to provide clear, actionable explanations for flagged transactions.
Analyze the data and provide:
1. Brief analysis (2-3 sentences)
2. Recommendation (BLOCK/VERIFY/MONITOR)
3. Key signals that triggered the alert
Keep explanations concise and operationally focused."""

user_prompt = f"""Analyze this transaction:
- ID: {tx_id}
- Amount: ${amount}
- Hour: {hour}
- City: {city}
- Merchant: {merchant_type}
- ML Score: {fraud_prob:.1%}

Provide analysis in French."""
```

### Error Handling

```python
try:
    response = gemini_model.generate_content(prompt)
    return response.text
except APIError as e:
    logger.error(f"Gemini API error: {e}")
    return fallback_explanation(transaction_data)
except RateLimitError as e:
    logger.warning(f"Rate limit: {e}")
    return cached_response if available else fallback
```

---

## Stack technologique

### Backend

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Language |
| **Pandas** | 1.5+ | Data manipulation |
| **Scikit-learn** | 1.0+ | Preprocessing (StandardScaler, SMOTE) |
| **XGBoost** | 1.5+ | ML model |
| **Imbalanced-learn** | 0.9+ | SMOTE implementation |
| **Joblib** | 1.0+ | Model serialization |

### Frontend

| Component | Version | Purpose |
|-----------|---------|---------|
| **Streamlit** | 1.20+ | Web application |
| **Plotly** | 5.0+ | Interactive visualizations |
| **Pandas** | 1.5+ | Data display |
| **Openpyxl** | 3.7+ | Excel export |

### AI & APIs

| Component | Version | Purpose |
|-----------|---------|---------|
| **Google Generative AI** | Latest | Gemini API client |
| **Python-dotenv** | 0.20+ | Environment variables |

### DevOps

| Component | Purpose |
|-----------|---------|
| **Git** | Version control |
| **GitHub Actions** | CI/CD pipeline |
| **Docker** | Containerization (optional) |
| **pytest** | Unit testing |

### Complete Stack

```
┌─────────────────────────────────────────────────────┐
│          TECHNOLOGY STACK                          │
├─────────────────────────────────────────────────────┤
│                                                    │
│  Frontend       Data         ML         AI         │
│  ─────────────────────────────────────────────     │
│  Streamlit      Pandas       XGBoost    Gemini    │
│  Plotly         NumPy        SMOTE      API       │
│  Openpyxl       Scikit-learn  RF        LLM       │
│                 Joblib       LGBM                 │
│                                                    │
│  Infrastructure:                                  │
│  • Python 3.8+                                    │
│  • pip/conda                                      │
│  • Git                                            │
│  • GitHub Actions (CI/CD)                         │
│                                                    │
└─────────────────────────────────────────────────────┘
```

---

## Flux de données

### 1. Training Flow

```
Raw Data (CSV)
    ↓ [Load]
Pandas DataFrame (284,807 rows)
    ↓ [Explore]
Statistical Analysis
    ↓ [Clean]
Processed Data
    ↓ [Normalize]
StandardScaler applied
    ↓ [Split]
Train (70%) | Test (30%)
    ↓ [SMOTE on Train]
Balanced Train Set
    ↓ [Train 3 Models]
Random Forest | XGBoost | LightGBM
    ↓ [Evaluate on Test]
Metrics: ROC-AUC, Recall, Precision, F1
    ↓ [Select Best]
XGBoost (ROC-AUC 0.9725)
    ↓ [Serialize]
model.pkl | scaler.pkl | metadata.json
```

### 2. Prediction Flow

```
New Transaction
    ↓ [Load Model + Scaler]
XGBoost Model (from pickle)
    ↓ [Normalize Features]
StandardScaler.transform()
    ↓ [Predict]
probability = model.predict_proba()
    ↓ [Threshold Check]
if proba > 0.50 → Alert
    ↓ [Call Gemini]
generate_content(transaction + score)
    ↓ [Format Response]
explanation + recommendation
    ↓ [Display in Streamlit]
Analyst sees alert with explanation
```

### 3. Application Flow

```
User opens Streamlit app
    ↓
Load cached model & scaler
    ↓
Load transaction data
    ↓
Generate predictions for all transactions
    ↓
Filter alerts (proba > 0.50)
    ↓
Display dashboard with metrics
    ↓
User selects tab
    ├─ Dashboard: Show KPIs
    ├─ Alerts: Show filtered alerts
    ├─ Analysis: Show detailed plots
    ├─ Synthesis: Call Gemini for summary
    ├─ Scenarios: Generate synthetic cases
    └─ Exports: Download Excel
```

---

## Performance & Scalabilité

### Benchmarks (current version)

| Operation | Time | Resources |
|-----------|------|-----------|
| Load model + scaler | ~100ms | 50MB RAM |
| Predict 5,000 transactions | ~200ms | 100MB RAM |
| Generate Gemini explanation | 1-2s | API call |
| Render dashboard | ~500ms | CPU + RAM |
| Export 1,000 rows to Excel | ~1s | Disk I/O |

### Scalability considerations

**Current bottleneck**: Gemini API latency (1-2s per call)

**Solutions for scaling**:
1. **Batch processing**: Queue Gemini requests, process asynchronously
2. **Caching**: Store Gemini responses for identical transactions
3. **Load balancing**: Distribute API calls across multiple API keys
4. **Background jobs**: Use Celery for async Gemini calls

**Future improvements**:
```python
# V2 Architecture with async processing
from celery import Celery

@app.task
def generate_gemini_explanation_async(transaction_id, data):
    # Call Gemini in background
    explanation = gemini_model.generate_content(prompt)
    # Store in database
    db.save(transaction_id, explanation)
    # Notify UI
    return explanation
```

---

## Sécurité

### API Key Management

```python
# ✅ CORRECT: Use environment variables
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# ❌ INCORRECT: Hardcode keys
api_key = "sk_xxxxxxxxxxxxx"  # NEVER!
```

### Secrets in .gitignore

```
.env                    # Environment variables
.env.local              # Local overrides
secrets.json            # Secret configs
credentials.json        # API credentials
*.pem                   # Private keys
```

### Data Privacy

```python
# Before storing/displaying transactions:
def anonymize_transaction(tx):
    tx['city'] = hashlib.md5(tx['city'].encode()).hexdigest()[:8]
    # Don't store raw transaction details
    return tx
```

### Rate Limiting for Gemini

```python
from functools import lru_cache
import time

@lru_cache(maxsize=1000)
def get_gemini_explanation(tx_hash):
    # Cache identical transactions
    # Avoid duplicate API calls
    return gemini_response

# Implement rate limiting
max_requests_per_minute = 60
```

---

## Deployment Architecture

### Local Development

```
Machine
├─ Python venv
├─ Models (pickle files)
├─ Streamlit app
└─ Gemini API (cloud)
```

### Production (Cloud - Optional)

```
┌──────────────────────────────────────┐
│        PRODUCTION DEPLOYMENT         │
├──────────────────────────────────────┤
│                                      │
│  Load Balancer (CloudFlare)         │
│         ↓                            │
│  Web Server (Streamlit Cloud)       │
│  ├─ app.py                          │
│  ├─ models/ (S3 storage)            │
│  └─ requirements.txt                │
│         ↓                            │
│  External APIs                      │
│  ├─ Google Gemini API               │
│  ├─ Database (PostgreSQL)           │
│  └─ Storage (S3/GCS)                │
│                                      │
└──────────────────────────────────────┘
```

---

## Monitoring & Logging

### Key Metrics to Monitor

```python
import logging

logger = logging.getLogger(__name__)

# Model performance
logger.info(f"Model accuracy: {accuracy:.4f}")
logger.info(f"Model ROC-AUC: {roc_auc:.4f}")

# API calls
logger.info(f"Gemini API calls: {gemini_calls}")
logger.warning(f"API errors: {api_errors}")

# Application performance
logger.info(f"Prediction time: {pred_time}ms")
logger.info(f"Memory usage: {memory}MB")
```

---

## Conclusion

Cette architecture est **modulaire, scalable et production-ready**. Les trois composants principaux (ML, App, AI) sont indépendants mais intégrés de façon cohérente.

**Prochaines étapes** :
1. Implémenter async Gemini calls
2. Ajouter base de données pour logging
3. Déployer sur Streamlit Cloud
4. Configurer CI/CD avec GitHub Actions
