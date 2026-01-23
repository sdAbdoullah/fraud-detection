# 🚀 Deployment Guide - Détection Proactive de Fraude Bancaire

**Guide complet pour déployer le système en production**

---

## 📋 Table des matières

1. [Local Development](#local-development)
2. [Environment Setup](#environment-setup)
3. [Model Training](#model-training)
4. [Testing](#testing)
5. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
6. [Docker Deployment](#docker-deployment)
7. [Production Checklist](#production-checklist)
8. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development

### Prerequisites

```bash
# System requirements
- Python 3.8 or higher
- pip or conda package manager
- 4GB RAM minimum
- 500MB disk space for models + data
```

### Setup Instructions

#### 1. Clone repository

```bash
git clone https://github.com/sdAbdoullah/fraud-detection.git
cd fraud-detection
```

#### 2. Create virtual environment

```bash
# Option A: venv (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Option B: conda
conda create -n fraud-detection python=3.9
conda activate fraud-detection
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download dataset (optional)

```bash
# Download from Kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/

# Or copy from existing location
cp /path/to/creditcard.csv data/
```

---

## Environment Setup

### .env Configuration

Create `.env` file in root directory:

```bash
# API Configuration
GEMINI_API_KEY=sk_your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=false
STREAMLIT_LOGGER_LEVEL=info
STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true

# Model Configuration
MODEL_PATH=models/xgboost_model.pkl
SCALER_PATH=models/standard_scaler.pkl
FRAUD_THRESHOLD=0.50

# Data Configuration
DATA_PATH=data/creditcard.csv
TEST_SIZE=0.3

# SMOTE Configuration
SMOTE_RATIO=1.0
RANDOM_STATE=42

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/fraud_detection.log
```

### Secure API Key

**NEVER commit .env to Git!**

```bash
# Add to .gitignore (already done)
echo ".env" >> .gitignore

# Use .env.example as template
cp .env.example .env
# Edit .env with your actual values
```

---

## Model Training

### Option 1: Full Training Pipeline

```bash
# Run Jupyter Notebook
jupyter notebook notebooks/fraud_detection_final.ipynb

# Or export to Python script
jupyter nbconvert --to script notebooks/fraud_detection_final.ipynb
python notebooks/fraud_detection_final.py
```

**Expected output**:
- `models/xgboost_model.pkl` (~5MB)
- `models/standard_scaler.pkl` (~1KB)
- `models/model_info.json` (metadata)
- Visualizations in `visualizations/` folder

**Training time**: 
- CPU: 3-5 minutes
- GPU: <1 minute

### Option 2: Using Pre-trained Model

```bash
# If model already trained, just load it
python -c "
import pickle
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model loaded successfully')
"
```

### Hyperparameters

Edit in notebook or via config:

```python
# SMOTE parameters
SMOTE_RATIO = 1.0
SMOTE_K_NEIGHBORS = 5

# XGBoost parameters
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.01
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# Prediction threshold
FRAUD_THRESHOLD = 0.50  # Adjust for recall vs precision trade-off
```

---

## Testing

### Unit Tests

```bash
# Run pytest
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Manual Testing

```bash
# Test model loading
python -c "
import pickle
from pathlib import Path

model_path = Path('models/xgboost_model.pkl')
scaler_path = Path('models/standard_scaler.pkl')

assert model_path.exists(), 'Model file not found'
assert scaler_path.exists(), 'Scaler file not found'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print('✅ Model loaded successfully')
print(f'Model type: {type(model)}')
"

# Test API connectivity
python -c "
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content('Test: say OK')
print(f'✅ API working: {response.text}')
"

# Test Streamlit
streamlit run app.py --logger.level=debug
```

---

## Streamlit Cloud Deployment

### Prerequisites

1. GitHub account with repository pushed
2. Streamlit Cloud account (https://streamlit.io/cloud)
3. API keys configured

### Step-by-Step Deployment

#### 1. Prepare repository

```bash
# Ensure all files committed
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main

# Check .gitignore includes:
# ├─ .env (sensitive data)
# ├─ models/ (large files - optional)
# ├─ data/ (large CSV)
# └─ __pycache__/
```

#### 2. Create Streamlit secrets

In GitHub repository settings:
```
Actions → Secrets and variables → Repository secrets
```

Add secret:
- Name: `GEMINI_API_KEY`
- Value: Your actual API key

Or in Streamlit Cloud dashboard:
```
App settings → Secrets
```

Add:
```
GEMINI_API_KEY = "sk_xxxxx..."
```

#### 3. Deploy on Streamlit Cloud

**Option A: Via Streamlit Cloud Dashboard**
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select repository: `sdAbdoullah/fraud-detection`
4. Branch: `main`
5. File path: `app/app.py`
6. Click "Deploy"

**Option B: Via CLI**
```bash
streamlit cloud deploy
```

### Post-Deployment

```bash
# App URL will be
https://sdAbdoullah-fraud-detection.streamlit.app

# Check logs
streamlit logs

# Monitor performance
# → Check Streamlit Cloud dashboard for metrics
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile` in root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  fraud-detection:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
```

### Build & Run

```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  fraud-detection:latest

# Or with docker-compose
docker-compose up -d

# Access at http://localhost:8501
```

---

## Production Checklist

### Before Going Live

- [ ] **Security**
  - [ ] `.env` NOT in Git
  - [ ] API keys in environment variables only
  - [ ] Database credentials secured
  - [ ] SSL/TLS enabled (HTTPS)

- [ ] **Code Quality**
  - [ ] All tests passing
  - [ ] Code linted (pylint, flake8)
  - [ ] Type hints added
  - [ ] Documentation complete

- [ ] **Performance**
  - [ ] Model loads in <1s
  - [ ] Prediction time <100ms per transaction
  - [ ] Memory usage <500MB
  - [ ] Cache configured (for Gemini responses)

- [ ] **Data**
  - [ ] Data pipeline tested
  - [ ] Missing data handling verified
  - [ ] Outlier handling confirmed
  - [ ] Preprocessing matches training

- [ ] **Model**
  - [ ] Model serialized correctly
  - [ ] Scaler serialized correctly
  - [ ] Metadata saved (version, metrics)
  - [ ] ROC-AUC stable (>0.96)
  - [ ] Recall maintained (>80%)

- [ ] **Deployment**
  - [ ] App tested locally
  - [ ] All dependencies in requirements.txt
  - [ ] Environment variables configured
  - [ ] Logs configured
  - [ ] Error handling working

- [ ] **Documentation**
  - [ ] README.md complete
  - [ ] ARCHITECTURE.md detailed
  - [ ] METHODOLOGY.md thorough
  - [ ] Code commented
  - [ ] API documented

- [ ] **Monitoring**
  - [ ] Logging configured
  - [ ] Alerts set up
  - [ ] Health check endpoint
  - [ ] Performance metrics tracked

---

## Monitoring & Maintenance

### Daily Monitoring

```python
# Log daily metrics
logger.info(f"Date: {datetime.now()}")
logger.info(f"Transactions processed: {n_transactions}")
logger.info(f"Alerts generated: {n_alerts}")
logger.info(f"Average fraud probability: {avg_prob:.4f}")
logger.info(f"Model performance: ROC-AUC {roc_auc:.4f}")
```

### Weekly Reports

```python
# Weekly summary
def generate_weekly_report():
    report = {
        'period': 'Last 7 days',
        'total_transactions': 35000,
        'total_alerts': 1750,
        'alert_rate': 5.0,
        'confirmed_frauds': 125,
        'false_positives': 1625,
        'roi': (125 * 300 - 1625 * 5) / (1625 * 5),
    }
    return report
```

### Monthly Retraining

```bash
# Schedule monthly retraining
# Add to crontab:
# 0 2 1 * * cd /path/to/app && python retrain.py

# Or use GitHub Actions:
# .github/workflows/retrain.yml
```

### Performance Degradation Alert

```python
# Alert if metrics drop
def check_performance():
    current_roc_auc = evaluate_model()
    
    if current_roc_auc < 0.94:  # Below expected
        logger.error(f"⚠️ ROC-AUC dropped to {current_roc_auc}")
        notify_team()
        trigger_retraining()
    
    current_recall = evaluate_recall()
    if current_recall < 0.75:  # Below 80% target
        logger.warning(f"⚠️ Recall dropped to {current_recall}")
        notify_team()
```

### Updating Model

```bash
# New model retraining
python notebooks/fraud_detection_final.ipynb

# Test new model
pytest tests/test_new_model.py

# If metrics better, replace
cp models/xgboost_model.pkl models/xgboost_model.pkl.backup
cp models/xgboost_model_new.pkl models/xgboost_model.pkl

# Restart service
systemctl restart fraud-detection-app

# Monitor
tail -f logs/fraud_detection.log
```

---

## Troubleshooting

### App won't start

```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip install -r requirements.txt --upgrade

# Check configuration
python -c "import os; print(os.getenv('GEMINI_API_KEY')[:10])"

# Run with verbose logging
streamlit run app/app.py --logger.level=debug
```

### Model not loading

```bash
# Check file exists
ls -la models/xgboost_model.pkl

# Check size (should be ~5MB)
du -h models/xgboost_model.pkl

# Try reloading
python -c "import pickle; pickle.load(open('models/xgboost_model.pkl', 'rb'))"
```

### Gemini API errors

```bash
# Check API key
echo $GEMINI_API_KEY

# Test connectivity
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
print('✅ API configured')
"

# Check rate limits
# Monitor in Google AI Studio dashboard
```

### Memory issues

```bash
# Check available RAM
free -h

# Profile memory usage
python -m memory_profiler app.py

# Optimize:
# - Cache predictions
# - Reduce batch size
# - Use generator for large datasets
```

---

## Support

For issues, contact:
- **GitHub Issues**: https://github.com/sdAbdoullah/fraud-detection/issues
- **Email**: Abdellahilimam181@gmail.com
- **Documentation**: See docs/ folder

---

**Ready to deploy?** Follow the checklist above and deploy with confidence! 🚀
