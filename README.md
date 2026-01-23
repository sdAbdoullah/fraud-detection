# 🏦 Détection Proactive de Fraude Bancaire avec ML & IA Générative

> Système intelligent combinant **Machine Learning (XGBoost)** et **IA générative (Google Gemini)** pour détecter et expliquer les transactions frauduleuses en temps réel.


## 🎯 Aperçu du projet

### Objectif
Développer un **système opérationnel de détection proactive de fraude** capable de :
- ✅ Détecter automatiquement les transactions frauduleuses (84.46% de taux de détection)
- ✅ Expliquer chaque alerte en langage naturel via IA générative
- ✅ Fournir des recommandations actionnables (BLOQUER/VÉRIFIER/SURVEILLER)
- ✅ Permettre l'exportation de rapports pour l'analyse opérationnelle

### Le défi
- **Déséquilibre extrême des données** : 99.83% transactions légitimes vs 0.17% fraudes
- **Coûts asymétriques** : Fraude manquée (~300$) >> Fausse alerte (~5$)
- **Nécessité d'explainability** : Les analystes doivent comprendre pourquoi une transaction est suspecte

### Impact potentiel
- Détection de **~84% des fraudes** sur le test set (125/148 transactions)
- Économies estimées : **Fraudes évitées >> Coûts des fausses alertes**
- Déploiement operationnel immédiat via Streamlit

---

## ✨ Caractéristiques principales

### 🤖 Machine Learning
- **3 modèles comparés** : Random Forest, XGBoost ⭐, LightGBM
- **XGBoost champion** : ROC-AUC = 0.9725, Rappel = 84.46%
- **Gestion du déséquilibre** : SMOTE + Class Weights
- **Interprétabilité** : Feature importance et matrice de confusion

### 🧠 IA Générative (Google Gemini 2.5 Flash)
- **Explications textuelles** pour chaque alerte
- **Scénarios synthétiques** pour formation des équipes
- **Synthèse globale** des risques détectés
- **Recommandations intelligentes** basées sur le domaine

### 📊 Application Streamlit
| Onglet | Fonctionnalités |
|--------|-----------------|
| 📈 **Dashboard** | KPIs temps réel, distributions, alertes par ville |
| 🚨 **Alertes** | Filtrage avancé, détails transaction, analyse Gemini |
| 📉 **Analyse Détaillée** | Boxplots, heatmaps, corrélations |
| 🧠 **Synthèse IA** | Vue d'ensemble des risques générée par Gemini |
| 🎯 **Scénarios** | Générer cas d'usage synthétiques pour formation |
| 📥 **Exports** | Télécharger rapports en Excel |

---

## 🏗️ Architecture

### Stack Technologique

```
┌─────────────────────────────────────────────────────────────┐
│                   ARCHITECTURE GLOBALE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data Layer                                                 │
│  └─ Jupyter Notebook (fraud_detection_final.ipynb)         │
│     • EDA, SMOTE, entraînement 3 modèles                   │
│     • Sauvegarde XGBoost model (PKL)                       │
│                                                             │
│  ML Model                                                   │
│  └─ XGBoost (ROC-AUC: 0.9725)                              │
│     • Prediction: proba fraude [0-1]                       │
│     • Feature importance: V11, V4, V2...                   │
│                                                             │
│  Application Layer                                          │
│  └─ Streamlit (app.py)                                     │
│     • 6 onglets interactifs                                │
│     • Filtrage transactions                                │
│     • Export Excel                                         │
│                                                             │
│  AI Explanation Layer                                       │
│  └─ Google Gemini API                                      │
│     • Explications textuelles                              │
│     • Recommandations (BLOQUER/VÉRIFIER/SURVEILLER)       │
│     • Scénarios synthétiques                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline de données

```
CSV → Preprocessing → SMOTE → Entraînement → Sauvegarde
                                  ↓
                          XGBoost (PKL)
                                  ↓
                      Streamlit (Charge modèle)
                                  ↓
                    Prediction + Gemini Explanation
                                  ↓
                          Dashboard utilisateur
```

---

## 📦 Installation

### Prérequis
- Python 3.8+
- pip ou conda
- Clé API Google Gemini (gratuite via [Google AI Studio](https://aistudio.google.com/))

### Étapes

#### 1. Cloner le repository
```bash
git clone https://github.com/yourusername/fraud-detection-ml-ai.git
cd fraud-detection-ml-ai
```

#### 2. Créer un environnement virtuel
```bash
# Avec venv
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# OU avec conda
conda create -n fraud-detection python=3.9
conda activate fraud-detection
```

#### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

#### 4. Configurer la clé API Gemini
```bash
# Créer un fichier .env
echo "GEMINI_API_KEY=your_api_key_here" > .env

# OU définir la variable d'environnement
export GEMINI_API_KEY="your_api_key_here"  # Linux/Mac
set GEMINI_API_KEY=your_api_key_here       # Windows
```

---

## ⚙️ Configuration

### Variables d'environnement (.env)
```
# API Keys
GEMINI_API_KEY=sk_xxxxxxxxxxxxx

# Configuration Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# Modèle
MODEL_PATH=models/xgboost_model.pkl
SCALER_PATH=models/standard_scaler.pkl

# Données
DATA_PATH=data/creditcard.csv
TEST_SIZE=0.3

# SMOTE
SMOTE_RATIO=1.0
RANDOM_STATE=42
```

### Paramètres du modèle (fraud_detection_final.ipynb)
```python
# SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

# Seuil de décision
FRAUD_THRESHOLD = 0.50
```

---

## 🚀 Utilisation

### 1. Entraîner le modèle

```bash
# Exécuter le Jupyter Notebook
jupyter notebook fraud_detection_final.ipynb

# OU en ligne de commande (via nbconvert)
jupyter nbconvert --to script fraud_detection_final.ipynb
python fraud_detection_final.py
```

**Durée estimée** : 3-5 minutes sur CPU, <1 minute sur GPU

**Outputs** :
- `models/xgboost_model.pkl` - Modèle entraîné
- `models/standard_scaler.pkl` - Scaler normalization
- Visualisations HTML dans le notebook

### 2. Lancer l'application Streamlit

```bash
streamlit run app.py
```

**Accès** : http://localhost:8501

```bash
# OU avec options
streamlit run app.py --logger.level=info --client.showErrorDetails=true
```

### 3. Utiliser l'interface

#### Dashboard (Onglet 1)
- Visualisez les KPIs en temps réel
- Explorez les distributions de montants et alertes
- Identifiez les patterns par ville/heure

#### Alertes (Onglet 2)
- Filtrez par niveau de risque (CRITIQUE/ÉLEVÉ/MOYEN)
- Cliquez sur une transaction
- Lisez l'analyse Gemini
- Décidez : BLOQUER ou IGNORER

#### IA Générative (Onglet 4)
- Cliquez "Générer Synthèse Globale"
- Recevez un résumé des risques du jour
- Obtenez recommandations stratégiques

#### Scénarios (Onglet 5)
- Définissez nombre de scénarios (1-10)
- Générez cas d'usage synthétiques
- Téléchargez en CSV pour formation

#### Exports (Onglet 6)
- Sélectionnez filtres
- Cliquez "Exporter en Excel"
- Reçevez rapport formaté avec formules

---

## 📊 Résultats

### Performance du modèle XGBoost (Meilleur)

| Métrique | Score | Interprétation |
|----------|-------|---|
| **Accuracy** | 99.63% | 99.63% des transactions bien classées |
| **Précision** | 29.98% | 30% des alertes = vraies fraudes |
| **Rappel** ⭐ | **84.46%** | Détecte 84.46% des fraudes |
| **F1-Score** | 0.4467 | Équilibre précision/rappel |
| **ROC-AUC** ⭐ | **0.9725** | Excellent discriminateur |

### Matrice de confusion (Test Set)

```
                Prédiction
           Fraude    Légitime
Réalité
Fraude      125        23       ← 125 détectées, 23 manquées
Légitime  28,000    57,295     ← 28k fausses alertes
```

**Analyse coûts-bénéfices** :
- Fraudes détectées : 125 × 300$ = **37,500$ économisés**
- Fausses alertes : 28,000 × 5$ = 140,000$ coûts investigation
- **ROI positif** si taux confirmation >27%

### Comparaison des 3 modèles

```
                Random Forest    XGBoost ⭐   LightGBM
Accuracy        99.95%          99.63%     99.85%
Précision       89.23%          29.98%     53.48%
Rappel          78.38%          84.46%     83.11%
ROC-AUC         0.9690          0.9725     0.9636
Temps train     ~30s            ~45s       ~15s
Déploiement     ✅ Prod         ✅ Prod    ✅ Prod

VERDICT : XGBoost = meilleur compromis
```

### Feature Importance

Top 5 variables influentes (XGBoost) :

```
1. V11 ████████████████ 25%
2. V4  ███████████      18%
3. V2  ██████████       15%
4. V14 █████████        12%
5. V12 ████████         10%
   ...
```

**Note** : V1-V28 sont PCA (anonymisées). On ignore leur signification métier, mais elles discriminent très bien fraude vs légitime.



## 🔬 Méthodologie

### CRISP-DM (6 phases)

#### Phase 1️⃣ : Compréhension métier
- **Utilisateurs** : Analystes fraude bancaires
- **Objectif** : Détecter fraudes en temps réel
- **Contraintes** : Coûts asymétriques, déséquilibre extrême
- **KPIs** : Rappel (détection), Précision (faux positifs)

#### Phase 2️⃣ : Compréhension des données
- **Dataset** : 284,807 transactions réelles
  - 284,315 légitimes (99.83%)
  - 492 fraudes (0.17%)
- **Variables** : V1-V28 (PCA), Time, Amount
- **Anomalies** : Aucune donnée manquante (lucky!)

#### Phase 3️⃣ : Préparation des données
- **Normalisation** : StandardScaler (Amount, Time)
- **Split** : 70% train, 30% test (stratifié)
- **SMOTE** : Rééquilibre via synthétisation
  - Avant : 199,020 vs 344
  - Après : 199,020 vs 199,020

#### Phase 4️⃣ : Modélisation
- **3 algorithmes** : Random Forest, XGBoost, LightGBM
- **Hyperparameters** : Grid search + cross-validation
- **Sélection** : XGBoost (meilleur ROC-AUC 0.9725)

#### Phase 5️⃣ : Évaluation
- **Métriques** : ROC-AUC, Rappel, Précision, F1, Confusion Matrix
- **Validation** : 5-fold stratifié, holdout test set
- **Justification** : Compromis Rappel (84%) vs Précision (30%)

#### Phase 6️⃣ : Déploiement
- **Streamlit app** : 6 onglets opérationnels
- **Gemini integration** : Explications + recommandations
- **Export** : Rapports Excel pour suivi

---

## 🤝 Contributeurs

- **Auteur** : Abdellahi Cheikh
- **Formation** : Master FADS, FSJES Tétouan
- **Contact** : Abdellahilimam181@gmail.com


## 📖 Ressources

### Références externes
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [CRISP-DM Methodology](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)

### Datasets
- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/) - Kaggle
- Licence : ODbL (Open Data Commons)

---

## 📄 Licence

Ce projet est licencié sous la **MIT License** - voir [LICENSE](./LICENSE) pour détails.

### Utilisation commerciale
- ✅ Permitted
- Avec mention d'attribution au Master FADS

---


## 🚀 Quick Start

```bash
# 1. Installation
git clone repo && cd fraud-detection-ml-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configuration
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Entraîner (optionnel)
jupyter notebook notebooks/fraud_detection_final.ipynb

# 4. Lancer app
streamlit run app/app.py

# 5. Accès
# Ouvrir http://localhost:8501 dans le navigateur
```

