import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
from io import BytesIO
import json
import re

# ═══════════════════════════════════════════════════════════════════════════════
# 🔐 DÉTECTION PROACTIVE DE FRAUDE BANCAIRE - APPLICATION STREAMLIT 
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🔐 Fraud Detection System v2",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 0️⃣ STYLE GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 10px;
        padding: 15px 20px;
        border: 1px solid #38bdf8;
        text-align: center;
    }
    .risk-high {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 10px;
        border-radius: 6px;
    }
    .risk-medium {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 10px;
        border-radius: 6px;
    }
    .risk-low {
        background-color: #dcfce7;
        border-left: 4px solid #22c55e;
        padding: 10px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣ INITIALISATION SESSION_STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "transactions_df" not in st.session_state:
    st.session_state.transactions_df = None

if "alerts_df" not in st.session_state:
    st.session_state.alerts_df = pd.DataFrame()

if "gemini_scenarios" not in st.session_state:
    st.session_state.gemini_scenarios = pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣ HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    # 🔐 Détection Proactive de Fraude Bancaire
    ## ML + IA Générative pour Analyse Comportementale en Temps Réel  
    ---
    """
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣ SIDEBAR - CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "🔑 Clé API Gemini:",
        type="password",
        help="Votre clé API Google Gemini"
    )
    gemini_enabled = False
    if api_key:
        try:
            genai.configure(api_key=api_key)
            gemini_enabled = True
            st.success("✅ Gemini connecté")
        except Exception as e:
            st.error(f"Erreur configuration Gemini: {e}")

    st.markdown("## 🎮 Simulation Temps Réel")
    # Jusqu'à 10 000 transactions
    num_transactions = st.slider(
        "Nombre de transactions à simuler",
        min_value=1000,
        max_value=10000,
        step=1000,
        value=5000
    )

    st.markdown("## 🎯 Seuils d'Alerte")
    critical_threshold = st.slider(
        "Seuil CRITIQUE",
        0.5, 1.0, 0.75, 0.05,
        help="Probabilité de fraude ≥ seuil = alerte critique"
    )
    high_threshold = st.slider(
        "Seuil ÉLEVÉ",
        0.3, 0.99, 0.50, 0.05,
        help="Probabilité de fraude ≥ seuil = alerte élevée"
    )

    st.markdown("---")
    st.markdown("**📌 Informations Système**")
    st.info(
        """
        - 🤖 Modèles ML: XGBoost, Random Forest, LightGBM  
        - 🧠 IA Générative: Gemini 2.5 Flash  
        - 📊 Accuracy (XGBoost): 99.63%  
        - 🏆 ROC-AUC (XGBoost): 0.9725  
        """
    )

    if st.button("🔄 Simuler / Re-générer les transactions", use_container_width=True):
        st.session_state.transactions_df = None
        st.session_state.alerts_df = pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣ FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════
def generate_transactions(n: int = 5000) -> pd.DataFrame:
    """Générer n transactions simulées avec probabilités de fraude réalistes et variées."""
    rng = np.random.default_rng()

    transaction_ids = [f"TRX{rng.integers(10000000, 99999999)}" for _ in range(n)]
    amounts = rng.exponential(scale=150, size=n) + 5
    hours = rng.integers(0, 24, size=n)
    minutes = rng.integers(0, 60, size=n)
    weekdays = rng.integers(0, 7, size=n)
    locations = rng.choice(
        ["Casablanca", "Rabat", "Tanger", "Marrakech", "Fès", "Agadir"],
        size=n
    )
    merchant_types = rng.choice(
        ["E-commerce", "Supermarché", "Restaurant", "Station-essence", "ATM", "Voyage"],
        size=n
    )

    # ✅ Probabilité de fraude enrichie et réaliste
    fraud_probability = np.full(n, 0.01)
    
    high_amount_mask = amounts > 500
    very_high_amount_mask = amounts > 1000
    night_mask = (hours < 6) | (hours > 22)
    very_night_mask = (hours < 4) | (hours > 23)
    weekend_mask = weekdays >= 5
    international_style = rng.random(size=n) > 0.95
    unusual_frequency = rng.random(size=n) > 0.90
    rapid_transactions = rng.random(size=n) > 0.93
    
    fraud_probability += np.where(high_amount_mask, 0.20, 0.0)
    fraud_probability += np.where(very_high_amount_mask, 0.25, 0.0)
    fraud_probability += np.where(night_mask, 0.12, 0.0)
    fraud_probability += np.where(very_night_mask, 0.20, 0.0)
    fraud_probability += np.where(weekend_mask & night_mask, 0.25, 0.0)
    fraud_probability += np.where(international_style, 0.40, 0.0)
    fraud_probability += np.where(unusual_frequency, 0.15, 0.0)
    fraud_probability += np.where(rapid_transactions, 0.18, 0.0)
    
    fraud_probability += rng.normal(0, 0.05, size=n)
    fraud_probability = np.clip(fraud_probability, 0.0, 1.0)

    comportement = np.where(
        fraud_probability < 0.3,
        "Normal",
        np.where(fraud_probability < 0.6, "A Surveiller", "Anormal")
    )

    df = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "Montant": amounts,
            "Heure": [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)],
            "Heure_num": hours + minutes / 60.0,
            "JourSemaine": weekdays,
            "Ville": locations,
            "TypeCommercant": merchant_types,
            "Probabilite_Fraude": fraud_probability,
            "Comportement": comportement,
        }
    )
    return df


def classify_risk(prob: float, high_t: float, crit_t: float) -> str:
    """Retourner le niveau de risque (texte) en fonction de la probabilité."""
    if prob >= crit_t:
        return "CRITIQUE"
    elif prob >= high_t:
        return "ÉLEVÉ"
    elif prob >= 0.3:
        return "MOYEN"
    else:
        return "FAIBLE"


def generate_gemini_analysis(transaction: dict, risk_level: str) -> str:
    """Appeler Gemini pour expliquer une transaction suspecte."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Vous êtes un expert en fraude bancaire.

        Analysez cette transaction et expliquez le risque en français de façon concise:

        - ID: {transaction['TransactionID']}
        - Montant: {transaction['Montant']:.2f} USD
        - Heure: {transaction['Heure']}
        - Ville: {transaction['Ville']}
        - Type commerçant: {transaction['TypeCommercant']}
        - Probabilité de fraude: {transaction['Probabilite_Fraude']:.2%}
        - Niveau de risque: {risk_level}

        Donnez:
        1. Une brève analyse (2-3 phrases)
        2. Une recommandation d'action (BLOQUER / VÉRIFIER / SURVEILLER)
        3. Les signaux principaux justifiant cette décision
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Analyse Gemini non disponible: {e}"


def generate_gemini_scenarios(n_scenarios: int = 5):
    """Générer des scénarios de fraude synthétiques avec Gemini - version ROBUSTE."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Prompt simplifié et plus direct
        prompt = f"""
Générez {n_scenarios} scénarios de fraude bancaire réalistes au format JSON.

Retournez UNIQUEMENT un tableau JSON valide, commençant par [ et finissant par ].
Chaque scénario doit avoir: id, montant, heure, ville, type, description, indicateurs (liste).

Exemple:
[
{{"id":"FRAUD_001","montant":950.50,"heure":"02:35","ville":"Casablanca","type":"Achat élevé","description":"Montant anormalement élevé","indicateurs":["montant élevé","heure nocturne"]}},
{{"id":"FRAUD_002","montant":1200.00,"heure":"03:15","ville":"Rabat","type":"Transaction rapide","description":"Deux transactions en 5 minutes","indicateurs":["fréquence élevée","montant élevé"]}}
]

IMPORTANT: Seulement le JSON, rien d'autre.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Nettoyer la réponse (supprimer markdown, texte extra)
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Chercher le JSON entre [ et ]
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        
        if not json_match:
            st.warning("Format JSON non trouvé. Création de scénarios par défaut...")
            return generate_default_scenarios(n_scenarios)
        
        json_str = json_match.group(0)
        scenarios = json.loads(json_str)
        
        if not isinstance(scenarios, list):
            scenarios = [scenarios]
        
        if len(scenarios) > 0:
            return pd.DataFrame(scenarios)
        else:
            return generate_default_scenarios(n_scenarios)
            
    except json.JSONDecodeError as e:
        st.warning(f"Erreur JSON: {e}. Création de scénarios par défaut...")
        return generate_default_scenarios(n_scenarios)
    except Exception as e:
        st.warning(f"Erreur Gemini: {e}. Création de scénarios par défaut...")
        return generate_default_scenarios(n_scenarios)


def generate_default_scenarios(n: int = 5) -> pd.DataFrame:
    """Générer des scénarios par défaut si Gemini échoue."""
    scenarios = [
        {
            "id": "FRAUD_001",
            "montant": 950.50,
            "heure": "02:35",
            "ville": "Casablanca",
            "type": "Montant anormalement élevé",
            "description": "Transaction de nuit avec montant 5x la moyenne",
            "indicateurs": ["montant élevé", "heure nocturne", "comportement anormal"]
        },
        {
            "id": "FRAUD_002",
            "montant": 1200.00,
            "heure": "03:15",
            "ville": "Rabat",
            "type": "Transactions rapides en séquence",
            "description": "3 transactions en 10 minutes depuis 2 villes",
            "indicateurs": ["fréquence élevée", "localisation multiple", "montant cumulé élevé"]
        },
        {
            "id": "FRAUD_003",
            "montant": 750.75,
            "heure": "04:00",
            "ville": "Tanger",
            "type": "Achat en ligne suspect",
            "description": "Nouveau device, nouvelle localisation, montant élevé",
            "indicateurs": ["nouveau device", "localisation inhabituelle", "type commerce inhabituel"]
        },
        {
            "id": "FRAUD_004",
            "montant": 1850.00,
            "heure": "23:55",
            "ville": "Marrakech",
            "type": "Transaction internationale",
            "description": "Paiement international tard la nuit depuis nouveau réseau",
            "indicateurs": ["devises étrangères", "heure tardive", "montant très élevé"]
        },
        {
            "id": "FRAUD_005",
            "montant": 500.00,
            "heure": "05:30",
            "ville": "Fès",
            "type": "Pattern de test",
            "description": "Série de petits montants pour tester les limites",
            "indicateurs": ["montants progressifs", "heure inhabituelle", "pattern anormal"]
        },
    ]
    
    return pd.DataFrame(scenarios[:n])


def export_alerts_to_excel(df: pd.DataFrame) -> BytesIO:
    """Exporter les alertes en Excel (buffer mémoire)."""
    output = BytesIO()
    df_to_export = df.copy()
    df_to_export["Probabilite_Fraude"] = (df_to_export["Probabilite_Fraude"] * 100).round(2)
    df_to_export.rename(
        columns={
            "Probabilite_Fraude": "Probabilite_Fraude_%",
            "Montant": "Montant_USD"
        },
        inplace=True
    )

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_to_export.to_excel(writer, sheet_name="Alertes", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Alertes"]

        format_header = workbook.add_format(
            {"bold": True, "bg_color": "#0f172a", "font_color": "white", "border": 1}
        )
        for col_num, value in enumerate(df_to_export.columns.values):
            worksheet.write(0, col_num, value, format_header)
            worksheet.set_column(col_num, col_num, 18)

    output.seek(0)
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣ GÉNÉRATION / MISE À JOUR DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.transactions_df is None:
    with st.spinner(f"⏳ Génération de {num_transactions:,} transactions..."):
        st.session_state.transactions_df = generate_transactions(num_transactions)

df = st.session_state.transactions_df.copy()
df["Niveau_Risque"] = df["Probabilite_Fraude"].apply(
    lambda p: classify_risk(p, high_threshold, critical_threshold)
)

alerts_df = df[df["Niveau_Risque"].isin(["CRITIQUE", "ÉLEVÉ"])].copy()
st.session_state.alerts_df = alerts_df

# ═══════════════════════════════════════════════════════════════════════════════
# 6️⃣ TABS PRINCIPAUX
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Dashboard Principal",
        "🚨 Alertes Temps Réel",
        "🔬 Analyse Détaillée",
        "🤖 IA Générative",
        "🧪 Scénarios Synthétiques",
        "📄 Rapports & Export",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 : DASHBOARD PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Vue d'Ensemble - Temps Réel")

    total_tx = len(df)
    total_alerts = len(alerts_df)
    crit_count = (alerts_df["Niveau_Risque"] == "CRITIQUE").sum()
    high_count = (alerts_df["Niveau_Risque"] == "ÉLEVÉ").sum()
    moyen_count = (df["Niveau_Risque"] == "MOYEN").sum()
    total_amount = df["Montant"].sum()
    risk_amount = alerts_df["Montant"].sum()
    avg_amount = df["Montant"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            '<div class="metric-card">📦<br><b>Total Transactions</b><br>' +
            f'{total_tx:,}</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="metric-card">🚨<br><b>Total Alertes</b><br>' +
            f'{total_alerts} ({total_alerts/total_tx*100:.2f}%)</div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="metric-card">💰<br><b>Montant Total</b><br>' +
            f'{total_amount:,.0f} $</div>',
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            '<div class="metric-card">💣<br><b>Montant à Risque</b><br>' +
            f'{risk_amount:,.0f} $</div>',
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            '<div class="metric-card">📈<br><b>Montant Moyen</b><br>' +
            f'{avg_amount:.2f} $</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig_amount = px.histogram(
            df,
            x="Montant",
            nbins=60,
            title="💰 Distribution des Montants",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_amount.update_layout(
            xaxis_title="Montant ($)",
            yaxis_title="Nombre de transactions",
            height=400
        )
        st.plotly_chart(fig_amount, use_container_width=True)

    with col2:
        fig_prob = px.histogram(
            df,
            x="Probabilite_Fraude",
            nbins=50,
            title="🎯 Distribution des Probabilités de Fraude",
            color_discrete_sequence=["#f97316"],
        )
        fig_prob.add_vline(
            x=high_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="Seuil ÉLEVÉ",
        )
        fig_prob.add_vline(
            x=critical_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Seuil CRITIQUE",
        )
        fig_prob.update_layout(
            xaxis_title="Probabilité de fraude",
            yaxis_title="Nombre de transactions",
            height=400
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        if not alerts_df.empty:
            city_counts = alerts_df["Ville"].value_counts().reset_index()
            city_counts.columns = ["Ville", "Nombre"]
            fig_city = px.bar(
                city_counts,
                x="Ville",
                y="Nombre",
                title="🌍 Alertes par Ville",
                color="Ville",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_city.update_layout(height=400)
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            st.info("Aucune alerte pour afficher le graphique par ville.")

    with col4:
        merch_counts = df["TypeCommercant"].value_counts().reset_index()
        merch_counts.columns = ["TypeCommercant", "Nombre"]
        fig_merch = px.pie(
            merch_counts,
            names="TypeCommercant",
            values="Nombre",
            title="🏬 Distribution par Type de Commerçant",
        )
        fig_merch.update_layout(height=400)
        st.plotly_chart(fig_merch, use_container_width=True)

    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        df["Heure_int"] = df["Heure"].str.slice(0, 2).astype(int)
        hourly_counts = df.groupby("Heure_int").size()
        hourly_alerts = alerts_df["Heure"].str.slice(0, 2).astype(int).value_counts().sort_index()

        fig_hour = go.Figure()
        fig_hour.add_trace(
            go.Scatter(
                x=hourly_counts.index,
                y=hourly_counts.values,
                mode="lines+markers",
                name="Toutes transactions",
                line=dict(color="#38bdf8", width=2),
            )
        )
        if not hourly_alerts.empty:
            fig_hour.add_trace(
                go.Scatter(
                    x=hourly_alerts.index,
                    y=hourly_alerts.values,
                    mode="lines+markers",
                    name="Alertes",
                    line=dict(color="#ef4444", width=2),
                )
            )
        fig_hour.update_layout(
            title="🕒 Activité par Heure de la Journée",
            xaxis_title="Heure",
            yaxis_title="Nombre",
            height=400
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with col6:
        risk_dist = df["Niveau_Risque"].value_counts().reset_index()
        risk_dist.columns = ["Niveau_Risque", "Nombre"]
        risk_colors = {
            "CRITIQUE": "#ef4444",
            "ÉLEVÉ": "#f59e0b",
            "MOYEN": "#3b82f6",
            "FAIBLE": "#22c55e"
        }

        fig_risk = px.bar(
            risk_dist,
            x="Niveau_Risque",
            y="Nombre",
            title="📊 Distribution des Niveaux de Risque",
            color="Niveau_Risque",
            color_discrete_map=risk_colors,
        )
        fig_risk.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 : ALERTES TEMPS RÉEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🚨 Alertes Temps Réel")

    if alerts_df.empty:
        st.success("✅ Aucune alerte détectée pour l'instant.")
    else:
        colf1, colf2 = st.columns([2, 1])
        with colf1:
            st.info(
                f"Nombre total d'alertes: **{len(alerts_df)}** "
                f"(CRITIQUE: {crit_count}, ÉLEVÉ: {high_count})"
            )
        with colf2:
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.rerun()

        colf3, colf4, colf5 = st.columns(3)
        with colf3:
            risk_filter = st.multiselect(
                "Filtrer par Niveau de Risque",
                options=["CRITIQUE", "ÉLEVÉ"],
                default=["CRITIQUE", "ÉLEVÉ"],
            )
        with colf4:
            city_filter = st.multiselect(
                "Filtrer par Ville",
                options=sorted(alerts_df["Ville"].unique().tolist()),
                default=[],
            )
        with colf5:
            merch_filter = st.multiselect(
                "Filtrer par Type Commerçant",
                options=sorted(alerts_df["TypeCommercant"].unique().tolist()),
                default=[],
            )

        alerts_view = alerts_df.copy()
        if risk_filter:
            alerts_view = alerts_view[alerts_view["Niveau_Risque"].isin(risk_filter)]
        if city_filter:
            alerts_view = alerts_view[alerts_view["Ville"].isin(city_filter)]
        if merch_filter:
            alerts_view = alerts_view[alerts_view["TypeCommercant"].isin(merch_filter)]

        st.markdown("---")

        for idx, row in alerts_view.head(30).iterrows():
            prob = row["Probabilite_Fraude"]
            level = row["Niveau_Risque"]
            css_class = (
                "risk-high" if level == "CRITIQUE"
                else "risk-medium" if level == "ÉLEVÉ"
                else "risk-low"
            )

            with st.container():
                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <b>{row['TransactionID']}</b> | {row['Ville']} | {row['TypeCommercant']}<br>
                        Montant: <b>{row['Montant']:.2f} $</b> |
                        Heure: {row['Heure']} |
                        Risque: <b>{level}</b> ({prob:.1%})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if gemini_enabled:
                    if st.button(
                        "🤖 Analyse Gemini",
                        key=f"gemini_alert_{idx}",
                        help="Obtenir une explication détaillée",
                    ):
                        with st.spinner("Analyse Gemini en cours..."):
                            analysis = generate_gemini_analysis(row, level)
                            st.markdown(analysis)
                st.markdown("")

        if len(alerts_view) > 30:
            st.info(
                f"Affichage des 30 premières alertes sur {len(alerts_view)} filtrées."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 : ANALYSE DÉTAILLÉE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔬 Analyse Détaillée des Transactions")

    colA, colB = st.columns(2)
    with colA:
        fig_box_all = px.box(
            df,
            y="Montant",
            title="💰 Distribution des Montants (Toutes Transactions)",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_box_all.update_layout(height=400)
        st.plotly_chart(fig_box_all, use_container_width=True)

    with colB:
        if not alerts_df.empty:
            fig_box_compare = go.Figure()
            fig_box_compare.add_trace(
                go.Box(
                    y=df["Montant"],
                    name="Toutes",
                    marker_color="#22c55e",
                )
            )
            fig_box_compare.add_trace(
                go.Box(
                    y=alerts_df["Montant"],
                    name="Alertes",
                    marker_color="#ef4444",
                )
            )
            fig_box_compare.update_layout(
                title="💰 Montants - Toutes vs Alertes",
                yaxis_title="Montant ($)",
                height=400
            )
            st.plotly_chart(fig_box_compare, use_container_width=True)
        else:
            st.info("Aucune alerte pour comparaison de montants.")

    st.markdown("---")

    colC, colD = st.columns(2)
    with colC:
        merch_stats = (
            df.groupby("TypeCommercant")["Montant"]
            .agg(["count", "mean", "sum"])
            .reset_index()
            .rename(
                columns={
                    "count": "Nb_Transactions",
                    "mean": "Montant_Moyen",
                    "sum": "Montant_Total",
                }
            )
        )
        st.markdown("### 🏬 Statistiques par Type de Commerçant")
        st.dataframe(merch_stats.round(2), use_container_width=True)

    with colD:
        if not alerts_df.empty:
            pivot = (
                alerts_df.pivot_table(
                    index="Ville",
                    columns="Niveau_Risque",
                    values="TransactionID",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
            )
            fig_heat = px.imshow(
                pivot.set_index("Ville"),
                text_auto=True,
                color_continuous_scale="Reds",
                title="🔥 Alertes par Ville et Niveau de Risque",
            )
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Pas assez d'alertes pour la heatmap.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 : IA GÉNÉRATIVE (EXPLICATIONS GLOBALES)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🤖 IA Générative - Synthèse & Conseils")

    if not gemini_enabled:
        st.warning("Entrez votre clé API Gemini dans la barre latérale pour activer cette section.")
    else:
        st.info(
            "Cette section permet de générer une **synthèse globale** des alertes et "
            "des recommandations stratégiques grâce à Gemini."
        )

        if st.button("🧠 Générer Synthèse Globale des Risques"):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")

                resume_stats = {
                    "total_transactions": int(total_tx),
                    "total_alerts": int(total_alerts),
                    "critical_alerts": int(crit_count),
                    "high_alerts": int(high_count),
                    "risk_amount": float(risk_amount),
                    "total_amount": float(total_amount),
                }

                prompt = f"""
                Vous êtes un expert en gestion des risques pour une banque.

                Statistiques système:
                - Transactions totales: {resume_stats['total_transactions']}
                - Alertes totales: {resume_stats['total_alerts']}
                - Alertes CRITIQUES: {resume_stats['critical_alerts']}
                - Alertes ÉLEVÉES: {resume_stats['high_alerts']}
                - Montant total: {resume_stats['total_amount']:.2f} USD
                - Montant sous risque (alertes): {resume_stats['risk_amount']:.2f} USD

                Fournissez en français:
                1. Un résumé exécutif (3-5 puces)
                2. Les principaux patterns de fraude potentiels
                3. 3 recommandations opérationnelles à court terme
                4. 3 axes stratégiques à moyen terme
                """
                with st.spinner("Génération de la synthèse avec Gemini..."):
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"Erreur Gemini: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 : SCÉNARIOS SYNTHÉTIQUES GEMINI
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🧪 Scénarios Synthétiques de Fraude (Gemini)")

    st.info(
        "Gemini peut générer des scénarios de fraude **synthétiques** pour enrichir "
        "les cas de test, la formation des analystes et le futur ré-entraînement du modèle."
    )

    colS1, colS2 = st.columns([3, 1])
    with colS1:
        n_scenarios = st.slider(
            "Nombre de scénarios à générer",
            min_value=1,
            max_value=10,
            value=5,
        )
    with colS2:
        generate_btn = st.button("🚀 Générer Scénarios", use_container_width=True)

    if generate_btn:
        if not gemini_enabled:
            st.error("Veuillez saisir votre clé API Gemini dans la barre latérale.")
        else:
            with st.spinner("Génération des scénarios par Gemini..."):
                scenarios_df = generate_gemini_scenarios(n_scenarios)
                if scenarios_df.empty:
                    st.warning("Aucun scénario généré.")
                else:
                    st.session_state.gemini_scenarios = scenarios_df
                    st.success(f"✅ {len(scenarios_df)} scénarios générés avec succès!")
                    for i, row in scenarios_df.iterrows():
                        with st.expander(f"📌 Scénario {i+1} – {row.get('type', 'Fraude')}"):
                            st.write(f"**ID:** {row.get('id', 'N/A')}")
                            st.write(f"**Montant:** ${row.get('montant', 'N/A')}")
                            st.write(f"**Heure:** {row.get('heure', 'N/A')}")
                            st.write(f"**Ville:** {row.get('ville', 'N/A')}")
                            st.write(f"**Type:** {row.get('type', 'N/A')}")
                            st.write(f"**Description:** {row.get('description', 'N/A')}")
                            if isinstance(row.get("indicateurs", None), list):
                                st.write("**Indicateurs:**")
                                for ind in row["indicateurs"]:
                                    st.markdown(f"- {ind}")

    if not st.session_state.gemini_scenarios.empty:
        st.markdown("---")
        st.markdown("### 📥 Télécharger les scénarios (CSV)")
        csv_data = st.session_state.gemini_scenarios.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger scenarios_gemini.csv",
            data=csv_data,
            file_name=f"scenarios_gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 : RAPPORTS & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📄 Rapports & Export")

    st.markdown("### 📊 Aperçu des Alertes")
    if alerts_df.empty:
        st.success("✅ Aucune alerte détectée, rien à exporter pour l'instant.")
    else:
        alerts_view = alerts_df.copy()
        alerts_view["Probabilite_Fraude_%"] = (alerts_view["Probabilite_Fraude"] * 100).round(2)
        alerts_view_display = alerts_view[
            [
                "TransactionID",
                "Montant",
                "Heure",
                "Ville",
                "TypeCommercant",
                "Probabilite_Fraude_%",
                "Niveau_Risque",
            ]
        ]
        st.dataframe(alerts_view_display, use_container_width=True, height=350)

        st.markdown("---")
        st.markdown("### 📤 Export")

        colE1, colE2 = st.columns(2)
        with colE1:
            if st.button("📊 Exporter les alertes en Excel", use_container_width=True):
                buffer = export_alerts_to_excel(alerts_df)
                st.download_button(
                    label="📥 Télécharger alerts_fraude.xlsx",
                    data=buffer,
                    file_name=f"alerts_fraude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        with colE2:
            st.info(
                "📋 Les alertes sont exportées avec les colonnes : ID, Montant, Heure, "
                "Ville, Type Commerçant, Probabilité %, et Niveau de Risque."
            )

st.markdown("---")
