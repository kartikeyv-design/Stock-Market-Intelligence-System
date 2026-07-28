
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import chromadb
from sentence_transformers import SentenceTransformer
import cohere
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# =======================================================================
# PAGE SETUP
# =======================================================================
st.set_page_config(page_title="Stock Intelligence & RAG System", layout="wide")
st.title("Stock Market Intelligence & RAG System")

STOCKS = ['^BSESN', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
COMPANIES = ["Sensex", "Reliance Industries", "Tata Consultancy Services", "Infosys", "HDFC Bank"]
COHERE_API_KEY = "647LWk4EuMFrrq6L8ST0eDTTx3PFMCS1bN9staOp"

# Map Tickers to News Company Names
TICKER_TO_COMPANY = dict(zip(STOCKS, COMPANIES))

# Helper to build featured CSV if not present
def ensure_featured_csv(ticker):
    file_path = f"featured_{ticker}.csv"
    if not os.path.exists(file_path):
        df = yf.download(ticker, period="5y")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().dropna()
        
        df["daily_return"] = df["Close"].pct_change() * 100
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["ma_7"] = df["Close"].rolling(7).mean()
        df["price_ma_ratio"] = df["Close"] / df["ma_7"]
        df["high_low_ratio"] = df["High"] / df["Low"]
        df["ma_14"] = df["Close"].rolling(14).mean()
        df["ma_30"] = df["Close"].rolling(30).mean()
        df["ma_crossover"] = (df["ma_7"] > df["ma_30"]).astype(int)
        df["volatility_7"] = df["daily_return"].rolling(7).std()
        df["volatility_14"] = df["daily_return"].rolling(14).std()
        df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek
        df["month"] = pd.to_datetime(df["Date"]).dt.month
        df["quarter"] = pd.to_datetime(df["Date"]).dt.quarter
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["volume_ma_7"] = df["Volume"].rolling(7).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_ma_7"]
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        
        df.to_csv(file_path, index=False)

# =======================================================================
# TASK 6 & 7: MACHINE LEARNING MODEL PIPELINE
# =======================================================================
@st.cache_resource
def train_model_and_predict(ticker):
    ensure_featured_csv(ticker)
    
    # Load featured stock dataset (Task 5)
    df = pd.read_csv(f"featured_{ticker}.csv")
    df = df.dropna()

    X = df.drop(columns=["Date", "target"])
    y = df["target"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Current sample (latest trading day)
    latest_sample = X.iloc[[-1]]
    prob = pipeline.predict_proba(latest_sample)[0][1]
    pred_signal = "UP 🟢" if prob >= 0.5 else "DOWN 🔴"
    confidence = (prob if prob >= 0.5 else 1 - prob) * 100

    # Task 10: SHAP Values computation
    classifier = pipeline.named_steps["classifier"]
    scaled_latest = pipeline.named_steps["scaler"].transform(latest_sample)
    
    explainer = shap.TreeExplainer(classifier)
    shap_vals = explainer.shap_values(scaled_latest)
    
    # Safely unpack 1D/2D/List SHAP structures
    if isinstance(shap_vals, list):
        # Multi-class or List output
        s_vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        base_v = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    elif len(np.shape(shap_vals)) == 3:
        s_vals = shap_vals[0, :, 1]
        base_v = explainer.expected_value[1]
    else:
        s_vals = shap_vals[0]
        base_v = explainer.expected_value

    s_vals_1d = np.array(s_vals).ravel()

    shap_exp = shap.Explanation(
        values=s_vals_1d,
        base_values=base_v,
        data=latest_sample.iloc[0].values,
        feature_names=X.columns.tolist()
    )

    return pipeline, df, pred_signal, confidence, shap_exp, X_test, y_test

# =======================================================================
# TASK 11: CHROMADB & COHERE RAG PIPELINE
# =======================================================================
@st.cache_resource
def init_rag_system():
    all_news = []
    for company in COMPANIES:
        try:
            df_news = pd.read_csv(f"cleaned_{company}_news.csv")
            df_news["Company"] = company
            all_news.append(df_news)
        except Exception:
            pass

    if not all_news:
        return None, None, None

    news_df = pd.concat(all_news, ignore_index=True)
    news_documents = (
        news_df["title"].fillna("") + ". " + news_df["description"].fillna("")
    ).tolist()

    ids = [str(i) for i in range(len(news_documents))]
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(news_documents, show_progress_bar=False)

    client = chromadb.Client()
    collection = client.create_collection("financial_news_app")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=news_documents
    )

    co_client = cohere.Client(COHERE_API_KEY)
    return collection, embedding_model, co_client

rag_collection, embedding_model, co_client = init_rag_system()

# =======================================================================
# SIDEBAR
# =======================================================================
st.sidebar.header("Navigation & Settings")
selected_asset = st.sidebar.selectbox("Select Asset", STOCKS)

# Pipeline outputs for selected asset
pipeline, df_stock, pred_signal, confidence, shap_exp, X_test, y_test = train_model_and_predict(selected_asset)

# =======================================================================
# TAB DEFINITIONS
# =======================================================================
tab1, tab2, tab3 = st.tabs(["Predictions", "Chat (RAG)", "Comparison"])

# -----------------------------------------------------------------------
# TAB 1: PREDICTIONS & SHAP EXPLANATIONS
# -----------------------------------------------------------------------
with tab1:
    st.header(f"Asset Analysis: {selected_asset}")
    
    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Prediction Signal", pred_signal)
    col2.metric("Confidence Score", f"{confidence:.2f}%")
    
    st.markdown("---")
    
    # Task 4: Interactive Closing Price Plot
    st.subheader("Historical Price Trend")
    fig_price = px.line(
        df_stock, 
        x="Date", 
        y="Close", 
        title=f"{selected_asset} Closing Price History"
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Task 10: SHAP Feature Importance
    st.subheader("SHAP Feature Importance")
    
    # Ensure 1D alignment between names and values
    feature_names = list(shap_exp.feature_names)
    importance_vals = np.abs(shap_exp.values).ravel()
    
    min_len = min(len(feature_names), len(importance_vals))
    
    shap_df = pd.DataFrame({
        "Feature": feature_names[:min_len],
        "Importance": importance_vals[:min_len]
    }).sort_values("Importance", ascending=True).tail(10)

    fig_shap_bar = px.bar(
        shap_df, 
        x="Importance", 
        y="Feature", 
        orientation="h",
        title="Top 10 Feature Contributions to Prediction"
    )
    st.plotly_chart(fig_shap_bar, use_container_width=True)
    
    # Task 10: SHAP Waterfall Plot
    st.subheader("SHAP Waterfall Plot")
    fig_waterfall, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap_exp, show=False)
    st.pyplot(fig_waterfall)

# -----------------------------------------------------------------------
# TAB 2: CHAT (RAG PIPELINE)
# -----------------------------------------------------------------------
with tab2:
    st.header("💬 Financial Assistant RAG Chat")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    def query_rag(question):
        if not rag_collection:
            return "RAG data is missing. Please ensure `cleaned_*_news.csv` files exist."

        q_embed = embedding_model.encode([question]).tolist()
        results = rag_collection.query(query_embeddings=q_embed, n_results=3)
        docs = results["documents"][0]
        context = "\n\n".join(docs)

        prompt = f"""
You are a financial news assistant.
Use ONLY the information below to answer the question.

News Articles:
{context}

Question:
{question}

Provide a concise answer based only on the news articles.
"""
        response = co_client.chat(
            model="command-a-03-2025",
            message=prompt,
            temperature=0.2
        )
        return response.text

    user_input = st.chat_input("Ask a question about news or earnings reports...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching ChromaDB & generating response..."):
                answer = query_rag(user_input)
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# -----------------------------------------------------------------------
# TAB 3: MULTI-ASSET COMPARISON & BUSINESS COSTS
# -----------------------------------------------------------------------
with tab3:
    st.header("🌐 Multi-Asset Comparison")
    
    cost_fp, cost_fn = 100, 50
    comparison_data = []

    for stock in STOCKS:
        pip, _, sig, conf, _, X_t, y_t = train_model_and_predict(stock)
        probs = pip.predict_proba(X_t)[:, 1]
        acc = pip.score(X_t, y_t)

        # Task 9: Cost Optimization
        min_cost = float("inf")
        opt_thresh = 0.5

        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
            cost = (fp * cost_fp) + (fn * cost_fn)
            if cost < min_cost:
                min_cost = cost
                opt_thresh = round(thresh, 2)

        comparison_data.append({
            "Asset": stock,
            "Accuracy (%)": round(acc * 100, 2),
            "Prediction": sig,
            "Confidence (%)": round(conf, 2),
            "Optimal Threshold": opt_thresh,
            "Min Business Cost (₹)": min_cost
        })

    df_summary = pd.DataFrame(comparison_data)

    st.subheader("Asset Comparison Metrics")
    st.dataframe(df_summary, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_metrics = px.bar(
            df_summary, 
            x="Asset", 
            y="Accuracy (%)", 
            color="Min Business Cost (₹)", 
            title="Model Accuracy vs Cost"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    with col_b:
        fig_pred = px.bar(
            df_summary, 
            x="Asset", 
            y="Confidence (%)", 
            color="Prediction", 
            title="Prediction Confidence across Assets"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
