
import streamlit as st
import pickle
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image
import plotly.express as px

# LOAD MODELS 
lr = pickle.load(open("models/lr_model.pkl", "rb"))
nb = pickle.load(open("models/nb_model.pkl", "rb"))
svm = pickle.load(open("models/svm_model.pkl", "rb"))
xgb = pickle.load(open("models/xgboost_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

tokenizer = BertTokenizer.from_pretrained("bert_model")
bert_model = BertForSequenceClassification.from_pretrained("bert_model")

# PAGE CONFIG 
st.set_page_config(
    page_title="ConflictGuard",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>

/* Main container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:900px;
}
            
.section-card{
    background:linear-gradient(135deg,#16213E,#0F172A);
    padding:22px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,.08);
    margin-bottom:22px;
    box-shadow:0 8px 25px rgba(0,0,0,.35);
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
    background:#0B5ED7;
    color:white;
    border:none;
}

.stButton>button:hover{
    background:#084298;
    color:white;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"]{
    border-radius:10px;
}

/* Text Area */
textarea{
    border-radius:10px !important;
}

/* Dataframe */
[data-testid="stDataFrame"]{
    border-radius:10px;
}
            
.stButton > button{
    width:100%;
    height:60px;
    border-radius:15px;
    font-size:22px;
    font-weight:bold;
    background:linear-gradient(90deg,#1976D2,#00C853);
    color:white;
    border:none;
    transition:0.3s;
}

.stButton > button:hover{
    transform:scale(1.02);
    box-shadow:0px 0px 20px rgba(0,200,83,0.4);
}
            
.stButton button:active{
transform:scale(.98);
}
            
.stApp{
background:
radial-gradient(circle at top,#1B3A5B 0%,#0E1117 35%),
linear-gradient(180deg,#0E1117,#131722,#0B0F17);
}
            
.streamlit-expanderHeader{
font-size:20px;
font-weight:bold;
color:#42A5F5;
}

details{
background:#171C28;
border-radius:18px;
padding:10px;
border:1px solid #2B3445;
}
            

</style>
""", unsafe_allow_html=True)

# TITLE 
# LOGO
logo = Image.open("logo.png")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(logo, width=400)

st.markdown("""
<div style="
height:12px;
background:radial-gradient(circle,#1976D2,transparent);
filter:blur(20px);
margin-top:-20px;
margin-bottom:20px;">
</div>
""",unsafe_allow_html=True)

st.markdown(
"<hr style='margin-top:5px;margin-bottom:25px;'>",
unsafe_allow_html=True
)

st.divider()

st.markdown("""
<h1 style="
text-align:center;
font-size:56px;
font-weight:800;
background:linear-gradient(90deg,#4FC3F7,#00E676);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:0px;">
🛡️ ConflictGuard
</h1>
""",unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;
font-size:20px;
color:#AAB8C2;
margin-top:-10px;'>
Hybrid AI-Based Conflict News Verification System
</p>
""", unsafe_allow_html=True)

with st.expander("ℹ️ About ConflictGuard", expanded=True):
    st.markdown("""
## 🛡️ About ConflictGuard

ConflictGuard is an AI-powered hybrid system developed to verify the credibility of conflict-related news using Machine Learning and Deep Learning models.

### ✨ Key Features

- Multi-model prediction
- Confidence-based verification
- Model comparison
- Accuracy comparison

### 🌍 Supported Domains

- Israel/US – Iran Conflict
- Russia – Ukraine Conflict

> ⚠ Research Prototype
""")
    

st.divider()

# ================= INPUT SECTION =================

sample_news = [
    "Iran launched missile strikes on Israel after military escalation.",
    "The United Nations called for an immediate ceasefire in Gaza.",
    "Russia launched fresh attacks on Ukrainian infrastructure.",
    "Oil prices surged after attacks in the Strait of Hormuz.",
    "The United States announced additional military support for Ukraine.",
    "Iran claimed new missile technology during the ongoing conflict.",
    "Israel announced evacuation of civilians near the border.",
    "Russia carried out overnight drone attacks on Kyiv.",
    "Iran officially announced it would destroy all US military bases worldwide within the next 24 hours.",
    "China officially entered the Iran-Israel war with 5 lakh soldiers."
]

st.header("📰 Input Method")

st.info("💡 Tip: Use a conflict-related news headline or short article for better prediction accuracy.")

input_method = st.radio(
    "Choose Input Method",
    ["Sample News", "Enter Your Own News"]
)

if input_method == "Sample News":
    selected_news = st.selectbox(
        "Choose Sample News",
        sample_news
    )
else:
    selected_news = st.text_area(
        "Enter News",
        height=180,
        placeholder="Enter or paste a conflict-related news article..."
    )

st.header("🧾 Selected News")

st.caption("The selected news article will be analyzed by the selected AI model.")

if selected_news.strip():
    st.write(selected_news)
else:
    st.info("Please select or enter a news article.")


st.divider()

# MODEL SELECTION 
st.markdown("---")
st.header("🤖 Select AI Model")
model_option = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Naive Bayes", "SVM", "XGBoost", "BERT"]
)


# PREDICTION

st.caption("Select a model and verify the credibility of the news article.")

if st.button("🛡️ Verify Credibility"):

    # Input Validation
    if not selected_news.strip():
        st.warning("⚠ Please enter or select a news article.")
        st.stop()

    results = {}
    predictions = {}

    vec = vectorizer.transform([selected_news])

    models = {
        "Logistic Regression": lr,
        "Naive Bayes": nb,
        "SVM": svm,
        "XGBoost": xgb
    }

    # Traditional Models
    for name, model in models.items():
        proba = model.predict_proba(vec)[0]

        real = float(proba[1])
        fake = float(proba[0])

        results[name] = {"real": real, "fake": fake}
        predictions[name] = "REAL" if real > fake else "FAKE"

    # BERT
    inputs = tokenizer(
        selected_news,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    outputs = bert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    real = float(probs[1].item())
    fake = float(probs[0].item())

    results["BERT"] = {"real": real, "fake": fake}
    predictions["BERT"] = "REAL" if real > fake else "FAKE"

    # SELECTED MODEL RESULT
    selected = results[model_option]
    real_prob = selected["real"]
    fake_prob = selected["fake"]

    color = "#00E676" if real_prob > fake_prob else "#FF5252"

    st.markdown(f"""
    <div style="
    background:linear-gradient(135deg,#1E293B,#111827);
    padding:25px;
    border-radius:15px;
    border-left:8px solid {color};
    box-shadow:0 8px 30px rgba(0,0,0,.45);
    margin-bottom:20px;
    ">

    <h3>🎯 Verification Result</h3>

    <h2 style="color:{color};">
    Prediction : {"REAL ✅" if real_prob > fake_prob else "FAKE ❌"}
    </h2>

    <p>
    Confidence :
    <b>{max(real_prob,fake_prob)*100:.2f}%</b>
    </p>

    </div>
    """, unsafe_allow_html=True)

    if real_prob > fake_prob:
        st.success(f"✅ Likely REAL News ({real_prob*100:.2f}%)")
        st.balloons()
    else:
        st.error(f"🚨 Likely FAKE News ({fake_prob*100:.2f}%)")

    # PROBABILITY
    st.write("### 📈 Prediction Confidence")
    st.progress(real_prob)
    col1,col2=st.columns(2)

    with col1:
        st.metric("✅ Real Probability",f"{real_prob*100:.2f}%")

    with col2:
        st.metric("❌ Fake Probability",f"{fake_prob*100:.2f}%")

    # TABLE
    st.markdown("---")
    st.subheader("⚔️ Model Confidence Comparison")

    comp_df = pd.DataFrame({
        "Model": results.keys(),
        "Real %": [round(v["real"]*100,2) for v in results.values()],
        "Fake %": [round(v["fake"]*100,2) for v in results.values()]
    })

    st.dataframe(comp_df, use_container_width=True,height=125)

    #  GRAPH 
    st.subheader("📈 Confidence Comparison Graph")

    fig = px.bar(
        comp_df,
        x="Model",
        y=["Real %", "Fake %"],
        barmode="group",
        text_auto=".1f",
        color_discrete_sequence=["#00C853", "#FF5252"],
        template="plotly_dark",
        height=420
    )

    fig.update_layout(
        title="Confidence Comparison",
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(size=15),
        legend_title=""
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Higher percentage indicates stronger confidence in the predicted class.")

    accuracy_data = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "SVM", "XGBoost", "BERT"],
    "Accuracy (%)": [94.05, 90.59, 94.32, 89.38, 93.58]
    })

    #  ACCURACY GRAPH 
    st.subheader("📈 Accuracy Comparison Graph")

    fig2 = px.bar(
        accuracy_data,
        x="Model",
        y="Accuracy (%)",
        text="Accuracy (%)",
        color="Accuracy (%)",
        color_continuous_scale="Blues",
        template="plotly_dark",
        height=420
    )

    fig2.update_layout(
        title="Model Accuracy Comparison",
        plot_bgcolor="#111827",
        paper_bgcolor="#111827"
    )

    fig2.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    fig2.update_coloraxes(showscale=False)

    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Accuracy values were obtained during model evaluation on the testing dataset.")

    #  INSIGHT 
    st.markdown("---")
    st.subheader("🧠 AI Decision Summary")

    real_count = list(predictions.values()).count("REAL")
    fake_count = list(predictions.values()).count("FAKE")

    final_decision = "REAL" if real_count > fake_count else "FAKE"

    if final_decision == "REAL":
        best_model = max(results, key=lambda x: results[x]["real"])
    else:
        best_model = max(results, key=lambda x: results[x]["fake"])

    confidence_value = max(real_prob, fake_prob)

    if confidence_value > 0.85:
        level = "High 🔥"
    elif confidence_value > 0.65:
        level = "Medium ⚡"
    else:
        level = "Low ⚠️"

    c1,c2,c3=st.columns(3)

    c1.metric("Final Decision",final_decision)

    c2.metric("Best Model",best_model)

    c3.metric("Confidence",level)

    st.success(f"Model Agreement : {real_count} REAL | {fake_count} FAKE")

st.divider()

st.markdown("""
<hr>

<div style="
text-align:center;
color:#8FA6C1;
font-size:15px;
line-height:1.8;
">

© 2026 <b style="color:#42A5F5;">ConflictGuard</b>

<br>

Hybrid AI-Based Conflict News Verification System

<br>

🎓 Developed as a Final Year Academic Project

</div>
""",unsafe_allow_html=True)

