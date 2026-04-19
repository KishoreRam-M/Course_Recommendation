import streamlit as st
import pandas as pd
from src.inference import CourseRecommenderEngine

st.set_page_config(
    page_title="Course AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #e2e8f0;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .premium-title {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    div.css-1r6slb0, div.css-12oz5g7 {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #38bdf8;
    }
    .stButton>button {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(168, 85, 247, 0.3);
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    engine = CourseRecommenderEngine()
    engine.load_artifacts()
    return engine

engine = get_engine()

if not engine.loaded:
    st.error("Failed to load models or dataset. Please ensure `python run_pipeline.py` ran successfully.")
    st.stop()

with st.sidebar:
    st.markdown("### 🎓 Course AI Settings")
    st.info(f"**Model:** `{engine.meta.get('best_model_name', 'Unknown')}`")
    st.metric("Total Courses", f"{len(engine.df):,}")
    st.metric("Total Categories", len(engine.meta.get('classes', [])))
    st.metric("TF-IDF Features", f"{engine.meta.get('n_features', 0):,}")
    st.markdown("---")
    st.markdown("Model Pipeline built on Kaggle Coursera Dataset. Deployed via **Streamlit** for premium UI experience.")

st.markdown("<div class='premium-title'>Coursera AI Assistant</div>", unsafe_allow_html=True)
st.markdown("Welcome to the advanced recommendation and classification engine. Use the tabs below to explore.")

tab1, tab2, tab3 = st.tabs(["🔍 Category Classifier", "🎯 AI Recommender", "📊 Dataset Explorer"])

with tab1:
    st.markdown("### Predict the best category for any course or description")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        text_input = st.text_area("📝 Enter Course Description or Skills", height=150, 
                                  placeholder="e.g., 'Learn Python for data analysis, pandas, matplotlib, machine learning'")
        
        classify_btn = st.button("🔍 Classify Text", key="btn_cls", use_container_width=True)
        
        st.markdown("**Quick Examples:**")
        examples = [
            "Anatomy, physiology, patient care, clinical diagnosis",
            "Financial markets, portfolio management, risk",
            "Photography, video editing, visual arts"
        ]
        for ex in examples:
            st.markdown(f"> *\"{ex}\"*")

    with col2:
        if classify_btn:
            if not text_input.strip():
                st.warning("⚠️ Please enter some text to classify.")
            else:
                with st.spinner("Classifying..."):
                    res_df = engine.classify(text_input)
                    top_cat = res_df.iloc[0]['Category']
                    top_conf = res_df.iloc[0]['Confidence']
                    
                    st.success(f"**Predicted:** {top_cat} ({top_conf:.1f}%)")
                    st.markdown("#### Confidence Scores")
                    st.dataframe(
                        res_df.head(5).style.format({"Confidence": "{:.1f}%"}).background_gradient(cmap="Purples"),
                        use_container_width=True
                    )

with tab2:
    st.markdown("### Find the most similar courses using AI Embeddings")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        rec_input = st.text_input("🔎 What do you want to learn?", 
                                  placeholder="e.g. 'I want to learn artificial intelligence and neural networks'")
    with col2:
        n_results = st.slider("Number of results", min_value=3, max_value=15, value=5)
        
    rec_btn = st.button("🎯 Find Courses", key="btn_rec", use_container_width=True)

    if rec_btn:
        if not rec_input.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("Searching..."):
                results = engine.recommend(rec_input, n_results=n_results)
                st.markdown(f"##### Top {len(results)} Results for your query")
                
                for r in results:
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color:#1e293b; padding:15px; border-radius:10px; margin-bottom:10px; border-left:4px solid #3b82f6;">
                            <h4 style="margin:0; color:#38bdf8;">
                                <a href="{r['URL']}" target="_blank" style="color:#38bdf8; text-decoration:none;">{r['Course Name']}</a>
                            </h4>
                            <p style="margin:5px 0 0 0; font-size:14px; color:#cbd5e1;">
                                <span style="background:#334155; padding:2px 6px; border-radius:4px; font-weight:600;">{r['Category']}</span>
                                &nbsp;&bull;&nbsp; Match: <b>{r['Similarity']}</b>
                            </p>
                            <p style="margin:5px 0 0 0; font-size:13px; color:#94a3b8;">
                                <i>Skills: {r['Skills']}...</i>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Browse the complete Coursera dataset")
    df_courses = engine.df
    all_cats = ["All"] + sorted(df_courses['category'].unique().tolist())
    selected_cat = st.selectbox("📂 Filter by Category", all_cats)
    
    if selected_cat == "All":
        filtered_df = df_courses.copy()
    else:
        filtered_df = df_courses[df_courses['category'] == selected_cat]
        
    st.markdown(f"**Showing {len(filtered_df):,} courses**")
    
    display_df = filtered_df[['name', 'category', 'skills', 'language', 'url']]
    st.dataframe(display_df, use_container_width=True, height=500)
    
    st.markdown("---")
    st.markdown("### Model Leaderboard (Reference)")
    leaderboard = pd.DataFrame(engine.meta.get('leaderboard', [])).sort_values('F1-Score (weighted)', ascending=False)
    leaderboard.index = leaderboard.index + 1
    st.dataframe(leaderboard.style.highlight_max(subset=['F1-Score (weighted)'], color='#065f46'))
