import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import pandas as pd
import time
import sqlite3
import matplotlib.pyplot as plt

# Loading environment variables
load_dotenv()

# Configuring Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Database setup
DB_FILE = "medical_agent.db"
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT,
    response TEXT,
    latency REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Streamlit App Config
st.set_page_config(page_title="AI Medical Diagnostic Agent", layout="wide")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
model_name = st.sidebar.selectbox("Choose Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
temperature = st.sidebar.slider("Creativity (Temperature)", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("Max Tokens", 100, 1000, 400, 50)

# Tabs for app sections
tabs = st.tabs(["Diagnostic Chat", "File Upload", "Evaluation Dashboard", "Database Viewer", "Admin Analytics", "Test Report Generator"])

# --- Disclaimer ---
st.sidebar.warning("This tool is for **educational purposes only** and should not be used as professional medical advice.")

# --- Diagnostic Chat Tab ---
with tabs[0]:
    st.title("AI Medical Diagnostic Agent")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Describe your symptoms or medical query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        c.execute("INSERT INTO interactions (role, content) VALUES (?, ?)", ("user", prompt))
        conn.commit()
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
            reply = response.text
        except Exception as e:
            reply = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        c.execute("INSERT INTO interactions (role, content) VALUES (?, ?)", ("assistant", reply))
        conn.commit()
        with st.chat_message("assistant"):
            st.markdown(reply)

# --- File Upload Tab ---
with tabs[1]:
    st.title("Upload Medical Reports")
    uploaded_file = st.file_uploader("Upload a medical report (PDF/TXT)", type=["pdf", "txt"])

    if uploaded_file:
        text = ""
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
            text = uploaded_file.read().decode("utf-8")

        st.subheader("Extracted Report Content:")
        st.text_area("", text, height=200)

        if st.button("Analyze Report with AI"):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    f"Analyze the following medical report and provide structured diagnostic insights (non-medical advice). Include Summary, Risk Factors, and Possible Next Steps:\n{text}",
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
                )
                st.subheader("AI Analysis Result:")
                st.write(response.text)

                # Save to DB
                c.execute("INSERT INTO interactions (role, content) VALUES (?, ?)", ("report_analysis", response.text))
                conn.commit()
            except Exception as e:
                st.error(f" Error: {e}")

# --- Evaluation Dashboard Tab ---
with tabs[2]:
    st.title("Evaluation Dashboard")
    st.write("Run benchmark prompts to test AI accuracy, response time, and consistency.")

    sample_prompts = [
        "What are the possible causes of persistent cough?",
        "Explain the difference between type 1 and type 2 diabetes.",
        "List common side effects of ibuprofen.",
        "Summarize symptoms of hypertension in under 50 words.",
    ]

    if st.button("Run Evaluation"):
        results = []
        model = genai.GenerativeModel(model_name)
        for q in sample_prompts:
            start_time = time.time()
            try:
                response = model.generate_content(q, generation_config={"temperature": 0.2, "max_output_tokens": max_tokens})
                latency = round(time.time() - start_time, 2)
                results.append((q, response.text, latency))

                # Save to DB
                c.execute("INSERT INTO evaluations (prompt, response, latency) VALUES (?, ?, ?)", (q, response.text, latency))
                conn.commit()
            except Exception as e:
                results.append((q, f"Error: {e}", None))

        df = pd.DataFrame(results, columns=["Prompt", "AI Response", "Response Time (s)"])
        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="Download Results (CSV)",
            data=df.to_csv(index=False),
            file_name="evaluation_results.csv",
            mime="text/csv"
        )

# --- Database Viewer Tab ---
with tabs[3]:
    st.title("Database Viewer")

    view_option = st.radio("Select data to view:", ["Interactions", "Evaluations"])

    if view_option == "Interactions":
        df = pd.read_sql_query("SELECT * FROM interactions ORDER BY timestamp DESC", conn)
        st.dataframe(df, use_container_width=True)
    else:
        df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY timestamp DESC", conn)
        st.dataframe(df, use_container_width=True)

# --- Admin Analytics Tab ---
with tabs[4]:
    st.title("Admin Analytics")
    st.write("Overview of system usage and performance trends.")

    # Load data
    df_interactions = pd.read_sql_query("SELECT * FROM interactions", conn)
    df_evaluations = pd.read_sql_query("SELECT * FROM evaluations", conn)

    # Chart 1: Interaction counts over time
    if not df_interactions.empty:
        st.subheader("User Interactions Over Time")
        df_interactions["date"] = pd.to_datetime(df_interactions["timestamp"]).dt.date
        daily_counts = df_interactions.groupby("date").size()
        fig, ax = plt.subplots()
        daily_counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("# of Interactions")
        st.pyplot(fig)

    # Chart 2: Evaluation latency distribution
    if not df_evaluations.empty:
        st.subheader("Evaluation Response Times")
        fig, ax = plt.subplots()
        df_evaluations["latency"].dropna().plot(kind="hist", bins=10, ax=ax)
        ax.set_xlabel("Response Time (s)")
        st.pyplot(fig)

    # Chart 3: Most common queries (top 5)
    if not df_interactions.empty:
        st.subheader("Most Common Queries (Top 5)")
        user_queries = df_interactions[df_interactions["role"] == "user"]["content"]
        top_queries = user_queries.value_counts().head(5)
        st.table(top_queries)

# --- Test Report Generator Tab ---
with tabs[5]:
    st.title("Test Report Generator")
    st.write("Generate a structured report from evaluation results.")

    df_eval = pd.read_sql_query("SELECT * FROM evaluations ORDER BY timestamp DESC", conn)

    if df_eval.empty:
        st.info("No evaluation data available. Please run evaluations first.")
    else:
        summary = {
            "Total Evaluations": len(df_eval),
            "Average Latency (s)": round(df_eval["latency"].dropna().mean(), 2),
            "Fastest Response (s)": round(df_eval["latency"].dropna().min(), 2),
            "Slowest Response (s)": round(df_eval["latency"].dropna().max(), 2)
        }

        st.subheader("Report Summary")
        st.json(summary)

        st.subheader("Detailed Results")
        st.dataframe(df_eval, use_container_width=True)

        st.download_button(
            label="Download Test Report (CSV)",
            data=df_eval.to_csv(index=False),
            file_name="test_report.csv",
            mime="text/csv"
        )

# ---- requirements.txt (create this file alongside app.py) ----
# streamlit
# google-generativeai
# python-dotenv
# PyPDF2
# pandas
# matplotlib
