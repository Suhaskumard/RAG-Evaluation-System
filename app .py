import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("📊 RAG Evaluation Dashboard")

try:
    records=[]
    with open("reports/failures.jsonl","r") as f:
        for line in f:
            records.append(json.loads(line))

    if records:
        df = pd.DataFrame([
            {
                "Query": r["query"],
                "Faithfulness": r["metrics"]["Faithfulness"],
                "Stability": r["metrics"]["Stability"]
            } for r in records
        ])
        st.dataframe(df)
        with st.expander("View Full Logs"):
            st.json(records)
    else:
        st.success("No failure cases logged")

except:
    st.info("Run main.py first")
    st.caption("Faithfulness < 0.6 indicates potential hallucination risk")

