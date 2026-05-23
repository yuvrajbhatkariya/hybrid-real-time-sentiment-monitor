import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time
import os

st.set_page_config(
    page_title="Brand Sentiment Monitor",
    layout="wide",
    page_icon="📊"
)

OUTPUT_DIR = "/home/master/project/output/"

def load_all_batches():
    files = glob.glob(OUTPUT_DIR + "batch_*.csv")
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_drift():
    path = OUTPUT_DIR + "drift_log.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

st.title("Real-Time Brand Sentiment Monitor")
st.caption("Kafka → Spark Streaming → VADER → HDFS")
st.divider()

placeholder = st.empty()

while True:
    with placeholder.container():
        df   = load_all_batches()
        d_df = load_drift()

        if df.empty:
            st.warning("Waiting for streaming data...")
            time.sleep(10)
            continue

        total = len(df)
        good  = (df['predicted_label'] == 'good').sum()
        poor  = (df['predicted_label'] == 'poor').sum()
        ok    = (df['predicted_label'] == 'ok').sum()

        # ── KPI cards ─────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Processed", f"{total:,}")
        c2.metric("Good",   f"{good}",
                  f"{good/total*100:.1f}%")
        c3.metric("Poor",   f"{poor}",
                  f"{poor/total*100:.1f}%")
        c4.metric("Ok",     f"{ok}",
                  f"{ok/total*100:.1f}%")

        st.divider()
        col1, col2 = st.columns(2)

        # ── Pie chart ─────────────────────────────────────────────
        with col1:
            st.subheader("Overall Sentiment")
            counts = df['predicted_label'].value_counts()
            colors = {'good':'#4CAF50','poor':'#F44336','ok':'#FFC107'}
            c = [colors.get(l,'#999') for l in counts.index]
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index,
                   autopct='%1.1f%%', colors=c)
            st.pyplot(fig)
            plt.close()

        # ── Brand comparison ──────────────────────────────────────
        with col2:
            st.subheader("Brand Comparison")
            top_brands = df['brand'].value_counts().head(6).index
            b_df  = df[df['brand'].isin(top_brands)]
            pivot = b_df.groupby(
                ['brand','predicted_label']
            ).size().unstack(fill_value=0)

            fig2, ax2 = plt.subplots(figsize=(8,4))
            pivot.plot(kind='bar', ax=ax2,
                       color=['#4CAF50','#FFC107','#F44336'])
            ax2.set_xticklabels(
                ax2.get_xticklabels(), rotation=30
            )
            st.pyplot(fig2)
            plt.close()

        st.divider()

        # ── Drift alerts ──────────────────────────────────────────
        st.subheader("Drift Monitor")
        if not d_df.empty:
            alerts = d_df[d_df['drift_flag'] == 'DRIFT_ALERT']
            col_a, col_b = st.columns(2)
            col_a.metric("Drift Alerts",   len(alerts))
            col_b.metric("Stable Brands",
                         d_df[d_df['drift_flag']=='STABLE']['brand'].nunique())

            if not alerts.empty:
                st.error("⚠️ Sentiment drift detected!")
                st.dataframe(
                    alerts[['brand','batch_id','good_ratio',
                             'poor_ratio','drift_delta']].tail(10),
                    use_container_width=True
                )
            else:
                st.success("✅ All brands stable")

        st.divider()

        # ── Live feed ─────────────────────────────────────────────
        st.subheader("Latest Reviews")
        st.dataframe(
            df[['brand','review_text','true_label',
                'predicted_label']].tail(20),
            use_container_width=True
        )

    time.sleep(15)
    placeholder.empty()