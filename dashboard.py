import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

CARD_FILE = "./Documentation/faraway_data.xlsx"

st.set_page_config(layout="wide")
st.title("🌍 Faraway – Analyse & Équilibrage des cartes")

# --- Emojis ressources ---
SYMBOL_EMOJIS = {
    "stone": "🪨",
    "night": "🌙",
    "map": "🗺️",
    "chimera": "🐉",
    "day": "☀️",
    "thistle": "🌿"
}

def format_symbol(symbol):
    return f"{SYMBOL_EMOJIS.get(symbol, '')} {symbol}"

# --- Chargement robuste ---
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)

    df["symbols"] = df["symbols"].fillna("").astype(str)

    df["symbols_list"] = df["symbols"].apply(
        lambda x: [s.strip() for s in x.split(",") if s.strip()]
    )

    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["multiplicator"] = pd.to_numeric(df["multiplicator"], errors="coerce")

    df["num_symbols"] = df["symbols_list"].apply(len)

    return df

df = load_data(CARD_FILE)

# --- Sidebar filtres ---
st.sidebar.header("🎛️ Filtres")

colors = st.sidebar.multiselect(
    "Couleurs",
    options=df["color"].dropna().unique(),
    default=df["color"].dropna().unique()
)

types = st.sidebar.multiselect(
    "Types",
    options=df["type"].dropna().unique(),
    default=df["type"].dropna().unique()
)

df = df[df["color"].isin(colors) & df["type"].isin(types)]

# --- KPIs ---
st.subheader("📌 Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Nb cartes", len(df))
col2.metric("Points moyens", round(df["points"].mean(), 2))
col3.metric("Nb ressources uniques", len(set(df["symbols_list"].explode())))
col4.metric("Complexité moyenne", round(df["num_symbols"].mean(), 2))

# --- Ressources ---
st.subheader("🧩 Ressources")

symbols = df["symbols_list"].explode()
symbol_counts = Counter(symbols)

sym_df = pd.DataFrame(symbol_counts.items(), columns=["Ressource", "Count"])
sym_df = sym_df.sort_values(by="Count", ascending=False)
sym_df["Ressource"] = sym_df["Ressource"].apply(format_symbol)

st.dataframe(sym_df, use_container_width=True)

fig1, ax1 = plt.subplots()
ax1.bar(sym_df["Ressource"], sym_df["Count"])
plt.xticks(rotation=45)
st.pyplot(fig1)

# --- Rareté ---
st.subheader("💎 Rareté des ressources")

total_symbols = len(symbols)

symbol_freq = {k: v / total_symbols for k, v in symbol_counts.items()}
symbol_rarity = {k: (1 / v if v > 0 else 0) for k, v in symbol_freq.items()}

# Normalisation
max_rarity = max(symbol_rarity.values())
symbol_rarity = {k: v / max_rarity for k, v in symbol_rarity.items()}

rarity_df = pd.DataFrame({
    "Ressource": list(symbol_rarity.keys()),
    "Rareté": list(symbol_rarity.values())
}).sort_values(by="Rareté", ascending=False)

rarity_df["Ressource"] = rarity_df["Ressource"].apply(format_symbol)

st.dataframe(rarity_df, use_container_width=True)

# --- Score rareté par carte ---
def compute_rarity_score(symbols_list):
    if not symbols_list:
        return 0
    return sum(symbol_rarity.get(s, 0) for s in symbols_list)

df["rarity_score"] = df["symbols_list"].apply(compute_rarity_score)

# --- Score puissance ---
df["power_score"] = df["points"] / (df["num_symbols"] + 1)

# --- Score ajusté ---
df["adjusted_power"] = df["points"] * (1 + df["rarity_score"]) / (df["num_symbols"] + 1)

# --- Distribution points ---
st.subheader("🎯 Distribution des points")

fig2, ax2 = plt.subplots()
df["points"].dropna().hist(ax=ax2)
st.pyplot(fig2)

# --- Points par couleur ---
st.subheader("🎨 Points par couleur")

fig3, ax3 = plt.subplots()
df.boxplot(column="points", by="color", ax=ax3)
st.pyplot(fig3)

# --- Complexité ---
st.subheader("⚙️ Complexité vs Points")

fig4, ax4 = plt.subplots()
ax4.scatter(df["num_symbols"], df["points"])
ax4.set_xlabel("Nb ressources")
ax4.set_ylabel("Points")
st.pyplot(fig4)

# --- Puissance brute ---
st.subheader("🔥 Cartes les plus fortes (brut)")

st.dataframe(
    df.sort_values(by="power_score", ascending=False)
    .head(10)[["id", "points", "num_symbols", "power_score"]]
)

# --- Puissance ajustée ---
st.subheader("🏆 Cartes optimisées (rareté incluse)")

st.dataframe(
    df.sort_values(by="adjusted_power", ascending=False)
    .head(10)[["id", "points", "rarity_score", "adjusted_power"]]
)

# --- Cartes faibles ---
st.subheader("📉 Cartes sous-évaluées")

st.dataframe(
    df.sort_values(by="adjusted_power")
    .head(10)[["id", "points", "rarity_score", "adjusted_power"]]
)

# --- Ressources vs performance ---
st.subheader("📈 Ressources vs performance")

df_exploded = df.explode("symbols_list")
res_perf = df_exploded.groupby("symbols_list")["points"].mean().sort_values()

res_perf.index = [format_symbol(x) for x in res_perf.index]

st.dataframe(res_perf)

fig5, ax5 = plt.subplots()
res_perf.plot(kind="bar", ax=ax5)
plt.xticks(rotation=45)
st.pyplot(fig5)

# --- Points vs rareté ---
st.subheader("📊 Points vs Rareté")

fig6, ax6 = plt.subplots()
ax6.scatter(df["rarity_score"], df["points"])
ax6.set_xlabel("Rareté")
ax6.set_ylabel("Points")
st.pyplot(fig6)

# --- Anomalies ---
st.subheader("⚠️ Anomalies")

high = df["points"].mean() + 2 * df["points"].std()
low = df["points"].mean() - 2 * df["points"].std()

st.write("Cartes trop fortes")
st.dataframe(df[df["points"] > high])

st.write("Cartes trop faibles")
st.dataframe(df[df["points"] < low])

# --- Données ---
st.subheader("📄 Données brutes")
st.dataframe(df)