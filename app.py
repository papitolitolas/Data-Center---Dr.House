# ═══════════════════════════════════════════════════════════════════
#  app.py — Trabalho Prático: Árvores de Decisão
#  Cadeira de Exploração de Dados · ISCIM
#  Executar: streamlit run app.py
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay
)

# ── Configuração da página ─────────────────────────────────────────
st.set_page_config(
    page_title="Árvores de Decisão · ISCIM",
    page_icon="🌳",
    layout="wide"
)

# ── Estilo personalizado ───────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f0e0d; }
    [data-testid="stSidebar"] * { color: #f5f0e8 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #d4a44c !important; font-weight: 600; }
    h1 { font-size: 2rem !important; }
    .metric-box {
        background: #f5f0e8;
        border: 1px solid #d0c8b8;
        border-left: 4px solid #c0392b;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        text-align: center;
    }
    .metric-box .val { font-size: 2rem; font-weight: 700; color: #0f0e0d; }
    .metric-box .lbl { font-size: .75rem; color: #6b6560; letter-spacing: .1em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def carregar_sklearn(nome):
    """Carrega datasets do scikit-learn."""
    loaders = {
        "Iris":          datasets.load_iris,
        "Wine":          datasets.load_wine,
        "Breast Cancer": datasets.load_breast_cancer,
    }
    ds = loaders[nome]()
    features = ds.feature_names.tolist() if hasattr(ds.feature_names, 'tolist') else list(ds.feature_names)
    return ds.data, ds.target, features, ds.target_names.tolist()


def carregar_csv(ficheiro, coluna_alvo):
    """Carrega um CSV enviado pelo utilizador."""
    df = pd.read_csv(ficheiro)
    if coluna_alvo not in df.columns:
        st.error(f"Coluna '{coluna_alvo}' não encontrada. Colunas disponíveis: {list(df.columns)}")
        st.stop()
    X_raw = df.drop(columns=[coluna_alvo])
    y_raw = df[coluna_alvo]
    for col in X_raw.select_dtypes(include="object").columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col])
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    X_clean = X_raw.fillna(X_raw.median(numeric_only=True))
    return X_clean.values, y_enc, X_raw.columns.tolist(), le.classes_.tolist()


# ══════════════════════════════════════════════════════════════════
#  BARRA LATERAL — CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuração")
    st.markdown("---")

    # --- Escolha do Dataset ---
    st.markdown("### 📂 Dataset")
    fonte = st.selectbox(
        "Fonte dos dados",
        ["Iris", "Wine", "Breast Cancer", "CSV personalizado"],
        help="Escolha um dataset pré-carregado ou faça upload do seu CSV."
    )

    csv_file    = None
    coluna_alvo = "target"

    if fonte == "CSV personalizado":
        csv_file = st.file_uploader("Carregar ficheiro CSV", type=["csv"])
        if csv_file:
            df_preview = pd.read_csv(csv_file)
            csv_file.seek(0)
            coluna_alvo = st.selectbox("Coluna alvo (variável a prever)", df_preview.columns.tolist())

    st.markdown("---")

    # --- Parâmetros da Árvore ---
    st.markdown("### 🌳 Parâmetros")
    max_depth    = st.slider("Profundidade máxima (max_depth)", 1, 15, 4)
    criterion    = st.selectbox("Critério", ["gini", "entropy"])
    test_size    = st.slider("Proporção de teste", 0.10, 0.40, 0.25, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    st.markdown("---")
    st.markdown("### 🔬 Extras")
    mostrar_cv          = st.checkbox("Validação cruzada (5-fold)", value=True)
    mostrar_overfitting = st.checkbox("Análise de overfitting", value=False)

    st.markdown("---")
    st.caption("ISCIM · Exploração de Dados")


# ══════════════════════════════════════════════════════════════════
#  CARREGAR DADOS
# ══════════════════════════════════════════════════════════════════

if fonte == "CSV personalizado":
    if csv_file is None:
        st.info("👈 Carregue um ficheiro CSV na barra lateral para começar.")
        st.stop()
    X, y, FEATURES, CLASSES = carregar_csv(csv_file, coluna_alvo)
else:
    X, y, FEATURES, CLASSES = carregar_sklearn(fonte)


# ══════════════════════════════════════════════════════════════════
#  CABEÇALHO
# ══════════════════════════════════════════════════════════════════

st.title("🌳 Trabalho Prático — Árvores de Decisão")
st.markdown(f"**Dataset:** `{fonte}`  |  **Amostras:** `{X.shape[0]}`  |  **Features:** `{X.shape[1]}`  |  **Classes:** `{len(CLASSES)}`")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploração", "🤖 Modelo", "🌳 Árvore", "📌 Features", "📋 Dados"
])


# ─────────────────────────────────────────────────────────────────
#  TAB 1 — Análise Exploratória
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Análise Exploratória dos Dados")

    df = pd.DataFrame(X, columns=FEATURES)
    df["classe"] = [CLASSES[i] for i in y]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribuição das Classes**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        contagem = df["classe"].value_counts()
        ax.bar(contagem.index, contagem.values,
               color=sns.color_palette("muted", len(contagem)))
        ax.set_xlabel("Classe")
        ax.set_ylabel("Contagem")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Correlação entre Features**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        corr = pd.DataFrame(X, columns=FEATURES).corr()
        sns.heatmap(corr, ax=ax, annot=len(FEATURES) <= 10,
                    fmt=".1f", cmap="coolwarm", linewidths=.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Estatísticas Descritivas**")
    st.dataframe(df.describe().round(3), use_container_width=True)


# ─────────────────────────────────────────────────────────────────
#  TREINO DO MODELO (partilhado entre tabs)
# ─────────────────────────────────────────────────────────────────

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=test_size,
    stratify=y,
    random_state=int(random_state)
)

modelo = DecisionTreeClassifier(
    max_depth=max_depth,
    criterion=criterion,
    random_state=int(random_state)
)
modelo.fit(X_treino, y_treino)
y_pred = modelo.predict(X_teste)
acc    = accuracy_score(y_teste, y_pred)


# ─────────────────────────────────────────────────────────────────
#  TAB 2 — Avaliação do Modelo
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Avaliação do Modelo")

    # Métricas rápidas
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy (Teste)", f"{acc:.4f}")
    m2.metric("Amostras Treino",  len(X_treino))
    m3.metric("Amostras Teste",   len(X_teste))

    # Validação cruzada
    if mostrar_cv:
        scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
        st.info(f"**Validação Cruzada (5-fold):** {scores.mean():.3f} ± {scores.std():.3f}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Relatório de Classificação**")
        report = classification_report(y_teste, y_pred, target_names=CLASSES, output_dict=True)
        df_report = pd.DataFrame(report).T.round(3)
        st.dataframe(df_report, use_container_width=True)

    with col2:
        st.markdown("**Matriz de Confusão**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_teste, y_pred,
            display_labels=CLASSES,
            cmap="Blues",
            ax=ax
        )
        ax.set_title("Matriz de Confusão")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Análise de overfitting
    if mostrar_overfitting:
        st.markdown("---")
        st.markdown("**Análise de Overfitting por Profundidade**")
        profundidades = range(1, 16)
        acc_tr, acc_te = [], []
        for d in profundidades:
            m = DecisionTreeClassifier(max_depth=d, random_state=int(random_state))
            m.fit(X_treino, y_treino)
            acc_tr.append(accuracy_score(y_treino, m.predict(X_treino)))
            acc_te.append(accuracy_score(y_teste,  m.predict(X_teste)))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(profundidades, acc_tr, marker="o", label="Treino")
        ax.plot(profundidades, acc_te, marker="s", label="Teste")
        ax.axvline(max_depth, color="red", ls="--", lw=1.5, label=f"Seleccionado (d={max_depth})")
        ax.set_xlabel("Profundidade")
        ax.set_ylabel("Accuracy")
        ax.set_title("Overfitting vs Underfitting")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────
#  TAB 3 — Visualização da Árvore
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Visualização da Árvore de Decisão")

    largura = max(14, max_depth * 4)
    altura  = max(6, max_depth * 2)

    fig, ax = plt.subplots(figsize=(largura, altura))
    plot_tree(
        modelo,
        feature_names=FEATURES,
        class_names=CLASSES,
        filled=True,
        rounded=True,
        fontsize=max(7, 10 - max_depth),
        ax=ax
    )
    ax.set_title(f"Árvore de Decisão — {fonte}  (profundidade={max_depth}, critério={criterion})",
                 fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Download da imagem
    import io
    buf = io.BytesIO()
    fig2, ax2 = plt.subplots(figsize=(largura, altura))
    plot_tree(modelo, feature_names=FEATURES, class_names=CLASSES,
              filled=True, rounded=True, fontsize=max(7, 10 - max_depth), ax=ax2)
    ax2.set_title(f"Árvore de Decisão — {fonte}")
    plt.tight_layout()
    fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    st.download_button("⬇️ Descarregar Árvore (PNG)", buf, "arvore_decisao.png", "image/png")

    st.markdown("---")
    st.markdown("**Regras de Decisão (texto)**")
    regras = export_text(modelo, feature_names=FEATURES)
    st.code(regras, language=None)


# ─────────────────────────────────────────────────────────────────
#  TAB 4 — Importância das Features
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Importância das Features")

    importancias = pd.Series(
        modelo.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, max(3, len(FEATURES) * 0.4)))
        cores = sns.color_palette("muted", len(importancias))
        importancias.plot(kind="barh", ax=ax, color=cores)
        ax.set_title("Importância das Features (Gini)")
        ax.set_xlabel("Importância relativa")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Tabela de Importâncias**")
        df_imp = importancias.reset_index()
        df_imp.columns = ["Feature", "Importância"]
        df_imp["Importância"] = df_imp["Importância"].round(4)
        st.dataframe(df_imp, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 5 — Dados Brutos
# ─────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Dados Brutos")
    df_show = pd.DataFrame(X, columns=FEATURES)
    df_show["classe"] = [CLASSES[i] for i in y]
    st.dataframe(df_show, use_container_width=True)

    csv_bytes = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descarregar CSV", csv_bytes, "dados_exportados.csv", "text/csv")
