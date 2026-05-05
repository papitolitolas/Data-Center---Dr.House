# 🌳 Trabalho Prático — Árvores de Decisão
**Cadeira de Exploração de Dados · ISCIM**

---

## ▶️ Abrir no Google Colab (sem instalação)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_UTILIZADOR/trabalho-pratico-dt/blob/main/trabalho_dt.ipynb)

> **Substitua `SEU_UTILIZADOR` pelo seu nome de utilizador do GitHub.**

---

## 📋 Sobre o Projeto

Este notebook implementa uma pipeline completa de **Árvores de Decisão (Decision Trees)**
para problemas de classificação, utilizando `scikit-learn`, `pandas`, `matplotlib` e `seaborn`.

---

## ⚙️ Para o Docente — Como Trocar o Dataset

Abra o notebook e edite **apenas a Célula 2**:

```python
# ═══════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURAÇÃO DO DATASET — O DOCENTE ALTERA APENAS ESTA SECÇÃO
# ═══════════════════════════════════════════════════════════════════

DATASET = "iris"   # ← altere aqui: "iris", "wine", "breast_cancer" ou "csv"

# Se usar CSV externo:
CAMINHO_CSV  = "dados.csv"
COLUNA_ALVO  = "target"    # nome da coluna a prever

# Parâmetros da árvore:
MAX_DEPTH    = 4
TEST_SIZE    = 0.25
RANDOM_STATE = 42
CRITERION    = "gini"      # "gini" ou "entropy"
```

Toda a pipeline adapta-se automaticamente ao novo dataset.

---

## 📦 Datasets Disponíveis

| Valor de `DATASET` | Descrição | Nº Classes |
|---|---|---|
| `"iris"` | Espécies de flores (sklearn) | 3 |
| `"wine"` | Tipos de vinho (sklearn) | 3 |
| `"breast_cancer"` | Diagnóstico de cancro (sklearn) | 2 |
| `"csv"` | Ficheiro CSV externo | variável |

Para usar um CSV próprio, coloque o ficheiro na mesma pasta que o notebook e defina `CAMINHO_CSV` e `COLUNA_ALVO`.

---

## 🗂️ Estrutura do Projeto

```
trabalho-pratico-dt/
├── trabalho_dt.ipynb       ← Notebook principal
├── requirements.txt        ← Dependências Python
├── dados.csv               ← Dataset externo (opcional)
├── index.html              ← Página GitHub Pages
└── README.md               ← Este ficheiro
```

---

## 💻 Instalação Local

```bash
pip install -r requirements.txt
jupyter notebook trabalho_dt.ipynb
```

---

## 📚 Conteúdo do Notebook

| Célula | Conteúdo |
|--------|----------|
| 1 | Importações de bibliotecas |
| 2 | ⚙️ **Configuração do dataset** (editar aqui) |
| 3 | Carregamento e preparação dos dados |
| 4 | Análise Exploratória (EDA) |
| 5 | Divisão treino/teste e treino do modelo |
| 6 | Avaliação: Accuracy, F1, Matriz de Confusão |
| 7 | Visualização gráfica da Árvore |
| 8 | Importância das features |
| 9 | *(Facultativo)* Análise de overfitting por profundidade |
| 10 | *(Facultativo)* Otimização com GridSearchCV |
