# 🌱 SAD-ADUBO — Sistema Inteligente de Apoio à Decisão para Recomendação de Adubação

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

> **TCC — Bacharelado em Sistemas de Informação | UEG — UnU Santa Helena de Goiás | 2026**
> 
> Autor: Eduardo Augusto de Oliveira Mendes | Orientadora: Prof.ª Me. Pollyana Queiroz

---

## 📋 Sobre o Projeto

O **SAD-ADUBO** é um sistema web que recebe dados físico-químicos de solo, variáveis climáticas e histórico de manejo agrícola, e retorna a recomendação do fertilizante mais adequado — acompanhada de uma **explicação interpretável (SHAP)** acessível ao produtor rural.

O MVP valida a metodologia usando o dataset público [Fertilizer Recommendation Dataset](https://www.kaggle.com/datasets/miadul/fertilizer-recommendation-dataset/data) (Kaggle, 10.000 registros, 19 features).

---

## 🚀 Funcionalidades

- **🌱 Recomendação:** Inserção de dados de solo, clima e manejo → recomendação de fertilizante com % de confiança
- **📊 Análise do Dataset:** Distribuição de classes, heatmap de correlação, boxplots interativos
- **🤖 Desempenho:** Comparação de 4 modelos (Decision Tree, Random Forest, SVM, XGBoost) com métricas detalhadas
- **🔍 Explicabilidade:** Gráfico SHAP waterfall + texto interpretativo em português para cada predição
- **📥 Exportação:** Download do resultado em CSV

---

## 🛠️ Tecnologias

| Camada | Tecnologia |
|---|---|
| Interface | Streamlit |
| ML / Backend | Python, scikit-learn, XGBoost |
| Explicabilidade | SHAP (TreeExplainer) |
| Visualização | Plotly, Matplotlib, Seaborn |
| Dados | Pandas, NumPy |

---

## 📦 Instalação Local

### Pré-requisitos

- Python 3.10 ou superior
- pip (gerenciador de pacotes)

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/tcc-ueg.git
cd tcc-ueg

# 2. Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Treinar os modelos (primeira vez)
python src/train.py

# 5. Executar a aplicação
streamlit run app.py
```

A aplicação estará disponível em: `http://localhost:8501`

---

## 📁 Estrutura de Pastas

```
tcc-ueg/
├── data/
│   └── fertilizer_recommendation.csv   # Dataset bruto (Kaggle)
├── notebooks/
│   └── eda.ipynb                        # Análise exploratória
├── models/
│   ├── best_model.pkl                   # Modelo serializado (melhor F1)
│   ├── scaler.pkl                       # MinMaxScaler treinado
│   ├── encoders.pkl                     # Dict de LabelEncoders
│   ├── metrics.json                     # Métricas de todos os modelos
│   ├── confusion_matrices.pkl           # Matrizes de confusão
│   ├── shap_global_values.pkl           # SHAP global pré-calculado
│   ├── feature_names.pkl                # Nomes das features
│   └── label_classes.pkl                # Classes da variável-alvo
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Pipeline de encoding + scaling
│   ├── train.py                         # Treinamento e avaliação
│   ├── predict.py                       # Inferência em tempo real
│   └── explainer.py                     # SHAP + texto interpretativo
├── app.py                               # Ponto de entrada do Streamlit
├── requirements.txt
├── .gitignore
├── GEMINI.md                            # Especificação do projeto
└── README.md
```

---

## 🔬 Metodologia

1. **Pré-processamento:** Label Encoding (categóricas) + MinMaxScaler (numéricas)
2. **Divisão:** 80% treino / 20% teste, com estratificação (`random_state=42`)
3. **Modelos:** Decision Tree, Random Forest, SVM (RBF), XGBoost
4. **Avaliação:** Cross-validation k=10, métrica primária: F1-Score macro
5. **Seleção:** Modelo com maior F1-Score macro no CV
6. **Explicabilidade:** SHAP TreeExplainer (global + local por predição)

---

## ⚠️ Limitações

- Dataset **não regionalizado** para o Brasil (prova de conceito acadêmica)
- **Não fornece dosagem** quantitativa (kg/ha)
- **Não substitui** consultoria agronômica profissional
- Sem integração com APIs climáticas em tempo real

---

## 📄 Licença

Projeto acadêmico — uso educacional.

---

## 👨‍🎓 Créditos

- **Autor:** Eduardo Augusto de Oliveira Mendes
- **Orientadora:** Prof.ª Me. Pollyana Queiroz
- **Instituição:** Universidade Estadual de Goiás (UEG) — UnU Santa Helena de Goiás
- **Curso:** Bacharelado em Sistemas de Informação
- **Ano:** 2026
