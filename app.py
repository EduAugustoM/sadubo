"""
app.py — Interface principal do SAD-ADUBO (Streamlit).

Sistema Inteligente de Apoio à Decisão para Recomendação de Adubação.
Organizado em 4 abas:
1. 🌱 Recomendação — Formulário + predição + SHAP
2. 📊 Análise do Dataset — EDA interativa
3. 🤖 Desempenho dos Modelos — Métricas + comparação
4. ℹ️ Sobre o Sistema — Informações do projeto
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from src.preprocessing import (
    CATEGORICAL_COLS, NUMERICAL_COLS, FEATURE_COLS, TARGET_COL,
    FEATURE_LABELS_PT, VALID_RANGES,
    load_data, get_unique_values, validate_input
)
from src.predict import load_all_artifacts, get_recommendation
from src.explainer import (
    create_explainer, plot_waterfall, plot_summary,
    generate_text_explanation
)

# =============================================================================
# Configuração da Página
# =============================================================================

st.set_page_config(
    page_title="SAD-ADUBO — Recomendação de Adubação",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS customizado
st.markdown("""
<style>
    /* Tema geral */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #2E86AB;
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #666;
        font-size: 0.95rem;
    }
    
    /* Cards de resultado */
    .result-card {
        background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
    }
    .result-card h2 {
        margin: 0;
        font-size: 1.8rem;
    }
    .result-card .confidence {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Seção de formulário */
    .form-section h3 {
        color: #2E86AB;
        border-left: 4px solid #2E86AB;
        padding-left: 10px;
        margin-bottom: 1rem;
    }

    /* Info boxes */
    .info-box {
        background: #f0f7fb;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Tabela de métricas destaque */
    .metric-highlight {
        background: #e8f5e9;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Funções de Cache
# =============================================================================

@st.cache_data
def cached_load_data():
    """Carrega o dataset com cache."""
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'fertilizer_recommendation.csv'
    )
    return load_data(data_path)


@st.cache_resource
def cached_load_artifacts():
    """Carrega artefatos do modelo com cache."""
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models'
    )
    return load_all_artifacts(models_dir)


@st.cache_resource
def cached_create_explainer(_model):
    """Cria SHAP explainer com cache."""
    return create_explainer(_model)


@st.cache_data
def cached_load_metrics():
    """Carrega métricas dos modelos."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models', 'metrics.json'
    )
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def cached_load_confusion_matrices():
    """Carrega matrizes de confusão."""
    cm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models', 'confusion_matrices.pkl'
    )
    return joblib.load(cm_path)


# =============================================================================
# Header
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🌱 SAD-ADUBO</h1>
    <p>Sistema Inteligente de Apoio à Decisão para Recomendação de Adubação</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Inicialização do Session State
# =============================================================================

if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'last_user_input' not in st.session_state:
    st.session_state['last_user_input'] = None


# =============================================================================
# Abas
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌱 Recomendação",
    "📊 Análise do Dataset",
    "🤖 Desempenho dos Modelos",
    "ℹ️ Sobre o Sistema"
])


# =============================================================================
# Aba 1 — 🌱 Recomendação
# =============================================================================

with tab1:
    st.markdown("### Preencha os dados para receber a recomendação de fertilizante")
    st.markdown("---")
    
    # Carregar dados para obter valores únicos dos dropdowns
    df = cached_load_data()
    unique_vals = get_unique_values(df)
    
    # Formulário em 3 colunas
    col_solo, col_clima, col_manejo = st.columns(3)
    
    user_input = {}
    
    # --- Coluna 1: Solo ---
    with col_solo:
        st.markdown("#### 🧪 Dados do Solo")
        
        user_input['Soil_Type'] = st.selectbox(
            FEATURE_LABELS_PT['Soil_Type'],
            options=unique_vals['Soil_Type'],
            key='soil_type'
        )
        user_input['Soil_pH'] = st.number_input(
            FEATURE_LABELS_PT['Soil_pH'],
            min_value=0.0, max_value=14.0,
            value=6.5, step=0.1,
            key='soil_ph'
        )
        user_input['Soil_Moisture'] = st.number_input(
            FEATURE_LABELS_PT['Soil_Moisture'],
            min_value=0.0, max_value=100.0,
            value=40.0, step=0.5,
            key='soil_moisture'
        )
        user_input['Organic_Carbon'] = st.number_input(
            FEATURE_LABELS_PT['Organic_Carbon'],
            min_value=0.0, max_value=10.0,
            value=1.0, step=0.01,
            key='organic_carbon'
        )
        user_input['Electrical_Conductivity'] = st.number_input(
            FEATURE_LABELS_PT['Electrical_Conductivity'],
            min_value=0.0, max_value=10.0,
            value=1.0, step=0.01,
            key='ec'
        )
        user_input['Nitrogen_Level'] = st.number_input(
            FEATURE_LABELS_PT['Nitrogen_Level'],
            min_value=0, max_value=300,
            value=50, step=1,
            key='nitrogen'
        )
        user_input['Phosphorus_Level'] = st.number_input(
            FEATURE_LABELS_PT['Phosphorus_Level'],
            min_value=0, max_value=300,
            value=40, step=1,
            key='phosphorus'
        )
        user_input['Potassium_Level'] = st.number_input(
            FEATURE_LABELS_PT['Potassium_Level'],
            min_value=0, max_value=500,
            value=50, step=1,
            key='potassium'
        )
    
    # --- Coluna 2: Clima ---
    with col_clima:
        st.markdown("#### 🌤️ Dados Climáticos")
        
        user_input['Temperature'] = st.number_input(
            FEATURE_LABELS_PT['Temperature'],
            min_value=-10.0, max_value=55.0,
            value=25.0, step=0.1,
            key='temperature'
        )
        user_input['Humidity'] = st.number_input(
            FEATURE_LABELS_PT['Humidity'],
            min_value=0.0, max_value=100.0,
            value=60.0, step=0.5,
            key='humidity'
        )
        user_input['Rainfall'] = st.number_input(
            FEATURE_LABELS_PT['Rainfall'],
            min_value=0.0, max_value=5000.0,
            value=1000.0, step=10.0,
            key='rainfall'
        )
    
    # --- Coluna 3: Manejo ---
    with col_manejo:
        st.markdown("#### 🚜 Dados de Manejo")
        
        user_input['Crop_Type'] = st.selectbox(
            FEATURE_LABELS_PT['Crop_Type'],
            options=unique_vals['Crop_Type'],
            key='crop_type'
        )
        user_input['Crop_Growth_Stage'] = st.selectbox(
            FEATURE_LABELS_PT['Crop_Growth_Stage'],
            options=unique_vals['Crop_Growth_Stage'],
            key='crop_growth_stage'
        )
        user_input['Season'] = st.selectbox(
            FEATURE_LABELS_PT['Season'],
            options=unique_vals['Season'],
            key='season'
        )
        user_input['Irrigation_Type'] = st.selectbox(
            FEATURE_LABELS_PT['Irrigation_Type'],
            options=unique_vals['Irrigation_Type'],
            key='irrigation_type'
        )
        user_input['Previous_Crop'] = st.selectbox(
            FEATURE_LABELS_PT['Previous_Crop'],
            options=unique_vals['Previous_Crop'],
            key='previous_crop'
        )
        user_input['Region'] = st.selectbox(
            FEATURE_LABELS_PT['Region'],
            options=unique_vals['Region'],
            key='region'
        )
        user_input['Fertilizer_Used_Last_Season'] = st.number_input(
            FEATURE_LABELS_PT['Fertilizer_Used_Last_Season'],
            min_value=0.0, max_value=1000.0,
            value=100.0, step=1.0,
            key='fert_last'
        )
        user_input['Yield_Last_Season'] = st.number_input(
            FEATURE_LABELS_PT['Yield_Last_Season'],
            min_value=0.0, max_value=20.0,
            value=3.0, step=0.1,
            key='yield_last'
        )
    
    st.markdown("---")
    
    # Botão de submissão
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
    with col_btn_center:
        submit_btn = st.button(
            "🔍 Gerar Recomendação",
            width='stretch',
            type="primary"
        )
    
    # Processar predição
    if submit_btn:
        # Validação
        warnings = validate_input(user_input)
        if warnings:
            for w in warnings:
                st.warning(w)
        
        # Carregar modelo e gerar recomendação
        with st.spinner("⏳ Processando recomendação..."):
            try:
                artifacts = cached_load_artifacts()
                result = get_recommendation(user_input, artifacts)
                # Persiste no session_state
                st.session_state['last_result'] = result
                st.session_state['last_user_input'] = user_input.copy()
            except Exception as e:
                st.error(f"❌ Erro ao gerar recomendação: {str(e)}")
                st.info(
                    "Certifique-se de que o modelo foi treinado. "
                    "Execute: `python src/train.py`"
                )
    
    # Exibe o resultado sempre que houver um resultado salvo
    if st.session_state['last_result'] is not None:
        result = st.session_state['last_result']
        user_input_saved = st.session_state['last_user_input']
        
        # Card de resultado
        st.markdown(f"""
        <div class="result-card">
            <h2>🌿 {result['fertilizer']}</h2>
            <div class="confidence">
                Confiança: {result['confidence']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico SHAP Waterfall
        st.markdown("### 📊 Por que este fertilizante?")
        
        artifacts = cached_load_artifacts()
        explainer = cached_create_explainer(artifacts['model'])
        fig_waterfall = plot_waterfall(
            explainer,
            result['X_processed'],
            result['predicted_class_index'],
            result['fertilizer']
        )
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)
        
        # Texto interpretativo
        st.markdown("### 📝 Explicação")
        text_explanation = generate_text_explanation(
            explainer,
            result['X_processed'],
            result['predicted_class_index'],
            user_input_saved,
            result['fertilizer']
        )
        st.markdown(f"""
        <div class="info-box">
            {text_explanation}
        </div>
        """, unsafe_allow_html=True)
        
        # Top 5 probabilidades
        st.markdown("### 📈 Probabilidades por Fertilizante")
        top_probs = dict(list(result['all_probabilities'].items())[:5])
        df_probs = pd.DataFrame({
            'Fertilizante': top_probs.keys(),
            'Probabilidade (%)': top_probs.values()
        })
        fig_probs = px.bar(
            df_probs, x='Probabilidade (%)', y='Fertilizante',
            orientation='h',
            color='Probabilidade (%)',
            color_continuous_scale='Blues',
            text='Probabilidade (%)'
        )
        fig_probs.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_probs.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis={'categoryorder': 'total ascending'},
            height=300,
            margin=dict(l=0, r=60, t=10, b=0)
        )
        st.plotly_chart(fig_probs, width='stretch')
        
        # Botão de download CSV
        st.markdown("### 📥 Exportar Resultado")
        csv_data = {**user_input_saved}
        csv_data['Fertilizante_Recomendado'] = result['fertilizer']
        csv_data['Confianca_%'] = result['confidence']
        df_export = pd.DataFrame([csv_data])
        csv_str = df_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Baixar resultado em CSV",
            data=csv_str,
            file_name="recomendacao_adubacao.csv",
            mime="text/csv",
            width='stretch'
        )


# =============================================================================
# Aba 2 — 📊 Análise do Dataset
# =============================================================================

with tab2:
    st.markdown("### Análise Exploratória do Dataset")
    st.markdown("---")
    
    df = cached_load_data()
    
    # Informações gerais
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("📊 Registros", f"{len(df):,}")
    with col_info2:
        st.metric("📋 Variáveis", f"{len(df.columns)}")
    with col_info3:
        st.metric("🎯 Classes", f"{df[TARGET_COL].nunique()}")
    with col_info4:
        st.metric("❌ Valores Nulos", f"{df.isnull().sum().sum()}")
    
    st.markdown("---")
    
    # F09 — Distribuição das classes-alvo
    st.markdown("#### 🎯 Distribuição dos Fertilizantes Recomendados")
    class_counts = df[TARGET_COL].value_counts().reset_index()
    class_counts.columns = ['Fertilizante', 'Contagem']
    
    fig_classes = px.bar(
        class_counts,
        x='Fertilizante', y='Contagem',
        color='Contagem',
        color_continuous_scale='Tealgrn',
        text='Contagem'
    )
    fig_classes.update_traces(textposition='outside')
    fig_classes.update_layout(
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        height=450,
        margin=dict(b=100)
    )
    st.plotly_chart(fig_classes, width='stretch')
    
    st.markdown("---")
    
    # F10 — Heatmap de correlação
    st.markdown("#### 🔥 Mapa de Correlação (Variáveis Numéricas)")
    numeric_df = df[NUMERICAL_COLS]
    corr_matrix = numeric_df.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        x=[FEATURE_LABELS_PT.get(c, c) for c in corr_matrix.columns],
        y=[FEATURE_LABELS_PT.get(c, c) for c in corr_matrix.index],
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto='.2f',
        aspect='auto'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, width='stretch')
    
    st.markdown("---")
    
    # F11 — Boxplots
    st.markdown("#### 📦 Boxplots por Variável Numérica")
    selected_var = st.selectbox(
        "Selecione a variável:",
        options=NUMERICAL_COLS,
        format_func=lambda x: FEATURE_LABELS_PT.get(x, x),
        key='boxplot_var'
    )
    
    fig_box = px.box(
        df, y=selected_var, x=TARGET_COL,
        color=TARGET_COL,
        labels={
            selected_var: FEATURE_LABELS_PT.get(selected_var, selected_var),
            TARGET_COL: 'Fertilizante'
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_box.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=500,
        margin=dict(b=100)
    )
    st.plotly_chart(fig_box, width='stretch')


# =============================================================================
# Aba 3 — 🤖 Desempenho dos Modelos
# =============================================================================

with tab3:
    st.markdown("### Comparação de Desempenho dos Modelos de Machine Learning")
    st.markdown("---")
    
    try:
        metrics = cached_load_metrics()
        best_model_name = metrics['best_model']
        
        # F12 — Tabela comparativa
        st.markdown("#### 📋 Tabela Comparativa de Métricas")
        
        rows = []
        for model_name, m in metrics['models'].items():
            rows.append({
                'Modelo': model_name,
                'F1-Score (CV)': f"{m['cv_f1_macro_mean']:.4f} ± {m['cv_f1_macro_std']:.4f}",
                'Acurácia (Teste)': f"{m['test_accuracy']:.4f}",
                'Precisão (Teste)': f"{m['test_precision_macro']:.4f}",
                'Recall (Teste)': f"{m['test_recall_macro']:.4f}",
                'F1-Score (Teste)': f"{m['test_f1_macro']:.4f}",
                'Tempo (s)': f"{m['training_time_seconds']:.2f}",
            })
        
        df_metrics = pd.DataFrame(rows)
        
        # Highlight do melhor modelo
        def highlight_best(row):
            if row['Modelo'] == best_model_name:
                return ['background-color: #d4edda; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        styled_df = df_metrics.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, width='stretch', hide_index=True)
        
        st.success(f"🏆 **Melhor modelo selecionado:** {best_model_name}")
        
        st.markdown("---")
        
        # Gráfico comparativo de F1-Score
        st.markdown("#### 📊 Comparação Visual — F1-Score Macro")
        
        model_names = list(metrics['models'].keys())
        f1_cv = [metrics['models'][m]['cv_f1_macro_mean'] for m in model_names]
        f1_test = [metrics['models'][m]['test_f1_macro'] for m in model_names]
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='F1 Cross-Validation',
            x=model_names, y=f1_cv,
            marker_color='#2E86AB',
            text=[f'{v:.4f}' for v in f1_cv],
            textposition='outside'
        ))
        fig_compare.add_trace(go.Bar(
            name='F1 Teste',
            x=model_names, y=f1_test,
            marker_color='#A23B72',
            text=[f'{v:.4f}' for v in f1_test],
            textposition='outside'
        ))
        fig_compare.update_layout(
            barmode='group',
            yaxis_range=[
                min(min(f1_cv), min(f1_test)) * 0.95,
                1.02
            ],
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1)
        )
        st.plotly_chart(fig_compare, width='stretch')
        
        st.markdown("---")
        
        # F13 — Matriz de confusão
        st.markdown("#### 🔢 Matriz de Confusão")
        
        conf_matrices = cached_load_confusion_matrices()
        label_classes = joblib.load(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models', 'label_classes.pkl'
        ))
        
        selected_model_cm = st.selectbox(
            "Selecione o modelo:",
            options=list(conf_matrices.keys()),
            index=list(conf_matrices.keys()).index(best_model_name),
            key='cm_model'
        )
        
        cm = conf_matrices[selected_model_cm]
        
        fig_cm = px.imshow(
            cm,
            x=label_classes,
            y=label_classes,
            labels=dict(x="Predito", y="Real", color="Contagem"),
            color_continuous_scale='Blues',
            text_auto=True,
            aspect='auto'
        )
        fig_cm.update_layout(
            height=600,
            xaxis_tickangle=-45,
            margin=dict(b=100)
        )
        st.plotly_chart(fig_cm, width='stretch')
        
        st.markdown("---")
        
        # F14 — SHAP summary plot
        st.markdown("#### 🎯 Importância Global das Variáveis (SHAP)")
        
        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models'
        )
        fig_shap = plot_summary(models_dir)
        if fig_shap:
            st.pyplot(fig_shap)
            plt.close(fig_shap)
        else:
            st.info("SHAP global não disponível. Execute o treinamento novamente.")
        
    except FileNotFoundError:
        st.warning(
            "⚠️ Modelos ainda não foram treinados. "
            "Execute `python src/train.py` para gerar as métricas."
        )
    except Exception as e:
        st.error(f"Erro ao carregar métricas: {str(e)}")


# =============================================================================
# Aba 4 — ℹ️ Sobre o Sistema
# =============================================================================

with tab4:
    st.markdown("### Sobre o SAD-ADUBO")
    st.markdown("---")
    
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.markdown("""
        #### 🎯 O que é o SAD-ADUBO?
        
        O **SAD-ADUBO** (Sistema de Apoio à Decisão para Adubação) é um sistema 
        inteligente que utiliza técnicas de **Machine Learning** para recomendar 
        o fertilizante mais adequado com base em dados de solo, clima e manejo agrícola.
        
        O sistema foi desenvolvido como **Trabalho de Conclusão de Curso (TCC)** 
        do Bacharelado em Sistemas de Informação na **Universidade Estadual de Goiás 
        (UEG)** — Unidade Universitária de Santa Helena de Goiás.
        
        #### 📋 Metodologia
        
        1. **Coleta de Dados:** Dataset público do Kaggle com 10.000 registros 
           de recomendações de fertilizantes
        2. **Pré-processamento:** Label Encoding para variáveis categóricas e 
           MinMaxScaler para variáveis numéricas
        3. **Treinamento:** Comparação de 4 algoritmos (Decision Tree, 
           Random Forest, SVM e XGBoost)
        4. **Avaliação:** Cross-validation k=10 com F1-Score macro como 
           métrica primária
        5. **Explicabilidade:** SHAP (SHapley Additive exPlanations) para 
           justificar cada recomendação
        
        #### 🔬 Variáveis Utilizadas
        
        O modelo utiliza **19 variáveis** organizadas em 4 grupos:
        - **Solo (8):** Tipo de solo, pH, umidade, carbono orgânico, 
          condutividade elétrica, nitrogênio, fósforo e potássio
        - **Clima (3):** Temperatura, umidade do ar e precipitação
        - **Manejo (6):** Tipo de cultura, estágio de crescimento, estação, 
          tipo de irrigação, cultura anterior e região
        - **Histórico (2):** Fertilizante e produtividade da última safra
        """)
        
        st.markdown("""
        #### ⚠️ Limitações do MVP
        
        - O dataset utilizado **não é regionalizado para o Brasil** — 
          os resultados servem como prova de conceito acadêmica
        - O sistema **não fornece dosagem quantitativa** (kg/ha) — 
          apenas indica o tipo de fertilizante
        - **Não substitui** a consultoria de um engenheiro agrônomo
        - Sem integração com API climática em tempo real
        - Sem banco de dados persistente ou histórico de consultas
        """)
    
    with col_about2:
        st.markdown("""
        #### 📚 Informações Acadêmicas
        
        **Curso:**  
        Bacharelado em Sistemas de Informação
        
        **Instituição:**  
        Universidade Estadual de Goiás (UEG)  
        UnU Santa Helena de Goiás
        
        **Autor:**  
        Eduardo Augusto de Oliveira Mendes
        
        **Orientadora:**  
        Prof.ª Me. Pollyana Queiroz
        
        **Ano:**  
        2026
        
        ---
        
        #### 📖 Referências
        
        - Athayde, G. B. (2023). *Machine Learning 
          aplicado à recomendação de adubação.*
        - Lundberg, S. M., & Lee, S.-I. (2017). 
          *A Unified Approach to Interpreting 
          Model Predictions.* NeurIPS.
        - Dataset: [Kaggle — Fertilizer 
          Recommendation](https://www.kaggle.com/datasets/miadul/fertilizer-recommendation-dataset)
        
        ---
        
        #### 🛠️ Tecnologias
        
        `Python` `Streamlit` `scikit-learn` 
        `XGBoost` `SHAP` `Pandas` `Plotly`
        """)


# =============================================================================
# Footer
# =============================================================================

st.markdown("""
<div class="footer">
    <p>
        SAD-ADUBO v1.0 — TCC Sistemas de Informação | UEG 2026<br>
        Desenvolvido por Eduardo Augusto de Oliveira Mendes
    </p>
</div>
""", unsafe_allow_html=True)
