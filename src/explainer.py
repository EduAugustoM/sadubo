"""
explainer.py — Módulo de explicabilidade SHAP do SAD-ADUBO.

Gera explicações interpretáveis para cada predição usando SHAP
(SHapley Additive exPlanations), incluindo:
- Waterfall plot para explicação local (por predição)
- Summary plot para importância global de features
- Texto interpretativo em português brasileiro
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Streamlit
import matplotlib.pyplot as plt

from src.preprocessing import FEATURE_LABELS_PT, FEATURE_COLS


# =============================================================================
# Criação do Explainer
# =============================================================================

def create_explainer(model, model_type: str = 'tree'):
    """
    Cria um SHAP explainer apropriado para o tipo de modelo.
    
    Args:
        model: modelo treinado
        model_type: 'tree' para DT/RF/XGBoost, 'kernel' para SVM
    
    Returns:
        shap.Explainer
    """
    if model_type == 'kernel':
        # Fallback para SVM — mais lento
        return None  # Será tratado caso a caso
    else:
        return shap.TreeExplainer(model)


# =============================================================================
# Explicação Local (por predição)
# =============================================================================

def explain_prediction(explainer, X_processed: pd.DataFrame,
                       predicted_class_index: int):
    """
    Calcula SHAP values para uma única predição.
    
    Args:
        explainer: shap.TreeExplainer
        X_processed: DataFrame com 1 linha (input processado)
        predicted_class_index: índice da classe predita
    
    Returns:
        shap_values para a classe predita (array 1D)
    """
    shap_values = explainer.shap_values(X_processed)
    
    # Para multiclasse, shap_values é uma lista de arrays (um por classe)
    if isinstance(shap_values, list):
        return shap_values[predicted_class_index][0]
    else:
        # Se for array 3D (classes × amostras × features)
        if shap_values.ndim == 3:
            return shap_values[0, :, predicted_class_index]
        return shap_values[0]


def plot_waterfall(explainer, X_processed: pd.DataFrame,
                   predicted_class_index: int,
                   fertilizer_name: str) -> plt.Figure:
    """
    Gera um waterfall plot SHAP para a predição local.
    
    Args:
        explainer: shap.TreeExplainer
        X_processed: DataFrame com 1 linha  
        predicted_class_index: índice da classe predita
        fertilizer_name: nome do fertilizante para o título
    
    Returns:
        matplotlib.Figure
    """
    shap_vals = explainer.shap_values(X_processed)
    
    # Para multiclasse: selecionar a classe predita
    if isinstance(shap_vals, list):
        sv = shap_vals[predicted_class_index][0]
        base_value = explainer.expected_value[predicted_class_index]
    elif shap_vals.ndim == 3:
        sv = shap_vals[0, :, predicted_class_index]
        base_value = explainer.expected_value[predicted_class_index]
    else:
        sv = shap_vals[0]
        base_value = explainer.expected_value
    
    # Nomes amigáveis das features
    feature_display_names = [
        FEATURE_LABELS_PT.get(f, f) for f in FEATURE_COLS
    ]
    
    # Criar Explanation object
    explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=X_processed.values[0],
        feature_names=feature_display_names
    )
    
    # Gerar figura
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f'Por que "{fertilizer_name}"?', fontsize=14, fontweight='bold',
              pad=20)
    plt.tight_layout()
    
    return fig


# =============================================================================
# Explicação Global (summary plot)
# =============================================================================

def load_shap_global(models_dir: str = 'models') -> dict:
    """Carrega os SHAP values globais pré-calculados."""
    path = os.path.join(models_dir, 'shap_global_values.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def plot_summary(models_dir: str = 'models') -> plt.Figure:
    """
    Gera um summary plot SHAP com a importância global das features.
    
    Returns:
        matplotlib.Figure ou None se os dados não existirem
    """
    shap_data = load_shap_global(models_dir)
    if shap_data is None:
        return None
    
    shap_values = shap_data['shap_values']
    X_sample = shap_data['X_sample']
    
    # Nomes amigáveis
    feature_display_names = [
        FEATURE_LABELS_PT.get(f, f) for f in FEATURE_COLS
    ]
    
    # Para multiclasse: calcular média da magnitude absoluta por classe
    if isinstance(shap_values, list):
        # Média da importância absoluta entre todas as classes
        mean_abs_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values],
            axis=0
        )
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        mean_abs_shap = np.mean(
            np.abs(shap_values), axis=(0, 2)
        )
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Ordenar por importância
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = min(19, len(sorted_idx))
    y_pos = range(top_n - 1, -1, -1)
    bars = ax.barh(
        list(y_pos),
        mean_abs_shap[sorted_idx[:top_n]],
        color='#2E86AB',
        edgecolor='#1a5276',
        height=0.7
    )
    
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [feature_display_names[i] for i in sorted_idx[:top_n]],
        fontsize=11
    )
    ax.set_xlabel('Importância Média |SHAP|', fontsize=12)
    ax.set_title('Importância Global das Variáveis', fontsize=14,
                 fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Texto Interpretativo em Português
# =============================================================================

def generate_text_explanation(explainer, X_processed: pd.DataFrame,
                              predicted_class_index: int,
                              user_dict: dict,
                              fertilizer_name: str,
                              top_n: int = 3) -> str:
    """
    Gera uma explicação em texto natural (português brasileiro)
    baseada nos top SHAP values.
    
    Args:
        explainer: shap.TreeExplainer
        X_processed: DataFrame processado (1 linha)
        predicted_class_index: índice da classe predita
        user_dict: dict com valores brutos do usuário
        fertilizer_name: nome do fertilizante recomendado
        top_n: número de features mais importantes a destacar
    
    Returns:
        Texto explicativo em português
    """
    # Calcular SHAP values
    shap_vals = explainer.shap_values(X_processed)
    
    if isinstance(shap_vals, list):
        sv = shap_vals[predicted_class_index][0]
    elif shap_vals.ndim == 3:
        sv = shap_vals[0, :, predicted_class_index]
    else:
        sv = shap_vals[0]
    
    # Pegar top N features por magnitude absoluta
    abs_sv = np.abs(sv)
    total_importance = abs_sv.sum()
    top_indices = np.argsort(abs_sv)[::-1][:top_n]
    
    # Montar explicação
    parts = []
    for i, idx in enumerate(top_indices):
        feature_name = FEATURE_COLS[idx]
        feature_label = FEATURE_LABELS_PT.get(feature_name, feature_name)
        
        # Valor bruto do usuário
        if feature_name in user_dict:
            raw_value = user_dict[feature_name]
        else:
            raw_value = X_processed.iloc[0, idx]
        
        # Direção do impacto
        direction = "positivamente" if sv[idx] > 0 else "negativamente"
        
        if i == 0:
            parts.append(
                f"o {feature_label} (valor: {raw_value}), que influenciou {direction} a recomendação"
            )
        else:
            parts.append(
                f"o {feature_label} ({raw_value}) — impacto {direction}"
            )
    
    # Calcular contribuição percentual das top features
    top_contribution = abs_sv[top_indices].sum()
    if total_importance > 0:
        pct = (top_contribution / total_importance) * 100
    else:
        pct = 0
    
    # Montar texto final
    if len(parts) == 1:
        features_text = parts[0]
    elif len(parts) == 2:
        features_text = f"{parts[0]}, seguido por {parts[1]}"
    else:
        features_text = f"{parts[0]}, seguido por {', '.join(parts[1:-1])} e {parts[-1]}"
    
    text = (
        f"O fator que mais influenciou a recomendação de {fertilizer_name} "
        f"foi {features_text}. "
        f"Esses {len(parts)} fatores juntos representam "
        f"{pct:.0f}% da decisão do modelo."
    )
    
    return text
