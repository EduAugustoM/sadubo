"""
predict.py — Módulo de inferência do SAD-ADUBO.

Carrega o modelo serializado e realiza predições em tempo real
para inputs do usuário.
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import (
    FEATURE_COLS, TARGET_COL, FEATURE_LABELS_PT,
    preprocess_user_input, load_artifacts
)


# =============================================================================
# Carregamento de Artefatos
# =============================================================================

def load_model(models_dir: str = 'models'):
    """Carrega o modelo serializado."""
    model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    return model


def load_label_classes(models_dir: str = 'models') -> list:
    """Carrega as classes da variável-alvo (nomes dos fertilizantes)."""
    return joblib.load(os.path.join(models_dir, 'label_classes.pkl'))


def load_feature_names(models_dir: str = 'models') -> list:
    """Carrega os nomes das features na ordem usada no treinamento."""
    return joblib.load(os.path.join(models_dir, 'feature_names.pkl'))


def load_all_artifacts(models_dir: str = 'models') -> dict:
    """
    Carrega todos os artefatos necessários para predição.
    
    Returns:
        dict com: model, encoders, scaler, label_classes, feature_names
    """
    model = load_model(models_dir)
    encoders, scaler = load_artifacts(models_dir)
    label_classes = load_label_classes(models_dir)
    feature_names = load_feature_names(models_dir)
    
    return {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'label_classes': label_classes,
        'feature_names': feature_names,
    }


# =============================================================================
# Predição
# =============================================================================

def predict(model, X_processed: pd.DataFrame) -> tuple:
    """
    Realiza predição para uma entrada processada.
    
    Args:
        model: modelo treinado
        X_processed: DataFrame processado (encoded + scaled)
    
    Returns:
        (classe_predita, array_de_probabilidades)
    """
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)
    
    return y_pred, y_proba


def get_recommendation(user_dict: dict, artifacts: dict) -> dict:
    """
    Pipeline completo de recomendação: recebe inputs brutos do usuário
    e retorna o fertilizante recomendado com probabilidades.
    
    Args:
        user_dict: dict com valores inseridos pelo usuário
        artifacts: dict retornado por load_all_artifacts()
    
    Returns:
        dict com:
        - fertilizer: nome do fertilizante recomendado
        - confidence: probabilidade da classe predita (0-100%)
        - all_probabilities: dict {fertilizante: probabilidade}
        - X_processed: DataFrame processado (para SHAP)
        - predicted_class_index: índice da classe predita
    """
    model = artifacts['model']
    encoders = artifacts['encoders']
    scaler = artifacts['scaler']
    label_classes = artifacts['label_classes']
    
    # Pré-processar input do usuário
    X_processed = preprocess_user_input(user_dict, encoders, scaler)
    
    # Predição
    y_pred, y_proba = predict(model, X_processed)
    
    # Decodificar classe predita
    predicted_class_idx = int(y_pred[0])
    fertilizer_name = label_classes[predicted_class_idx]
    confidence = float(y_proba[0][predicted_class_idx]) * 100
    
    # Todas as probabilidades
    all_probs = {}
    for i, cls_name in enumerate(label_classes):
        all_probs[cls_name] = round(float(y_proba[0][i]) * 100, 2)
    
    # Ordenar por probabilidade decrescente
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'fertilizer': fertilizer_name,
        'confidence': round(confidence, 2),
        'all_probabilities': all_probs,
        'X_processed': X_processed,
        'predicted_class_index': predicted_class_idx,
    }
