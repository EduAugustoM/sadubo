"""
train.py — Treinamento, avaliação e serialização dos modelos do SAD-ADUBO.

Treina 4 algoritmos (Decision Tree, Random Forest, SVM, XGBoost),
avalia com cross-validation k=10 e seleciona o melhor por F1-Score macro.
Serializa o melhor modelo e calcula SHAP global.

Uso:
    python src/train.py
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    prepare_training_data, save_artifacts,
    FEATURE_COLS, TARGET_COL, inverse_transform_target
)

warnings.filterwarnings('ignore')

# =============================================================================
# Configuração
# =============================================================================

RANDOM_STATE = 42
CV_FOLDS = 10
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'fertilizer_recommendation.csv'
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models'
)


def get_models() -> dict:
    """Retorna dicionário com os 4 modelos a serem comparados."""
    return {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            max_features='sqrt',
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            verbosity=0
        ),
    }


# =============================================================================
# Treinamento e Avaliação
# =============================================================================

def evaluate_models(X_train, X_test, y_train, y_test):
    """
    Treina e avalia todos os modelos.
    
    Para cada modelo:
    1. Cross-validation k=10 no treino (F1-Score macro)
    2. Treina no treino completo
    3. Avalia no teste (Acurácia, Precisão, Recall, F1)
    4. Gera Matriz de Confusão
    
    Returns:
        results: dict com métricas de cada modelo
        trained_models: dict com modelos treinados
        confusion_matrices: dict com matrizes de confusão
    """
    models = get_models()
    results = {}
    trained_models = {}
    conf_matrices = {}
    
    print("=" * 70)
    print("SAD-ADUBO — Treinamento e Avaliação de Modelos")
    print("=" * 70)
    print(f"\nDataset: {len(X_train)} treino / {len(X_test)} teste")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Cross-validation: {CV_FOLDS}-fold\n")
    
    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"📊 {name}")
        print(f"{'─' * 50}")
        
        start_time = time.time()
        
        # 1. Cross-validation (F1-Score macro)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS,
            scoring='f1_macro',
            n_jobs=-1
        )
        cv_f1_mean = cv_scores.mean()
        cv_f1_std = cv_scores.std()
        
        # 2. Treinar no treino completo
        model.fit(X_train, y_train)
        
        # 3. Predição no teste
        y_pred = model.predict(X_test)
        
        # 4. Métricas no teste
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        elapsed = time.time() - start_time
        
        # Salvar resultados
        results[name] = {
            'cv_f1_macro_mean': round(cv_f1_mean, 6),
            'cv_f1_macro_std': round(cv_f1_std, 6),
            'test_accuracy': round(acc, 6),
            'test_precision_macro': round(prec, 6),
            'test_recall_macro': round(rec, 6),
            'test_f1_macro': round(f1, 6),
            'training_time_seconds': round(elapsed, 2),
        }
        trained_models[name] = model
        conf_matrices[name] = cm
        
        # Imprimir resultado
        print(f"  CV F1-macro:     {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        print(f"  Teste Acurácia:  {acc:.4f}")
        print(f"  Teste Precisão:  {prec:.4f}")
        print(f"  Teste Recall:    {rec:.4f}")
        print(f"  Teste F1-macro:  {f1:.4f}")
        print(f"  Tempo:           {elapsed:.2f}s")
    
    return results, trained_models, conf_matrices


def select_best_model(results: dict, trained_models: dict) -> tuple:
    """
    Seleciona o modelo com maior F1-Score macro médio no cross-validation.
    
    Returns:
        (nome_do_modelo, modelo_treinado)
    """
    best_name = max(results, key=lambda k: results[k]['cv_f1_macro_mean'])
    best_model = trained_models[best_name]
    
    print(f"\n{'=' * 70}")
    print(f"🏆 Melhor modelo: {best_name}")
    print(f"   CV F1-macro: {results[best_name]['cv_f1_macro_mean']:.4f}")
    print(f"   Teste F1-macro: {results[best_name]['test_f1_macro']:.4f}")
    print(f"{'=' * 70}")
    
    return best_name, best_model


# =============================================================================
# Serialização
# =============================================================================

def save_model_artifacts(best_model, best_name: str, results: dict,
                         conf_matrices: dict, encoders: dict,
                         X_test, y_test,
                         output_dir: str = 'models'):
    """Serializa o melhor modelo e todos os artefatos necessários."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Modelo
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    print(f"✅ Modelo salvo: {output_dir}/best_model.pkl")
    
    # 2. Métricas (JSON legível)
    metrics_output = {
        'best_model': best_name,
        'models': {}
    }
    for name, metrics in results.items():
        metrics_output['models'][name] = metrics
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)
    print(f"✅ Métricas salvas: {output_dir}/metrics.json")
    
    # 3. Matrizes de confusão
    joblib.dump(conf_matrices, os.path.join(output_dir, 'confusion_matrices.pkl'))
    print(f"✅ Matrizes de confusão salvas: {output_dir}/confusion_matrices.pkl")
    
    # 4. Nomes das features
    joblib.dump(FEATURE_COLS, os.path.join(output_dir, 'feature_names.pkl'))
    print(f"✅ Nomes das features salvos: {output_dir}/feature_names.pkl")
    
    # 5. Classes da variável-alvo
    label_classes = encoders[TARGET_COL].classes_.tolist()
    joblib.dump(label_classes, os.path.join(output_dir, 'label_classes.pkl'))
    print(f"✅ Classes salvas: {output_dir}/label_classes.pkl")
    
    # 6. SHAP global (amostra do teste)
    print("\n⏳ Calculando SHAP global (pode levar alguns segundos)...")
    try:
        import shap
        
        # Usar amostra para acelerar
        sample_size = min(500, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        
        # Verificar tipo do modelo para escolher explainer adequado
        if best_name == 'SVM (RBF)':
            # SVM não é tree-based; usar KernelExplainer (mais lento)
            explainer = shap.KernelExplainer(
                best_model.predict_proba,
                shap.sample(X_test, min(100, len(X_test)))
            )
            shap_values = explainer.shap_values(X_sample)
        else:
            # Decision Tree, Random Forest, XGBoost → TreeExplainer
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_sample)
        
        shap_data = {
            'shap_values': shap_values,
            'X_sample': X_sample,
            'feature_names': FEATURE_COLS,
        }
        joblib.dump(shap_data, os.path.join(output_dir, 'shap_global_values.pkl'))
        print(f"✅ SHAP global salvo: {output_dir}/shap_global_values.pkl")
        
    except Exception as e:
        print(f"⚠️ Erro ao calcular SHAP global: {e}")
        print("   O SHAP local (por predição) ainda funcionará normalmente.")


# =============================================================================
# Main
# =============================================================================

def main():
    """Pipeline principal de treinamento."""
    print("\n🌱 SAD-ADUBO — Iniciando pipeline de treinamento\n")
    
    # 1. Pré-processamento
    print("1️⃣  Pré-processando dados...")
    X_train, X_test, y_train, y_test, encoders, scaler = \
        prepare_training_data(DATA_PATH)
    print(f"   Treino: {X_train.shape} | Teste: {X_test.shape}")
    
    # 2. Salvar encoders e scaler
    print("\n2️⃣  Salvando artefatos de pré-processamento...")
    save_artifacts(encoders, scaler, MODELS_DIR)
    
    # 3. Treinar e avaliar modelos
    print("\n3️⃣  Treinando e avaliando modelos...")
    results, trained_models, conf_matrices = evaluate_models(
        X_train, X_test, y_train, y_test
    )
    
    # 4. Selecionar melhor modelo
    print("\n4️⃣  Selecionando melhor modelo...")
    best_name, best_model = select_best_model(results, trained_models)
    
    # 5. Verificar critério de aceite (F1 ≥ 95%)
    test_f1 = results[best_name]['test_f1_macro']
    if test_f1 >= 0.95:
        print(f"\n✅ Critério de aceite ATENDIDO: F1-macro = {test_f1:.4f} ≥ 0.95")
    else:
        print(f"\n⚠️ Critério de aceite NÃO atendido: F1-macro = {test_f1:.4f} < 0.95")
    
    # 6. Serializar tudo
    print("\n5️⃣  Serializando artefatos...")
    save_model_artifacts(
        best_model, best_name, results,
        conf_matrices, encoders, X_test, y_test,
        output_dir=MODELS_DIR
    )
    
    print("\n" + "=" * 70)
    print("🎉 Pipeline de treinamento concluído com sucesso!")
    print("=" * 70)
    
    return results, best_name, best_model


if __name__ == '__main__':
    main()
