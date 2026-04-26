"""
preprocessing.py — Pipeline de pré-processamento para o SAD-ADUBO.

Responsável por:
- Carregamento do dataset
- Definição de colunas (categóricas, numéricas, alvo)
- Label Encoding das variáveis categóricas
- MinMaxScaler das variáveis numéricas
- Divisão treino/teste (80/20, stratify, random_state=42)
- Transformação de inputs do usuário para predição
- Validação de ranges agronômicos
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


# =============================================================================
# Configuração de Colunas (baseada no dataset real do Kaggle)
# =============================================================================

CATEGORICAL_COLS = [
    'Soil_Type',
    'Crop_Type',
    'Crop_Growth_Stage',
    'Season',
    'Irrigation_Type',
    'Previous_Crop',
    'Region',
]

NUMERICAL_COLS = [
    'Soil_pH',
    'Soil_Moisture',
    'Organic_Carbon',
    'Electrical_Conductivity',
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level',
    'Temperature',
    'Humidity',
    'Rainfall',
    'Fertilizer_Used_Last_Season',
    'Yield_Last_Season',
]

TARGET_COL = 'Recommended_Fertilizer'

# Ordem completa das features (categóricas primeiro, depois numéricas)
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# Ranges válidos para validação agronômica dos inputs
VALID_RANGES = {
    'Soil_pH':                    (3.0, 10.0),
    'Soil_Moisture':              (0.0, 100.0),
    'Organic_Carbon':             (0.0, 10.0),
    'Electrical_Conductivity':    (0.0, 10.0),
    'Nitrogen_Level':             (0, 200),
    'Phosphorus_Level':           (0, 200),
    'Potassium_Level':            (0, 300),
    'Temperature':                (-10.0, 55.0),
    'Humidity':                   (0.0, 100.0),
    'Rainfall':                   (0.0, 5000.0),
    'Fertilizer_Used_Last_Season': (0.0, 1000.0),
    'Yield_Last_Season':          (0.0, 20.0),
}

# Nomes amigáveis em português para exibição na interface
FEATURE_LABELS_PT = {
    'Soil_Type':                  'Tipo de Solo',
    'Crop_Type':                  'Tipo de Cultura',
    'Crop_Growth_Stage':          'Estágio de Crescimento',
    'Season':                     'Estação do Ano',
    'Irrigation_Type':            'Tipo de Irrigação',
    'Previous_Crop':              'Cultura Anterior',
    'Region':                     'Região',
    'Soil_pH':                    'pH do Solo',
    'Soil_Moisture':              'Umidade do Solo (%)',
    'Organic_Carbon':             'Carbono Orgânico (%)',
    'Electrical_Conductivity':    'Condutividade Elétrica (dS/m)',
    'Nitrogen_Level':             'Nível de Nitrogênio (mg/kg)',
    'Phosphorus_Level':           'Nível de Fósforo (mg/kg)',
    'Potassium_Level':            'Nível de Potássio (mg/kg)',
    'Temperature':                'Temperatura (°C)',
    'Humidity':                   'Umidade do Ar (%)',
    'Rainfall':                   'Precipitação (mm)',
    'Fertilizer_Used_Last_Season': 'Fertilizante Usado Última Safra (kg/ha)',
    'Yield_Last_Season':          'Produtividade Última Safra (ton/ha)',
}


# =============================================================================
# Carregamento de Dados
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV."""
    df = pd.read_csv(path)
    return df


def get_unique_values(df: pd.DataFrame) -> dict:
    """Retorna os valores únicos de cada coluna categórica (para dropdowns)."""
    unique_vals = {}
    for col in CATEGORICAL_COLS:
        unique_vals[col] = sorted(df[col].unique().tolist())
    # Inclui os valores da variável-alvo
    unique_vals[TARGET_COL] = sorted(df[TARGET_COL].unique().tolist())
    return unique_vals


# =============================================================================
# Encoding (LabelEncoder)
# =============================================================================

def fit_encoders(df: pd.DataFrame) -> dict:
    """
    Ajusta um LabelEncoder para cada coluna categórica.
    Retorna um dict: {nome_coluna: LabelEncoder_fitado}.
    
    IMPORTANTE: deve ser chamado APENAS com dados de treino.
    """
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le
    
    # Encoder para a variável-alvo
    le_target = LabelEncoder()
    le_target.fit(df[TARGET_COL])
    encoders[TARGET_COL] = le_target
    
    return encoders


def transform_encoders(df: pd.DataFrame, encoders: dict,
                       include_target: bool = True) -> pd.DataFrame:
    """
    Aplica LabelEncoding nas colunas categóricas usando encoders já fitados.
    Retorna uma cópia do DataFrame com as colunas transformadas.
    """
    df_encoded = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df_encoded.columns:
            df_encoded[col] = encoders[col].transform(df_encoded[col])
    
    if include_target and TARGET_COL in df_encoded.columns:
        df_encoded[TARGET_COL] = encoders[TARGET_COL].transform(
            df_encoded[TARGET_COL]
        )
    
    return df_encoded


def inverse_transform_target(encoded_values, encoders: dict):
    """Converte valores numéricos de volta para nomes de fertilizantes."""
    return encoders[TARGET_COL].inverse_transform(encoded_values)


# =============================================================================
# Scaling (MinMaxScaler)
# =============================================================================

def fit_scaler(X_train: pd.DataFrame) -> MinMaxScaler:
    """
    Ajusta MinMaxScaler nas colunas numéricas do conjunto de treino.
    
    IMPORTANTE: deve ser chamado APENAS com dados de treino.
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train[NUMERICAL_COLS])
    return scaler


def transform_scaler(X: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Aplica MinMaxScaler nas colunas numéricas. Retorna cópia transformada."""
    X_scaled = X.copy()
    X_scaled[NUMERICAL_COLS] = scaler.transform(X[NUMERICAL_COLS])
    return X_scaled


# =============================================================================
# Split dos Dados
# =============================================================================

def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42):
    """
    Divide o dataset em treino e teste.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Pipeline Completo para Treinamento
# =============================================================================

def prepare_training_data(data_path: str):
    """
    Pipeline completo de pré-processamento para treinamento.
    
    1. Carrega dados
    2. Fita encoders no dataset completo (antes do split, pois LabelEncoder
       precisa conhecer todas as categorias)
    3. Aplica encoding
    4. Divide treino/teste
    5. Fita scaler no treino
    6. Aplica scaling em treino e teste
    
    Returns:
        X_train_processed, X_test_processed, y_train, y_test, encoders, scaler
    """
    # 1. Carregar
    df = load_data(data_path)
    
    # 2. Fitar encoders (precisa ver todas as categorias)
    encoders = fit_encoders(df)
    
    # 3. Aplicar encoding
    df_encoded = transform_encoders(df, encoders, include_target=True)
    
    # 4. Split
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    
    # 5. Fitar scaler APENAS no treino
    scaler = fit_scaler(X_train)
    
    # 6. Aplicar scaling
    X_train_processed = transform_scaler(X_train, scaler)
    X_test_processed = transform_scaler(X_test, scaler)
    
    return X_train_processed, X_test_processed, y_train, y_test, encoders, scaler


# =============================================================================
# Pré-processamento de Input do Usuário (Streamlit)
# =============================================================================

def preprocess_user_input(user_dict: dict, encoders: dict,
                          scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Converte os inputs do formulário Streamlit em um DataFrame
    normalizado pronto para predição.
    
    Args:
        user_dict: dict com os valores inseridos pelo usuário
                   (chaves = nomes das colunas do dataset)
        encoders: dict de LabelEncoders fitados
        scaler: MinMaxScaler fitado
    
    Returns:
        DataFrame com 1 linha, colunas na ordem correta, processado
    """
    # Criar DataFrame com uma linha
    df_input = pd.DataFrame([user_dict])
    
    # Garantir ordem correta das colunas
    df_input = df_input[FEATURE_COLS]
    
    # Aplicar encoding nas categóricas
    for col in CATEGORICAL_COLS:
        df_input[col] = encoders[col].transform(df_input[col])
    
    # Aplicar scaling nas numéricas
    df_input[NUMERICAL_COLS] = scaler.transform(df_input[NUMERICAL_COLS])
    
    return df_input


def validate_input(user_dict: dict) -> list:
    """
    Valida os inputs numéricos do usuário contra os ranges agronômicos.
    
    Returns:
        Lista de strings com mensagens de warning (vazia se tudo OK)
    """
    warnings = []
    for col, (vmin, vmax) in VALID_RANGES.items():
        if col in user_dict:
            val = user_dict[col]
            label = FEATURE_LABELS_PT.get(col, col)
            if val < vmin or val > vmax:
                warnings.append(
                    f"⚠️ {label}: valor {val} está fora do range "
                    f"esperado ({vmin}–{vmax})"
                )
    return warnings


# =============================================================================
# Serialização dos Artefatos
# =============================================================================

def save_artifacts(encoders: dict, scaler: MinMaxScaler,
                   output_dir: str = 'models'):
    """Salva encoders e scaler como arquivos .pkl."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(encoders, os.path.join(output_dir, 'encoders.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"✅ Artefatos salvos em {output_dir}/")


def load_artifacts(models_dir: str = 'models'):
    """Carrega encoders e scaler de arquivos .pkl."""
    encoders = joblib.load(os.path.join(models_dir, 'encoders.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    return encoders, scaler
