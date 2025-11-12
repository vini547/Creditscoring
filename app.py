# app_streamlit_pipeline.py
import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from io import BytesIO

st.set_page_config(page_title="Credit Scoring LightGBM + PCA", layout="wide")
st.title("Credit Scoring - LightGBM + PCA")

uploaded_file = st.file_uploader("Selecione um arquivo CSV ou Feather", type=['csv','ftr'])

if uploaded_file is not None:
    try:
        # Ler arquivo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_feather(uploaded_file)

        st.success(f"Arquivo carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        st.dataframe(df.head())

        # ---------- Pré-processamento ----------
        df_proc = df.copy()

        # Substituir nulos
        if 'tempo_emprego' in df_proc.columns:
            df_proc['tempo_emprego'] = df_proc['tempo_emprego'].fillna(df_proc['tempo_emprego'].median())

        # Log + winsor
        if 'renda' in df_proc.columns:
            df_proc['renda'] = np.log1p(df_proc['renda'])
            df_proc['renda'] = np.clip(df_proc['renda'], None, np.log1p(40594.15))
        if 'tempo_emprego' in df_proc.columns:
            df_proc['tempo_emprego'] = np.clip(df_proc['tempo_emprego'], None, 25)
        if 'qt_pessoas_residencia' in df_proc.columns:
            df_proc['qt_pessoas_residencia'] = np.clip(df_proc['qt_pessoas_residencia'], None, 6)

        # Selecionar features originais que foram usadas no treino
        features_orig = ['idade','tempo_emprego','qt_pessoas_residencia','renda','qtd_filhos',
                         'sexo','posse_de_veiculo','posse_de_imovel','tipo_renda','educacao','estado_civil','tipo_residencia']

        X_new = df_proc[features_orig].copy()

        # One-hot encoding
        cat_cols = X_new.select_dtypes(include='object').columns.tolist()
        num_cols = X_new.select_dtypes(exclude='object').columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])

        X_encoded = preprocessor.fit_transform(X_new)  # ⚠️ aqui deve idealmente usar fit do treino original

        # ---------- PCA ----------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)  # ⚠️ usar fit do treino original
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)  # ⚠️ usar fit do treino original

        df_pca = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(5)]
        )

        # ---------- Aplicar modelo PyCaret ----------
        model = load_model('credit_scoring_lightgbm_pca_pipeline')
        df_result = predict_model(model, data=df_pca)

        st.subheader("Base escorada (Top 10 linhas)")
        st.dataframe(df_result.head(10))

        # ---------- Download ----------
        to_write = BytesIO()
        df_result.to_csv(to_write, index=False)
        to_write.seek(0)
        st.download_button(
            "Baixar base escorada",
            data=to_write,
            file_name="credit_scoring_scored.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
