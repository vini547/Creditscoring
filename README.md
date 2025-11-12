# Credit Scoring - LightGBM + PCA

Aplicativo em **Streamlit** para realizar scoring de crédito utilizando um modelo treinado com **LightGBM** e **PCA**.

---

## Funcionalidades

- Upload de arquivos CSV ou Feather com dados dos clientes.
- Pré-processamento automático de colunas numéricas e categóricas (tratamento de nulos, log, winsorização, encoding).
- Transformação via **PCA** e aplicação do modelo LightGBM treinado com PyCaret.
- Visualização dos resultados (top 10 linhas) no app.
- Download da base escorada em CSV.

---

## Tecnologias

- Python 3.11+
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [PyCaret](https://pycaret.org/)

---

## Requisitos

```text
streamlit>=1.25
pandas>=2.1
numpy>=1.26
scikit-learn>=1.3
lightgbm>=4.0
pycaret>=3.0
joblib>=1.3
