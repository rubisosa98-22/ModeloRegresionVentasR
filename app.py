
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

st.set_page_config(page_title="Pronóstico de Ventas | Regresión", layout="centered")

st.title("Pronóstico de Ventas con Regresión (Simple + Ponderada)")
st.caption("Carga tus series históricas y proyecta ventas del próximo periodo usando regresiones simples y un promedio ponderado por R².")

# ---------- Funciones auxiliares ----------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        c_norm = str(c).strip().lower()
        if c_norm in {"ano", "año", "year"}:
            col_map[c] = "Año"
        elif c_norm in {"pib", "gdp"}:
            col_map[c] = "PIB"
        elif c_norm in {"desemp", "desempleo", "unemployment"}:
            col_map[c] = "Desemp"
        elif "tipo" in c_norm and "cam" in c_norm:
            col_map[c] = "TipoCam"
        elif c_norm in {"inflacion", "inflación", "inflation"}:
            col_map[c] = "Inflación"
        elif c_norm in {"ventas", "sales"}:
            col_map[c] = "Ventas"
    return df.rename(columns=col_map)

def pct_format(x, decimals=2):
    if pd.isna(x):
        return ""
    return f"{x:.{decimals}f}%"

def num_format(x, decimals=2):
    if pd.isna(x):
        return ""
    return f"{x:,.{decimals}f}"

# ---------- Entrada de archivo ----------
st.subheader("1) Carga tu archivo")
up = st.file_uploader("Archivo (.xlsx, .xls, .csv) con columnas: Año, PIB, Desemp, TipoCam, Inflación, Ventas", type=["xlsx","xls","csv"])

if not up:
    st.stop()

if up.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(up)
else:
    df_raw = pd.read_excel(up, engine="openpyxl")

df = standardize_columns(df_raw.copy())
required_cols = ["Año","PIB","Desemp","TipoCam","Inflación","Ventas"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Falta la columna {col}")
        st.stop()

for c in ["PIB","Desemp","TipoCam","Inflación","Ventas"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("Año").reset_index(drop=True)
df["Crecimiento"] = df["Ventas"].pct_change() * 100

st.subheader("2) Datos reconocidos")
st.dataframe(df)

# ---------- Parámetros de pronóstico ----------
st.subheader("3) Proyección para el siguiente periodo")
last_year = int(df["Año"].dropna().max())
next_label = st.text_input("Etiqueta del periodo a pronosticar", value=f"{last_year+1}p")

col1, col2, col3, col4 = st.columns(4)
with col1:
    x_pib = st.number_input("PIB (%)", value=0.0, step=0.1, format="%.2f")
with col2:
    x_des = st.number_input("Desempleo (%)", value=0.0, step=0.1, format="%.2f")
with col3:
    x_tc = st.number_input("Tipo de Cambio (%)", value=0.0, step=0.1, format="%.2f")
with col4:
    x_inf = st.number_input("Inflación (%)", value=0.0, step=0.1, format="%.2f")

x_2004 = {"PIB": x_pib, "Desemp": x_des, "TipoCam": x_tc, "Inflación": x_inf}

# ---------- Regresiones ----------
resultados = []
for var in ["PIB","Desemp","TipoCam","Inflación"]:
    X = df[[var]].iloc[1:].values
    y = df["Crecimiento"].iloc[1:].values
    model = LinearRegression().fit(X,y)
    pendiente = model.coef_[0]
    intercepto = model.intercept_
    r2 = model.score(X,y)
    corr, _ = pearsonr(df[var].iloc[1:], df["Crecimiento"].iloc[1:])
    tasa_pred = model.predict([[x_2004[var]]])[0]
    ventas_2004 = df["Ventas"].iloc[-1] * (1 + tasa_pred/100)
    resultados.append({
        "Variable": var,
        "Correlación": round(abs(corr),2),
        "Pendiente": round(pendiente,4),
        "Intercepto": round(intercepto,4),
        "R2": round(r2*100,0),
        "TasaPronosticada_%": round(tasa_pred,2),
        "VentasPronosticadas": round(ventas_2004,2),
        "Ponderador": round(r2*100,2)
    })

peso_total = sum(r["Ponderador"] for r in resultados)
tasa_total = sum(r["TasaPronosticada_%"]*r["Ponderador"] for r in resultados)/peso_total if peso_total>0 else 0
ventas_total = df["Ventas"].iloc[-1] * (1 + tasa_total/100)
resultados.append({
    "Variable": "PROMEDIO PONDERADO",
    "Correlación": None,
    "Pendiente": None,
    "Intercepto": None,
    "R2": None,
    "TasaPronosticada_%": round(tasa_total,2),
    "VentasPronosticadas": round(ventas_total,2),
    "Ponderador": 100
})

df_resultados = pd.DataFrame(resultados)

st.subheader("4) Resultados")
st.dataframe(df_resultados)
