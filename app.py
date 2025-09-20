# app.py — Pronóstico de Ventas con Regresiones y Ponderación por R²
# Autor: Rubi + ChatGPT
# Descripción: Carga un Excel con años en la fila 1 y variables en la columna A (PIB, Desempleo,
# Tipo de Cambio, Inflación, Ventas). Detecta automáticamente años históricos (XXXX) y el año
# con 'p' como pronóstico. Ejecuta regresiones simples y el modelo ponderado por R². Permite
# editar los supuestos del año 'p' para las macros de forma dinámica y descargar resultados a Excel.

import re
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Pronóstico de Ventas (Regresiones + R²)", layout="wide")
st.title("📈 Pronóstico de Ventas (Regresiones + R²)")
st.caption("Formato esperado: **Fila 1** = Años (Año | 2000 | 2001 | ... | 2011p), **Columna A** = variables (PIB, Desempleo, Tipo de Cambio, Inflación, Ventas). Valores YoY en decimales (0.025 = 2.5%).")

# ---------- Utilidades ----------
YEAR_RE = re.compile(r"^\d{4}p?$")

def normalize_header(h):
    """Convierte encabezados de 2000.0 -> '2000', mantiene '2004p'."""
    s = str(h).strip()
    try:
        f = float(s)
        if float(f).is_integer() and 1900 <= int(f) <= 2100:
            return str(int(f))
    except Exception:
        pass
    if re.fullmatch(r"\d{4}p", s) or re.fullmatch(r"\d{4}", s):
        return s
    return s

def detect_structure(df_raw: pd.DataFrame):
    """Asume fila 0 = encabezados y col 0 = variables, como en tu instructivo."""
    headers = [normalize_header(x) for x in df_raw.iloc[0].tolist()]
    df = df_raw.iloc[1:].copy()
    df.columns = headers
    var_col = df.columns[0]
    year_cols = [c for c in df.columns if YEAR_RE.fullmatch(str(c))]
    tidy = df[[var_col] + year_cols].copy()
    tidy.rename(columns={var_col: "variable"}, inplace=True)
    # Numerificar
    for c in year_cols:
        tidy[c] = pd.to_numeric(tidy[c], errors="coerce")
    # Identificar años hist y pronóstico
    hist_years = [y for y in year_cols if not str(y).endswith("p")]
    hist_years_sorted = sorted(hist_years, key=lambda x: int(str(x)[:4]))
    fore_years = [y for y in year_cols if str(y).endswith("p")]
    fore_year = fore_years[0] if fore_years else None
    return tidy, hist_years_sorted, fore_year

def run_simple_reg(x, y, x_fore=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < 2:
        return dict(n=n, r=np.nan, beta=np.nan, alpha=np.nan, r2=np.nan, yhat=np.nan)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    yhat = intercept + slope * x_fore if x_fore is not None and not np.isnan(x_fore) else np.nan
    return dict(n=n, r=float(r_value), beta=float(slope), alpha=float(intercept), r2=float(r_value**2), yhat=float(yhat) if not np.isnan(yhat) else np.nan)

def to_percent(x, dec=2):
    if pd.isna(x): return None
    return round(float(x)*100, dec)

def to_fixed(x, dec=2):
    if pd.isna(x): return None
    return round(float(x), dec)

def build_excel(simple_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Genera un archivo Excel en memoria con 3 hojas: simples, ponderado y resumen."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # H1: Regresiones simples
        simple_save = simple_df.copy()
        simple_save.to_excel(writer, index=False, sheet_name="1_Regresiones_Simples")
        # H2: Ponderado por R2
        if "Aporte ponderado" in simple_save.columns and "Peso R2" in simple_save.columns:
            keep_cols = ["Variable macro", "R2", "Peso R2", "Pronóstico simple", "Aporte ponderado"]
            keep_cols = [c for c in keep_cols if c in simple_save.columns]
            simple_save[keep_cols].to_excel(writer, index=False, sheet_name="2_Ponderado_R2")
        # H3: Resumen
        summary_df.to_excel(writer, index=False, sheet_name="3_Resumen")
    output.seek(0)
    return output

# ---------- Sidebar: Carga de datos ----------
with st.sidebar:
    st.header("⚙️ Parámetros")
    up = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.markdown("**Tips de formato**")
    st.markdown("- Fila 1: Año | 2000 | 2001 | ... | 2011p")
    st.markdown("- Columna A: PIB, Desempleo, Tipo de Cambio, Inflación, Ventas")
    st.markdown("- Valores YoY en decimales (0.025 = 2.5%)")

if not up:
    st.info("Sube un archivo Excel para comenzar.")
    st.stop()

# ---------- Lectura ----------
try:
    df_raw = pd.read_excel(up, sheet_name=0, header=None)
except Exception as e:
    st.error(f"No pude leer el archivo: {e}")
    st.stop()

# ---------- Detección de estructura ----------
try:
    tidy, hist_years, fore_year = detect_structure(df_raw)
except Exception as e:
    st.error(f"Error al detectar estructura: {e}")
    st.stop()

st.subheader("🔎 Detección automática")
c1, c2, c3 = st.columns(3)
with c1: st.write("**Años históricos:**", hist_years if hist_years else "No detectados")
with c2: st.write("**Año de pronóstico:**", fore_year if fore_year else "No detectado")
with c3:
    st.write("**Variables detectadas:**")
    st.dataframe(tidy[["variable"]])

if not hist_years or fore_year is None:
    st.warning("Asegúrate de que existan columnas de años 'XXXX' (históricos) y una columna 'XXXXp' (pronóstico).")
    st.stop()

# ---------- Selección y edición de supuestos del año 'p' ----------
st.subheader("🧮 Supuestos del año de pronóstico (editable)")
macros = ["PIB", "Desempleo", "Tipo de Cambio", "Inflación"]
macros_present = [m for m in macros if m in tidy["variable"].astype(str).tolist()]
if not macros_present:
    st.error("No encontré las variables macro esperadas (PIB, Desempleo, Tipo de Cambio, Inflación).")
    st.stop()

# Tomar valores actuales del año 'p' para los macros y permitir edición
fore_inputs = {}
edit_cols = st.columns(len(macros_present))
for i, m in enumerate(macros_present):
    v = tidy.loc[tidy["variable"]==m, fore_year].astype(float).values[0] if not tidy.loc[tidy["variable"]==m, fore_year].isna().all() else np.nan
    fore_inputs[m] = edit_cols[i].number_input(f"{m} ({fore_year})", value=float(v) if pd.notna(v) else 0.0, format="%.6f")

# Reemplazar con inputs del usuario
for m in macros_present:
    tidy.loc[tidy["variable"]==m, fore_year] = fore_inputs[m]

# ---------- Construcción de Y e X ----------
if "Ventas" not in tidy["variable"].astype(str).tolist():
    st.error("No encontré la fila 'Ventas' (variable dependiente).")
    st.stop()

y_row = tidy[tidy["variable"]=="Ventas"].iloc[0]
y_hist = y_row[hist_years].astype(float).values

# ---------- Regresiones simples ----------
rows = []
for m in macros_present:
    x_row = tidy[tidy["variable"]==m].iloc[0]
    x_hist = x_row[hist_years].astype(float).values
    x_fore = float(x_row[fore_year]) if pd.notna(x_row[fore_year]) else np.nan
    met = run_simple_reg(x_hist, y_hist, x_fore)
    rows.append({
        "Variable macro": m,
        "n": met["n"],
        "Coef. Pearson": met["r"],
        "Beta": met["beta"],
        "Alpha": met["alpha"],
        "R2": met["r2"],
        "Pronóstico simple": met["yhat"]
    })

simple_df = pd.DataFrame(rows)

# ---------- Ponderado por R² ----------
if not simple_df.empty:
    r2s = simple_df["R2"].clip(lower=0).fillna(0.0)
    if r2s.sum() > 0:
        w = r2s / r2s.sum()
    else:
        w = pd.Series([1/len(r2s)]*len(r2s)) if len(r2s)>0 else pd.Series(dtype=float)
    simple_df["Peso R2"] = w
    simple_df["Aporte ponderado"] = simple_df["Peso R2"] * simple_df["Pronóstico simple"]
    total_growth = simple_df["Aporte ponderado"].sum()
else:
    total_growth = np.nan

# ---------- Presentación ----------
st.subheader("📋 Resultados")
# Formato legible (% con 2 decimales para crecimientos; no % con 2 decimales)
simple_show = simple_df.copy()
# columnas no porcentaje con 2 decimales
for col in ["Coef. Pearson", "Beta", "Alpha", "R2"]:
    if col in simple_show.columns:
        simple_show[col] = simple_show[col].apply(to_fixed)
# columnas porcentaje con 2 decimales
simple_show["Pronóstico simple (%)"] = simple_show["Pronóstico simple"].apply(to_percent)
simple_show["Aporte ponderado (%)"] = simple_show["Aporte ponderado"].apply(to_percent)

st.markdown("**Regresiones simples**")
st.dataframe(simple_show[["Variable macro","n","Coef. Pearson","Beta","Alpha","R2","Pronóstico simple (%)","Aporte ponderado (%)"]])

st.markdown("**Resumen ponderado por R²**")
summary_df = pd.DataFrame({
    "Variable objetivo": ["Ventas"],
    "Año de pronóstico": [fore_year],
    "Crecimiento total ponderado": [to_fixed(total_growth)],
    "Crecimiento total ponderado (%)": [to_percent(total_growth)]
})
st.dataframe(summary_df)

st.success(f"**Crecimiento ponderado estimado ({fore_year}): {to_percent(total_growth)}%**")

# ---------- Descarga a Excel ----------
xls_bytes = build_excel(simple_df, summary_df)
st.download_button(
    label="💾 Descargar resultados en Excel",
    data=xls_bytes,
    file_name="Resultados_Pronostico_Regresiones.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
st.caption("Tip: ajusta los valores del año 'p' en la sección de supuestos y observa cómo cambian los resultados en tiempo real.")
