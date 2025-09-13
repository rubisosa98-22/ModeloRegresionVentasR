import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from typing import Tuple, Dict
from statsmodels.api import OLS, add_constant
from scipy.stats import pearsonr

st.set_page_config(page_title="Pronóstico de Ventas por Regresión", layout="wide")
st.title("Pronóstico de Ventas con Modelos de Regresión")
st.caption("Sube una tabla histórica (CSV/Excel) con **ventas** y variables macroeconómicas. Años en formato `YYYY` y el año a pronosticar como `YYYYp`.")

# -------------------- Helpers --------------------
YEAR_COL_RE = re.compile(r"^\d{4}p?$")

def is_wide(df: pd.DataFrame) -> bool:
    year_like_cols = [c for c in df.columns if YEAR_COL_RE.match(str(c))]
    return len(year_like_cols) >= 2

def to_wide(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Normalize uploaded data to wide format:
      - Index = variable name
      - Columns = years (YYYY or YYYYP)
    Accepts two formats:
      (A) Wide: first column is variable name; year columns across
      (B) Long: columns include ['variable','year','value'] (case-insensitive)
    Returns: (wide_df, variable_col_name)
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # Try LONG format detection
    lower_cols = [c.lower() for c in d.columns]
    if set(["variable","year","value"]).issubset(set(lower_cols)):
        # Map to canonical names
        mapping = {c: c.lower() for c in d.columns}
        d = d.rename(columns=mapping)
        d["variable"] = d["variable"].astype(str).str.strip()
        d["year"] = d["year"].astype(str).str.strip()
        # Keep only rows with year pattern
        d = d[d["year"].str.match(YEAR_COL_RE)]
        wide = d.pivot_table(index="variable", columns="year", values="value", aggfunc="first")
        wide = wide.sort_index(axis=1)
        return wide, "variable"

    # Otherwise assume WIDE
    # Heuristic: first non-year column is the variable name
    non_year_cols = [c for c in d.columns if not YEAR_COL_RE.match(str(c))]
    if not non_year_cols:
        raise ValueError("No encontré columna de 'variable'. Agrega una columna inicial con los nombres de variables.")
    var_col = non_year_cols[0]
    wide = d.set_index(var_col)
    # Keep only year-like columns
    year_cols = [c for c in wide.columns if YEAR_COL_RE.match(str(c))]
    if not year_cols:
        raise ValueError("No encontré columnas de años (YYYY o YYYYP).")
    wide = wide[year_cols]
    # Try to coerce to numeric
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide = wide.sort_index(axis=1)
    return wide, var_col

def detect_years(cols) -> Tuple[list, str]:
    years = [c for c in cols if re.match(r"^\d{4}$", str(c))]
    years_p = [c for c in cols if re.match(r"^\d{4}p$", str(c))]
    years = sorted(years)
    forecast_col = years_p[0] if years_p else None
    return years, forecast_col

def compute_sales_growth(series: pd.Series) -> pd.Series:
    """ y_t = (Sales_t / Sales_{t-1}) - 1. Index are years (YYYY). """
    s = series.copy().astype(float)
    s = s.dropna()
    s = s.sort_index()
    growth = s.pct_change()
    return growth

def fmt_pct(x, dec=2):
    if pd.isna(x): return np.nan
    return float(np.round(100.0 * x, dec))

def fmt_abs(x, dec=2):
    if pd.isna(x): return np.nan
    return float(np.round(x, dec))

# -------------------- UI: Upload --------------------
up = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])
if not up:
    st.info("El archivo puede estar en formato **wide** (filas=variables, columnas=años) o **long** (columnas: variable, year, value).")
    st.stop()

# Read file
if up.name.lower().endswith(".csv"):
    raw = pd.read_csv(up)
else:
    raw = pd.read_excel(up)

try:
    wide, var_col_name = to_wide(raw)
except Exception as e:
    st.error(f"Error normalizando datos: {e}")
    st.stop()

years, forecast_col = detect_years(wide.columns)
if not years:
    st.error("No hay años históricos (YYYY).")
    st.stop()
if not forecast_col:
    st.warning("No encontré una columna de pronóstico (formato `YYYYp`). Puedes agregarla para obtener el pronóstico.")
    
variables = list(wide.index)

# -------------------- UI: Select Sales Row --------------------
# Try common labels
default_sales = None
for guess in ["Ventas", "ventas", "Sales", "sales", "Ingresos", "ingresos", "Revenue", "revenue"]:
    if guess in variables:
        default_sales = guess
        break

sales_var = st.selectbox("Selecciona la fila que corresponde a **Ventas (en valor absoluto)**", options=variables, index=(variables.index(default_sales) if default_sales in variables else 0))

# Build growth target from historical years
sales_hist = wide.loc[sales_var, years]
y_growth = compute_sales_growth(sales_hist)  # indexed by years

if y_growth.dropna().shape[0] < 2:
    st.error("Se requieren al menos 2 puntos de crecimiento histórico para ajustar regresiones.")
    st.stop()

st.subheader("Variables Independientes")
exclude = st.multiselect("Excluir variables (opcional)", options=[v for v in variables if v != sales_var])
candidates = [v for v in variables if v not in exclude and v != sales_var]

# -------------------- Regresiones Simples --------------------
rows = []
preds = []
weights = []

for var in candidates:
    x_hist = wide.loc[var, years].astype(float)
    # Align on overlapping years where both y_growth and x exist
    common_years = sorted(list(set(y_growth.index.astype(str)).intersection(set(x_hist.index.astype(str)))))
    if len(common_years) < 2:
        continue
    y = y_growth.loc[common_years].astype(float).values
    x = x_hist.loc[common_years].astype(float).values

    # Pearson
    try:
        r, _ = pearsonr(x, y)
    except Exception:
        r = np.nan

    # OLS y = a + b x
    X = add_constant(x)
    model = OLS(y, X, missing="drop").fit()
    a = model.params[0]  # intercept
    b = model.params[1]  # slope
    r2 = max(0.0, min(1.0, model.rsquared))  # clip to [0,1]

    # Prediction for forecast year (growth)
    pred_growth = np.nan
    if forecast_col is not None and forecast_col in wide.columns:
        x_p = wide.loc[var, forecast_col]
        if pd.notna(x_p):
            pred_growth = a + b * float(x_p)

    # Absolute sales forecast (using last actual sales)
    forecast_abs = np.nan
    if not np.isnan(pred_growth):
        last_year = max(y_growth.dropna().index.astype(int))
        last_sales = sales_hist[str(last_year)]
        if pd.notna(last_sales):
            forecast_abs = float(last_sales) * (1.0 + float(pred_growth))

    # Weight from R^2
    rows.append({
        "Variable": var,
        "Abs(Pearson)": abs(r) if pd.notna(r) else np.nan,
        "Coef_x": abs(b) if pd.notna(b) else np.nan,
        "Intercepto": abs(a) if pd.notna(a) else np.nan,
        "R2": r2,
        "Tasa_g_pronosticada": pred_growth,
        "Pronostico_abs": forecast_abs,
    })
    preds.append(pred_growth)
    weights.append(r2)

# Build results table
res_df = pd.DataFrame(rows)
if res_df.empty:
    st.error("No se pudieron ajustar regresiones simples válidas con las variables seleccionadas.")
    st.stop()

# Normalize weights over variables with a prediction available
valid_mask = res_df["Tasa_g_pronosticada"].notna() & res_df["R2"].notna()
sum_r2 = res_df.loc[valid_mask, "R2"].sum()
res_df["Ponderador"] = np.where(valid_mask, res_df["R2"] / sum_r2 if sum_r2 > 0 else 0.0, np.nan)
res_df["Tasa_g_pronosticada_ponderada"] = res_df["Ponderador"] * res_df["Tasa_g_pronosticada"]

# Weighted average growth
weighted_avg_growth = res_df["Tasa_g_pronosticada_ponderada"].sum(skipna=True)

# Final absolute forecast using weighted avg
final_forecast_abs = np.nan
if not np.isnan(weighted_avg_growth):
    last_year = max(y_growth.dropna().index.astype(int))
    last_sales = sales_hist[str(last_year)]
    if pd.notna(last_sales):
        final_forecast_abs = float(last_sales) * (1.0 + float(weighted_avg_growth))

# Formatting for display
disp = res_df.copy()
disp["Abs(Pearson)"] = disp["Abs(Pearson)"].apply(lambda x: np.round(abs(x), 2) if pd.notna(x) else np.nan)
disp["Coef_x"] = disp["Coef_x"].apply(lambda x: np.round(abs(x), 4) if pd.notna(x) else np.nan)
disp["Intercepto"] = disp["Intercepto"].apply(lambda x: np.round(abs(x), 4) if pd.notna(x) else np.nan)
disp["R2 (%)"] = disp["R2"].apply(lambda x: np.round(100.0 * x, 0) if pd.notna(x) else np.nan)
disp["Tasa_g_pronosticada (%)"] = disp["Tasa_g_pronosticada"].apply(lambda x: np.round(100.0 * x, 2) if pd.notna(x) else np.nan)
disp["Pronostico_abs"] = disp["Pronostico_abs"].apply(lambda x: np.round(x, 2) if pd.notna(x) else np.nan)
disp["Ponderador (%)"] = disp["Ponderador"].apply(lambda x: np.round(100.0 * x, 2) if pd.notna(x) else np.nan)
disp["Tasa_g_pronosticada_ponderada (%)"] = disp["Tasa_g_pronosticada_ponderada"].apply(lambda x: np.round(100.0 * x, 2) if pd.notna(x) else np.nan)

disp = disp[["Variable","Abs(Pearson)","Coef_x","Intercepto","R2 (%)","Tasa_g_pronosticada (%)","Pronostico_abs","Ponderador (%)","Tasa_g_pronosticada_ponderada (%)"]]

st.subheader("Resultados por Variable (Regresiones Simples)")
st.dataframe(disp, use_container_width=True)

# Summary
st.subheader("Resumen Ponderado")
col1, col2 = st.columns(2)
with col1:
    st.metric("Tasa de crecimiento ponderada (%)", value=(np.round(100.0 * weighted_avg_growth, 2) if not np.isnan(weighted_avg_growth) else "NA"))
with col2:
    st.metric(f"Pronóstico de ventas {forecast_col if forecast_col else '(sin año p)'}", value=(np.round(final_forecast_abs, 2) if not np.isnan(final_forecast_abs) else "NA"))

# Sales series table with forecast column
st.subheader("Serie de Ventas (histórico y pronóstico)")
sales_table = pd.DataFrame({
    **{str(y): [sales_hist.get(str(y), np.nan)] for y in years}
})
if not np.isnan(final_forecast_abs) and forecast_col:
    sales_table[forecast_col] = [final_forecast_abs]
sales_table.index = [sales_var]
st.dataframe(sales_table, use_container_width=True)

st.markdown("""
**Notas y validaciones:**
- Se calcula el objetivo como crecimiento anual de ventas: \\(g_t = \\frac{Ventas_t}{Ventas_{t-1}} - 1\\).
- Cada regresión simple se ajusta con los años donde hay datos tanto de crecimiento como de la variable (mínimo 2 observaciones).
- La predicción para `YYYYp` usa el valor `x_{YYYYp}` de cada variable (si está presente).
- El **ponderador** es el R² de cada regresión, normalizado para sumar 1 (solo entre variables con predicción disponible).
- El pronóstico final es \\(Ventas_{\\text{último año}} \\times (1 + \\bar g_{ponderado})\\).
""")