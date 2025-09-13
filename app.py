import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict
from statsmodels.api import OLS, add_constant

st.set_page_config(page_title="Pronóstico por Regresión", layout="wide")
st.title("Pronóstico de Ventas: Regresiones Simples y Múltiple")
st.caption("Sube un CSV/Excel con filas=variables y columnas=años (YYYY) y opcionalmente YYYYP; o formato long: variable, year, value.")

YEAR_RE = re.compile(r"^\d{4}p?$")
HIST_RE = re.compile(r"^\d{4}$")
FORE_RE = re.compile(r"^\d{4}p$")

def to_wide(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Convierte entrada a formato wide: index=variable, cols=años (YYYY / YYYYP)."""
    # Normaliza encabezados
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in df.columns]

    # Formato LONG (variable, year, value)
    if set(["variable","year","value"]).issubset(lower):
        mapping = {c: c.lower() for c in df.columns}
        df = df.rename(columns=mapping)
        df["variable"] = df["variable"].astype(str).str.strip()
        df["year"] = df["year"].astype(str).str.strip()
        df = df[df["year"].str.match(YEAR_RE)]
        wide = df.pivot_table(index="variable", columns="year", values="value", aggfunc="first" )
        wide = wide.apply(pd.to_numeric, errors="coerce").sort_index(axis=1)
        return wide, "variable"

    # Formato WIDE (primera col = variable, resto años)
    non_year_cols = [c for c in df.columns if not YEAR_RE.match(str(c))]
    if not non_year_cols:
        raise ValueError("No encontré una columna de nombres de variable. Agrega una columna inicial con las variables.")
    var_col = non_year_cols[0]
    wide = df.set_index(var_col)
    year_cols = [c for c in wide.columns if YEAR_RE.match(str(c))]
    if not year_cols:
        raise ValueError("No hay columnas de años (YYYY o YYYYP)." )
    wide = wide[year_cols].apply(pd.to_numeric, errors="coerce").sort_index(axis=1)
    return wide, var_col

def compute_growth(sales_hist: pd.Series) -> pd.Series:
    """g_t = Ventas_t / Ventas_{t-1} - 1, con años históricos ordenados."""
    s = sales_hist.dropna().astype(float)
    s.index = s.index.astype(str)
    s = s.sort_index()
    return s.pct_change().dropna()

def detect_sales_label(index_vals: List[str]) -> str:
    candidates = ["ventas", "sales", "ingresos", "revenue"]
    for v in index_vals:
        if str(v).strip().lower() in candidates:
            return v
    return None

def next_year_str(last_year_str: str) -> str:
    return f"{int(last_year_str)+1}p"

def format_pct(x, dec=2):
    return np.round(100.0*x, dec) if pd.notna(x) else np.nan

def format_abs(x, dec=2):
    return np.round(x, dec) if pd.notna(x) else np.nan

# --------------- UPLOAD ---------------
up = st.file_uploader("Carga CSV/Excel", type=["csv","xlsx","xls"])
if not up:
    st.stop()

try:
    raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    wide, var_col_name = to_wide(raw)
except Exception as e:
    st.error(f"Error leyendo/normalizando archivo: {e}")
    st.stop()

variables = list(wide.index.astype(str))
hist_years = [c for c in wide.columns if HIST_RE.match(str(c))]
fore_cols = [c for c in wide.columns if FORE_RE.match(str(c))]
fore_col = fore_cols[0] if fore_cols else None

if not hist_years:
    st.error("El archivo no contiene años históricos (formato YYYY)." )
    st.stop()

# --------------- DETECCIÓN DE VENTAS ---------------
default_sales = detect_sales_label(variables)
sales_var = st.selectbox(
    "Selecciona la fila de **Ventas (valores absolutos)**",
    options=variables,
    index=(variables.index(default_sales) if default_sales in variables else 0)
)

# --------------- ELECCIÓN DE AÑOS HISTÓRICOS ---------------
with st.expander("Años históricos a usar en el entrenamiento", expanded=True):
    years_selected = st.multiselect(
        "Elige los años históricos (YYYY) a considerar en el modelo",
        options=sorted(hist_years),
        default=sorted(hist_years)
    )
if len(years_selected) < 2:
    st.error("Selecciona al menos 2 años históricos para ajustar las regresiones.")
    st.stop()

# --------------- VARIABLES MACRO (X) ---------------
all_macros = [v for v in variables if v != sales_var]
with st.expander("Variables macroeconómicas (X) a incluir", expanded=True):
    macros_selected = st.multiselect(
        "Selecciona/deselecciona las variables a usar como regresores (X)",
        options=all_macros,
        default=all_macros
    )
if len(macros_selected) == 0:
    st.error("Selecciona al menos una variable macro para continuar.")
    st.stop()

# --------------- SERIE DE VENTAS Y CRECIMIENTO (Y) ---------------
sales_hist = wide.loc[sales_var, years_selected].astype(float)
g = compute_growth(sales_hist)
if g.shape[0] < 2:
    st.error("Con los años seleccionados, el crecimiento de ventas tiene menos de 2 observaciones.")
    st.stop()

last_hist_year = max(map(int, years_selected))
default_fore_col = fore_col if fore_col else next_year_str(str(last_hist_year))

# --------------- VALORES PARA AÑO A PRONOSTICAR ---------------
st.subheader("Año a pronosticar")
col_a, col_b = st.columns([1,2])
with col_a:
    target_year_label = st.text_input("Etiqueta del año a pronosticar", value=default_fore_col)
with col_b:
    if fore_col:
        st.info(f"Se encontró en el archivo una columna de pronóstico: **{fore_col}**. Se usará por defecto si tiene datos." )
    else:
        st.warning(f"No hay columna `YYYYp` en el archivo. Se pronosticará el año **{int(last_hist_year)+1}** (etiqueta sugerida: **{default_fore_col}**)." )

method = st.radio(
    "Valores de las macros para el año a pronosticar",
    options=["Usar último valor disponible (hold)", "Capturar manualmente"],
    index=0,
    horizontal=True
)

# Construye vector x_p (valores de X en el año a pronosticar)
x_p_series = pd.Series(index=macros_selected, dtype=float)
if fore_col and fore_col in wide.columns:
    x_p_series = wide.loc[macros_selected, fore_col].astype(float)

if method == "Usar último valor disponible (hold)":
    for v in macros_selected:
        if pd.isna(x_p_series.get(v, np.nan)):
            hist_vals = wide.loc[v, years_selected].dropna().astype(float)
            x_p_series[v] = hist_vals.iloc[-1] if hist_vals.size else np.nan
else:
    st.markdown("**Captura/edita los valores de las macros para el año a pronosticar:**")
    base_df = pd.DataFrame({"Variable": macros_selected, target_year_label: x_p_series.values})
    base_df = st.data_editor(base_df, num_rows="fixed", use_container_width=True)
    x_p_series = pd.Series(base_df[target_year_label].values, index=base_df["Variable"].values, dtype=float)

# --------------- REGRESIONES SIMPLES ---------------
st.header("Paso 1 · Regresiones lineales simples (una por variable)" )

rows = []
for var in macros_selected:
    x_hist = wide.loc[var, years_selected].astype(float)
    common = sorted(list(set(g.index.astype(str)).intersection(set(x_hist.index.astype(str)))))
    if len(common) < 2:
        continue

    y = g.loc[common].values
    x = x_hist.loc[common].values
    X = add_constant(x)
    model = OLS(y, X, missing="drop").fit()

    a, b = model.params
    r2 = max(0.0, min(1.0, float(model.rsquared)))

    pred_g = np.nan
    x_p = x_p_series.get(var, np.nan)
    if pd.notna(x_p):
        pred_g = a + b * float(x_p)

    forecast_abs = np.nan
    if pd.notna(pred_g):
        last_sales = sales_hist.dropna().iloc[-1]
        forecast_abs = float(last_sales) * (1.0 + float(pred_g))

    rows.append({
        "Variable": var,
        "Intercepto": a,
        "Pendiente": b,
        "R2": r2,
        "g_pronosticado": pred_g,
        "Ventas_pronosticadas": forecast_abs
    })

simple_df = pd.DataFrame(rows)
if simple_df.empty:
    st.error("No fue posible ajustar regresiones simples válidas con las selecciones actuales (revisa años y variables)." )
    st.stop()

disp = simple_df.copy()
disp["R2 (%)"] = disp["R2"].apply(lambda x: format_pct(x, 0))
disp["g_pronosticado (%)"] = disp["g_pronosticado"].apply(lambda x: format_pct(x, 2))
disp["Ventas_pronosticadas"] = disp["Ventas_pronosticadas"].apply(lambda x: format_abs(x, 2))
disp = disp[["Variable","Intercepto","Pendiente","R2 (%)","g_pronosticado (%)","Ventas_pronosticadas"]]

st.dataframe(disp, use_container_width=True)
st.caption("Notas: g = crecimiento anual de ventas. R² indica el ajuste de cada regresión simple. Pronósticos calculados con los valores de X seleccionados para el año objetivo." )

# --------------- REGRESIÓN MÚLTIPLE ---------------
st.header("Paso 2 · Regresión lineal múltiple (todas las X seleccionadas)" )

def build_panel_for_multiple(macros: List[str], y_growth: pd.Series, years_pool: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    valid_years = set(y_growth.index.astype(str))
    Xcols = {}
    for v in macros:
        x_hist = wide.loc[v, years_pool].astype(float)
        Xcols[v] = x_hist
        valid_years = valid_years.intersection(set(x_hist.dropna().index.astype(str)))

    valid_years = sorted(list(valid_years))
    if len(valid_years) < 2:
        return None, None

    y = y_growth.loc[valid_years].astype(float)
    X = pd.DataFrame({v: Xcols[v].loc[valid_years].values for v in macros}, index=valid_years)
    return X, y

X_mult, y_mult = build_panel_for_multiple(macros_selected, g, years_selected)

if X_mult is None or y_mult is None or X_mult.shape[0] < 2:
    st.warning("No hay suficientes observaciones simultáneas para ajustar una regresión múltiple con las selecciones actuales." )
else:
    n_obs, k_vars = X_mult.shape
    if n_obs <= k_vars:
        st.warning(f"No es posible ajustar OLS múltiple estable: observaciones={n_obs}, variables={k_vars}. Quita algunas variables o amplía años." )
    else:
        X_m = add_constant(X_mult.values)
        model_m = OLS(y_mult.values, X_m).fit()

        x_p_mult = x_p_series.reindex(macros_selected).astype(float)
        if x_p_mult.isna().any():
            st.error("Faltan valores de X para el año a pronosticar en la regresión múltiple. Completa los valores en la sección anterior." )
        else:
            a_m = model_m.params[0]
            b_m = pd.Series(model_m.params[1:], index=macros_selected)
            g_hat_mult = float(a_m + (b_m * x_p_mult).sum())

            last_sales = sales_hist.dropna().iloc[-1]
            sales_hat_mult = float(last_sales) * (1.0 + g_hat_mult)

            coef_df = pd.DataFrame({
                "Coeficiente": [a_m] + list(b_m.values),
            }, index=["Intercepto"] + list(b_m.index))
            coef_df["Coeficiente"] = coef_df["Coeficiente"].round(6)

            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("Observaciones (n)", n_obs)
            with met2:
                st.metric("Variables (k)", k_vars)
            with met3:
                st.metric("R² (%)", value=np.round(100*model_m.rsquared, 1))

            st.subheader("Coeficientes de la regresión múltiple" )
            st.dataframe(coef_df, use_container_width=True)

            st.subheader("Pronóstico con regresión múltiple" )
            colm1, colm2 = st.columns(2)
            with colm1:
                st.metric("g pronosticado (múltiple, %)", value=np.round(100*g_hat_mult, 2))
            with colm2:
                st.metric(f"Ventas pronosticadas {target_year_label}", value=np.round(sales_hat_mult, 2))

with st.expander("Validaciones y supuestos"):
    st.markdown("""
- **Objetivo**: crecimiento de ventas anual, \\( g_t = \\frac{Ventas_t}{Ventas_{t-1}} - 1 \\).
- **Simples**: una OLS por variable (g ~ X), usando años con datos para ambas series.
- **Múltiple**: OLS con todas las X seleccionadas; requiere **n > k** y al menos 2 observaciones simultáneas.
- **Año a pronosticar**: si no hay `YYYYp` en el archivo, se define el inmediato siguiente al último histórico.  
  - Puedes **usar último valor** de cada macro (hold) o **capturarlo manualmente**.
- Pronóstico de ventas = Ventas del último año histórico × (1 + g pronosticado).
""")
