import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Tuple, List
from statsmodels.api import OLS, add_constant

# ---------------- Basic Page Setup ----------------
st.set_page_config(page_title="Pronóstico por Regresión (con supuestos de macros)", layout="wide")
st.title("Pronóstico de Ventas con Supuestos de Macros (niveles o crecimiento como Y)")
st.caption("Sube CSV/Excel en formato wide (filas=variables, columnas=años YYYY y opcional YYYYP) o long (variable, year, value). Luego elige años históricos, variables X y define supuestos para el año objetivo.")

# ---------------- Regex for Year Columns ----------------
YEAR_RE = re.compile(r"^\d{4}p?$")
HIST_RE = re.compile(r"^\d{4}$")
FORE_RE = re.compile(r"^\d{4}p$")

# ---------------- Helpers ----------------
def to_wide(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    # Convert any input to wide format: index=variable, columns=years (YYYY/YYYYP)
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in df.columns]

    # LONG format (variable, year, value)
    if set(["variable","year","value"]).issubset(lower):
        mapping = {c: c.lower() for c in df.columns}
        df = df.rename(columns=mapping)
        df["variable"] = df["variable"].astype(str).str.strip()
        df["year"] = df["year"].astype(str).str.strip()
        df = df[df["year"].str.match(YEAR_RE)]
        wide = df.pivot_table(index="variable", columns="year", values="value", aggfunc="first")
        wide = wide.apply(pd.to_numeric, errors="coerce").sort_index(axis=1)
        return wide, "variable"

    # WIDE format: first non-year column is variable name
    non_year_cols = [c for c in df.columns if not YEAR_RE.match(str(c))]
    if not non_year_cols:
        raise ValueError("No encontré una columna de nombres de variables.")
    var_col = non_year_cols[0]
    wide = df.set_index(var_col)
    year_cols = [c for c in wide.columns if YEAR_RE.match(str(c))]
    if not year_cols:
        raise ValueError("No hay columnas de años (YYYY o YYYYP).")
    wide = wide[year_cols].apply(pd.to_numeric, errors="coerce").sort_index(axis=1)
    return wide, var_col

def compute_growth_from_levels(level_series: pd.Series) -> pd.Series:
    # g_t = V_t / V_{t-1} - 1
    s = level_series.dropna().astype(float)
    s.index = s.index.astype(str)
    s = s.sort_index()
    return s.pct_change().dropna()

def detect_sales_label(index_vals: List[str]) -> str:
    candidates = ["ventas", "sales", "ingresos", "revenue"]
    for v in index_vals:
        if str(v).strip().lower() in candidates:
            return v
    return None

def infer_sales_mode(series: pd.Series) -> str:
    # Heuristic: decide if sales row contains levels or growth
    s = series.dropna().astype(float).values
    if s.size == 0:
        return "levels"
    frac_small = np.mean((s > -1.2) & (s < 1.2))
    mean_abs = np.mean(np.abs(s))
    if frac_small > 0.7 and mean_abs < 0.5:
        return "growth_dec"  # decimals, e.g., 0.05 = 5%
    if np.all((s > -200) & (s < 200)) and mean_abs >= 0.5 and mean_abs <= 80:
        return "growth_pct"  # percentages, e.g., 5 = 5%
    return "levels"

def format_pct(x, dec=2):
    return np.round(100.0*x, dec) if pd.notna(x) else np.nan

def format_abs(x, dec=2):
    return np.round(x, dec) if pd.notna(x) else np.nan

# ---------------- Upload ----------------
up = st.file_uploader("Carga CSV/Excel principal", type=["csv","xlsx","xls"])
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
    st.error("El archivo no contiene años históricos (YYYY).")
    st.stop()

# ---------------- Select Sales Row ----------------
default_sales = detect_sales_label(variables)
sales_var = st.selectbox(
    "Selecciona la fila objetivo (Ventas). Puede estar en NIVELES o CRECIMIENTO.",
    options=variables,
    index=(variables.index(default_sales) if default_sales in variables else 0)
)

with st.expander("Interpretación de la fila de Ventas (Y)", expanded=True):
    sales_hist_all = wide.loc[sales_var, hist_years].astype(float)
    inferred = infer_sales_mode(sales_hist_all)
    mode = st.radio(
        "¿Qué contiene la fila de Ventas?",
        options=[
            "Niveles absolutos (ej. 1000, 1200, 1300)",
            "Crecimiento en decimales (ej. 0.05 = 5%)",
            "Crecimiento en porcentaje (ej. 5 = 5%)"
        ],
        index={"levels":0, "growth_dec":1, "growth_pct":2}[inferred],
        horizontal=False
    )
    st.caption("Sugerencia automática: " + ({"levels":"Niveles","growth_dec":"Crecimiento (decimal)","growth_pct":"Crecimiento (%)"}[inferred]))

# ---------------- Choose Historical Years ----------------
with st.expander("Años históricos para entrenar", expanded=True):
    years_selected = st.multiselect(
        "Elige los años (YYYY) a considerar",
        options=sorted(hist_years),
        default=sorted(hist_years)
    )
if len(years_selected) < 2:
    st.error("Selecciona al menos 2 años históricos.")
    st.stop()

# ---------------- Select Macro Variables (X) ----------------
all_macros = [v for v in variables if v != sales_var]
with st.expander("Variables macro (X) a incluir", expanded=True):
    macros_selected = st.multiselect(
        "Activa/desactiva las variables explicativas",
        options=all_macros,
        default=all_macros
    )
if len(macros_selected) == 0:
    st.error("Selecciona al menos una variable macro.")
    st.stop()

# ---------------- Build Y (growth) ----------------
sales_hist = wide.loc[sales_var, years_selected].astype(float)
if "Niveles absolutos" in mode:
    g = compute_growth_from_levels(sales_hist)
elif "Crecimiento en decimales" in mode:
    g = sales_hist.dropna().astype(float).copy()
    g.index = g.index.astype(str)
    g = g.sort_index()
else:
    g = (sales_hist.dropna().astype(float) / 100.0).copy()
    g.index = g.index.astype(str)
    g = g.sort_index()

if g.shape[0] < 2:
    st.error("Con los años seleccionados, la serie Y (crecimiento) tiene menos de 2 observaciones.")
    st.stop()

# ---------------- Target Year ----------------
last_hist_year = max(map(int, years_selected))
default_fore_col = fore_col if fore_col else f"{int(last_hist_year)+1}p"

st.subheader("Año a pronosticar")
col_a, col_b = st.columns([1,2])
with col_a:
    target_year_label = st.text_input("Etiqueta del año objetivo", value=default_fore_col)
with col_b:
    if fore_col:
        st.info(f"Se encontró en el archivo una columna de pronóstico: **{fore_col}**. Si hay valores en esa columna, se usarán.")
    else:
        st.warning(f"No hay `YYYYp` en el archivo. Se pronosticará el año siguiente: **{int(last_hist_year)+1}** (etiqueta sugerida: **{default_fore_col}**).")

# ---------------- Assumptions for Target Year ----------------
st.subheader("Supuestos de variables macro para el año objetivo")
assumption_mode = st.radio(
    "¿Cómo quieres definir los valores de X para el año objetivo?",
    options=["Cargar archivo de supuestos", "Capturar manualmente", "Usar último valor histórico (hold)", "Tomar de columna YYYYP (si existe)"],
    index=0,
    horizontal=False
)

x_p_series = pd.Series(index=macros_selected, dtype=float)

if assumption_mode == "Cargar archivo de supuestos":
    sup = st.file_uploader("Carga CSV/Excel con supuestos (ver plantilla abajo)", type=["csv","xlsx","xls"], key="assumptions")
    if sup is not None:
        try:
            if sup.name.lower().endswith(".csv"):
                df_sup_raw = pd.read_csv(sup)
            else:
                df_sup_raw = pd.read_excel(sup)
            # Soportar formato wide (Variable + columna target) o long (variable, year, value) para target_year_label
            cols_lower = [c.lower() for c in df_sup_raw.columns]
            if set(["variable","year","value"]).issubset(cols_lower):
                m = {c: c.lower() for c in df_sup_raw.columns}
                df_sup = df_sup_raw.rename(columns=m)
                df_sup = df_sup[df_sup["year"].astype(str) == target_year_label]
                df_sup = df_sup[["variable","value"]].dropna()
                df_sup["variable"] = df_sup["variable"].astype(str).str.strip()
                x_p_series = pd.Series(df_sup["value"].values, index=df_sup["variable"].values, dtype=float)
            else:
                # Wide: primera col = Variable, una col cuyo nombre sea exactamente target_year_label
                df_sup = df_sup_raw.copy()
                df_sup.columns = [str(c).strip() for c in df_sup.columns]
                non_year_cols = [c for c in df_sup.columns if not re.match(r"^\d{4}p?$", str(c))]
                if len(non_year_cols) == 0:
                    st.error("El archivo de supuestos debe tener una columna para 'Variable'.")
                else:
                    varc = non_year_cols[0]
                    if target_year_label not in df_sup.columns:
                        st.error(f"No encuentro la columna '{target_year_label}' en el archivo de supuestos.")
                    else:
                        df_sup = df_sup[[varc, target_year_label]].dropna()
                        df_sup[varc] = df_sup[varc].astype(str).str.strip()
                        x_p_series = pd.Series(df_sup[target_year_label].values, index=df_sup[varc].values, dtype=float)
        except Exception as e:
            st.error(f"Error leyendo supuestos: {e}")
elif assumption_mode == "Capturar manualmente":
    st.markdown("**Ingresa los valores de las macros para el año objetivo:**")
    base_df = pd.DataFrame({"Variable": macros_selected, target_year_label: x_p_series.values})
    base_df = st.data_editor(base_df, num_rows="fixed", use_container_width=True)
    x_p_series = pd.Series(base_df[target_year_label].values, index=base_df["Variable"].values, dtype=float)
elif assumption_mode == "Usar último valor histórico (hold)":
    for v in macros_selected:
        hist_vals = wide.loc[v, years_selected].dropna().astype(float)
        x_p_series[v] = hist_vals.iloc[-1] if hist_vals.size else np.nan
else:  # Tomar de columna YYYYP (si existe)
    if fore_col and fore_col in wide.columns:
        x_p_series = wide.loc[macros_selected, fore_col].astype(float)
    else:
        st.warning("No existe columna YYYYP en el archivo principal; selecciona otro método de supuestos.")
        x_p_series = pd.Series(index=macros_selected, dtype=float)

# Confirmación visual de los supuestos cargados
st.subheader("Revisión de supuestos aplicados")
sup_check = pd.DataFrame({"Variable": macros_selected, target_year_label: x_p_series.reindex(macros_selected).values})
st.dataframe(sup_check, use_container_width=True)

# Validación
if x_p_series.reindex(macros_selected).isna().any():
    st.error("Faltan valores de X para el año objetivo. Completa los supuestos o cambia el método.")
    st.stop()

# ---------------- Simple Regressions ----------------
st.header("Paso 1 · Regresiones lineales simples (g ~ X)")

rows = []
for var in macros_selected:
    x_hist = wide.loc[var, years_selected].astype(float)
    # intersect on years with data in both y and x
    common = sorted(list(set(g.index.astype(str)).intersection(set(x_hist.dropna().index.astype(str)))))
    if len(common) < 2:
        continue
    y = g.loc[common].values
    x = x_hist.loc[common].values
    X = add_constant(x)
    model = OLS(y, X, missing="drop").fit()

    a, b = model.params
    r2 = max(0.0, min(1.0, float(model.rsquared)))

    pred_g = a + b * float(x_p_series.get(var))

    forecast_abs = np.nan
    if "Niveles absolutos" in mode:
        last_sales_level = sales_hist.dropna().iloc[-1]
        forecast_abs = float(last_sales_level) * (1.0 + float(pred_g))

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
    st.error("No fue posible ajustar regresiones simples válidas con las selecciones actuales (revisa años y variables).")
    st.stop()

disp = simple_df.copy()
disp["R2 (%)"] = disp["R2"].apply(lambda x: format_pct(x, 0))
disp["g_pronosticado (%)"] = disp["g_pronosticado"].apply(lambda x: format_pct(x, 2))
disp["Ventas_pronosticadas"] = disp["Ventas_pronosticadas"].apply(lambda x: format_abs(x, 2))
disp = disp[["Variable","Intercepto","Pendiente","R2 (%)","g_pronosticado (%)","Ventas_pronosticadas"]]

st.dataframe(disp, use_container_width=True)
st.caption("Notas: Y = crecimiento de ventas. Si Ventas venía en niveles, g se calculó con pct_change; si venía como crecimiento, se usó tal cual (decimal o %).")

# ---------------- Multiple Regression ----------------
st.header("Paso 2 · Regresión lineal múltiple (g ~ X1 + X2 + ...)")

def build_panel_for_multiple(macros: List[str], y_growth: pd.Series, years_pool: List[str]):
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
    st.warning("No hay suficientes observaciones simultáneas para OLS múltiple con las selecciones actuales.")
else:
    n_obs, k_vars = X_mult.shape
    if n_obs <= k_vars:
        st.warning(f"No es posible ajustar OLS múltiple estable: observaciones={n_obs}, variables={k_vars}. Quita algunas X o amplía años.")
    else:
        X_m = add_constant(X_mult.values)
        model_m = OLS(y_mult.values, X_m).fit()

        x_p_mult = x_p_series.reindex(macros_selected).astype(float)
        a_m = model_m.params[0]
        b_m = pd.Series(model_m.params[1:], index=macros_selected)
        g_hat_mult = float(a_m + (b_m * x_p_mult).sum())

        sales_hat_mult = np.nan
        if "Niveles absolutos" in mode:
            last_sales_level = sales_hist.dropna().iloc[-1]
            sales_hat_mult = float(last_sales_level) * (1.0 + g_hat_mult)

        coef_df = pd.DataFrame({"Coeficiente": [a_m] + list(b_m.values)},
                               index=["Intercepto"] + list(b_m.index)).round(6)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Observaciones (n)", n_obs)
        with c2: st.metric("Variables (k)", k_vars)
        with c3: st.metric("R² (%)", value=np.round(100*model_m.rsquared, 1))

        st.subheader("Coeficientes de la regresión múltiple")
        st.dataframe(coef_df, use_container_width=True)

        st.subheader("Pronóstico con regresión múltiple")
        m1, m2 = st.columns(2)
        with m1: st.metric("g pronosticado (múltiple, %)", value=np.round(100*g_hat_mult, 2))
        with m2: st.metric(f"Ventas pronosticadas {target_year_label}", value=(np.round(sales_hat_mult, 2) if pd.notna(sales_hat_mult) else "NA"))

# ---------------- Footer ----------------
with st.expander("Validaciones y supuestos"):
    st.markdown(
        "- **Objetivo Y = crecimiento de ventas (g).** Si Ventas venía en niveles, se convierte a g con g_t = V_t/V_{t-1} - 1. Si ya venía en crecimiento, se usa directo (decimal o %).\n"
        "- Puedes **cargar supuestos** en archivo (formato wide o long), **capturarlos manualmente**, **usar hold** o **tomarlos de YYYYP**.\n"
        "- **Simples**: OLS por variable (g ~ X). **Múltiple**: OLS con todas las X seleccionadas; requiere n > k y al menos 2 observaciones simultáneas.\n"
        "- **Ventas absolutas pronosticadas** solo se calculan si hay ventas en niveles para el último año (para multiplicar 1+g)."
    )

st.divider()
st.markdown("**Plantilla para supuestos (descargable):** dos opciones:")
st.markdown("1) **Formato long** con columnas: `variable, year, value` (usa `year = " + '"'+str(default_fore_col)+'"' + "`)")
st.markdown("2) **Formato wide** con columnas: `Variable, " + str(default_fore_col) + "`")

