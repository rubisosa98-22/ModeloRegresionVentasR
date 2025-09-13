
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm

st.set_page_config(page_title="Pronóstico con Regresiones (Ventas vs Macros)", layout="wide")
st.title("Pronóstico de Ventas con Regresiones — Simples, Múltiple y Promedio ponderado por R²")
st.caption("Sube tu archivo con variables macroeconómicas, crecimiento de ventas y ventas históricas.")

# ---------- Utilidades ----------
def _norm(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    rep = {
        "á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u","ñ":"n",
    }
    for k,v in rep.items():
        s = s.replace(k,v)
    s = re.sub(r"\s+", " ", s)
    return s

def find_row(df, patterns):
    # patterns: list of regex to match normalized content in first column
    col0 = df.columns[0]
    mask = pd.Series(False, index=df.index)
    for pat in patterns:
        mask |= df[col0].astype(str).map(_norm).str.contains(pat, regex=True, na=False)
    idx = df[mask].index.tolist()
    return idx

def get_period_cols(df):
    # Periods son columnas cuyo nombre contiene dígitos (años / periodos) y no es la primera columna
    cols = []
    for c in df.columns[1:]:
        if re.search(r"\d", str(c)):
            cols.append(c)
    # Mantener orden original
    return cols

def ols_simple(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model

def predict_row(model, value, name):
    # construir con mismos nombres de columnas (incluyendo const)
    cols = list(model.model.exog_names)  # ['const', name]
    row = pd.DataFrame([[1.0, value]], columns=cols)
    return float(model.predict(row)[0])

st.subheader("1) Carga de archivo")
up = st.file_uploader("Excel o CSV (la primera columna debe contener los nombres de variables/filas)", type=["xlsx","xls","csv"])
if not up:
    st.info("Estructura esperada (ejemplo de filas): PIB, Desempleo, TipoCambio, Inflación, Crecimiento de Ventas, Ventas")
    st.stop()

# Leer archivo
if up.name.lower().endswith(".csv"):
    df0 = pd.read_csv(up)
else:
    # Buscar primera hoja
    df0 = pd.read_excel(up, sheet_name=None)
    # elegir hoja con más columnas
    sheet = max(df0, key=lambda k: df0[k].shape[1])
    df0 = df0[sheet]

# Limpieza básica
df = df0.copy()
df.columns = [str(c).strip() for c in df.columns]
df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip()

period_cols = get_period_cols(df)
if not period_cols:
    st.error("No detecté columnas de periodos (años/meses). Asegúrate de que las columnas tengan nombres como 2000, 2001, 2002, 2003, 2004p, etc.")
    st.stop()

st.write("**Columnas de periodos detectadas:**", ", ".join(map(str, period_cols)))

# Detectar filas por patrón
row_pib = find_row(df, [r"\bpib\b", r"producto interno"])
row_des = find_row(df, [r"desempleo"])
row_tc  = find_row(df, [r"tipo.?cambio"])
row_inf = find_row(df, [r"inflacion"])
row_crec = find_row(df, [r"crec(imiento)?\s*de\s*ventas|yoy|variacion\s*ventas|growth"])
row_ventas = find_row(df, [r"^ventas\b"])

def pick_first(idx_list, nombre):
    if not idx_list:
        return None
    if len(idx_list) > 1:
        st.warning(f"Se encontraron múltiples filas para {nombre}; usaré la primera: {idx_list[0]}")
    return idx_list[0]

i_pib = pick_first(row_pib, "PIB")
i_des = pick_first(row_des, "Desempleo")
i_tc  = pick_first(row_tc, "Tipo de cambio")
i_inf = pick_first(row_inf, "Inflación")
i_cre = pick_first(row_crec, "Crecimiento de ventas")
i_ven = pick_first(row_ventas, "Ventas")

# Construir series
def row_to_series(i, name):
    s = df.loc[i, period_cols] if i is not None else pd.Series(index=period_cols, dtype=float)
    s = pd.to_numeric(s, errors="coerce")
    s.name = name
    return s

PIB = row_to_series(i_pib, "PIB")
Des = row_to_series(i_des, "Desempleo")
TC  = row_to_series(i_tc,  "TipoCambio")
Inf = row_to_series(i_inf, "Inflacion")
Crec = row_to_series(i_cre, "Crec_Ventas")
Ventas = row_to_series(i_ven, "Ventas")

# Si no hay crecimiento, calcularlo a partir de ventas
if Crec.isna().all() and not Ventas.isna().all():
    Crec = Ventas.pct_change()*100
    Crec.name = "Crec_Ventas"
    st.info("No hallé 'Crecimiento de Ventas'; lo calculé a partir de 'Ventas' (variación % YoY).")

# UI de selección de periodos históricos
st.subheader("2) Selección de periodos históricos y variables")
hist_periods = [c for c in period_cols if not str(c).lower().endswith("p")]
hist_ok = st.multiselect("Elige los periodos HISTÓRICOS para estimar (ej. 2000–2003):", hist_periods, default=hist_periods)
if len(hist_ok) < 2:
    st.error("Selecciona al menos dos periodos históricos.")
    st.stop()

# Variables disponibles
vars_found = {
    "PIB": PIB,
    "Desempleo": Des,
    "TipoCambio": TC,
    "Inflacion": Inf,
}
vars_presentes = [k for k,v in vars_found.items() if not v[hist_ok].isna().all()]
sel_vars = st.multiselect("Variables macro a incluir (regresión simple y múltiple):", vars_presentes, default=vars_presentes)

if not sel_vars:
    st.error("Selecciona al menos una variable macro.")
    st.stop()

# Periodo a pronosticar (uno de los periodos, típicamente con 'p') o escribir uno nuevo
cand_fore = [c for c in period_cols if str(c) not in hist_ok]
default_fore = cand_fore[0] if cand_fore else "Pronostico"
target_label = st.text_input("Etiqueta del periodo a pronosticar", value=str(default_fore))

# Supuestos para el periodo objetivo (si la celda existe la tomamos como valor inicial)
st.subheader("3) Supuestos macroeconómicos para el periodo a pronosticar")
supuestos = {}
cols = st.columns(len(sel_vars))
for j, name in enumerate(sel_vars):
    series = vars_found[name]
    init = series.get(target_label, np.nan)
    supuestos[name] = cols[j].number_input(f"{name} ({target_label})", value=float(init) if pd.notna(init) else 0.0)

# Construir dataset histórico (Y y X)
Y = Crec[hist_ok]
X_dict = {name: vars_found[name][hist_ok] for name in sel_vars}
X_df = pd.DataFrame(X_dict)

# Eliminar filas con NaN
data = pd.concat([Y.rename("Crec_Ventas"), X_df], axis=1).dropna()
if data.shape[0] < 2:
    st.error("Muy pocos datos válidos después de limpiar NaN. Revisa entradas.")
    st.stop()

st.write("**Datos usados (históricos, post-limpieza):**")
st.dataframe(data, use_container_width=True)

# ------------- Regresiones simples -------------
st.subheader("4) Regresiones simples (una X contra Crec. Ventas)")
rows = []
preds = []
r2s = []

for name in sel_vars:
    model = ols_simple(data["Crec_Ventas"], data[[name]])
    r = np.corrcoef(data[name], data["Crec_Ventas"])[0,1]
    b1 = model.params[name]
    b0 = model.params["const"]
    r2 = model.rsquared
    # Pronóstico con supuesto
    pred = predict_row(model, supuestos[name], name)
    rows.append({
        "Variable": name,
        "Correlacion (r)": r,
        "Pendiente (β1)": b1,
        "Intercepto (β0)": b0,
        "R2": r2,
        f"Pronóstico {target_label} (Crec_Ventas %)": pred
    })
    preds.append(pred); r2s.append(r2)

tabla_simple = pd.DataFrame(rows).set_index("Variable")
st.dataframe(tabla_simple.round(4), use_container_width=True)

# Promedio ponderado por R2
r2s = np.array(r2s)
if r2s.sum() > 0:
    pesos = r2s / r2s.sum()
else:
    pesos = np.ones_like(r2s)/len(r2s)
prom_ponderado = float(np.dot(pesos, np.array(preds)))

tabla_pesos = tabla_simple.copy()
tabla_pesos["Peso_R2"] = pesos
tabla_pesos["Aporte"] = tabla_pesos["Peso_R2"] * tabla_pesos[f"Pronóstico {target_label} (Crec_Ventas %)"]
st.markdown("**Promedio ponderado por R² (pronóstico de crecimiento):** {:.2f} %".format(prom_ponderado))
st.dataframe(tabla_pesos.round(4), use_container_width=True)

# ------------- Regresión múltiple -------------
st.subheader("5) Regresión múltiple (todas las X seleccionadas)")
X_multi = sm.add_constant(data[sel_vars])
model_multi = sm.OLS(data["Crec_Ventas"], X_multi).fit()

x_fore = pd.DataFrame({v: [supuestos[v]] for v in sel_vars})
x_fore = sm.add_constant(x_fore, has_constant="add")
pred_multi = float(model_multi.predict(x_fore)[0])

st.write("**R² múltiple:** {:.4f}".format(model_multi.rsquared))
st.write("**Pronóstico {} (Crec_Ventas %):** {:.2f}".format(target_label, pred_multi))
with st.expander("Ver coeficientes de la múltiple"):
    coef = pd.DataFrame({
        "Parametro": model_multi.params.index,
        "Valor": model_multi.params.values
    })
    st.dataframe(coef, use_container_width=True)
with st.expander("Resumen statsmodels (avanzado)"):
    st.text(model_multi.summary())

# ------------- Proyección de Ventas absolutas -------------
st.subheader("6) Proyección de Ventas absolutas")
if Ventas[hist_ok].dropna().empty:
    st.info("No encontré ventas históricas suficientes para proyectar niveles. Puedo proyectar solo el crecimiento.")
else:
    base_year = st.selectbox("Elige el año base para multiplicar (último histórico por defecto)", options=list(Ventas[hist_ok].dropna().index), index=len(Ventas[hist_ok].dropna())-1)
    ventas_base = float(Ventas[base_year])
    ventas_ponderado = ventas_base * (1 + prom_ponderado/100.0)
    ventas_multi = ventas_base * (1 + pred_multi/100.0)

    out = pd.DataFrame({
        "Elemento": [f"Ventas {target_label} (promedio ponderado R²)", f"Ventas {target_label} (regresión múltiple)"],
        "Valor": [round(ventas_ponderado,2), round(ventas_multi,2)],
        "Nota": [f"Base = Ventas {base_year} = {ventas_base:,.2f}", f"Base = Ventas {base_year} = {ventas_base:,.2f}"]
    })
    st.dataframe(out, use_container_width=True)

    # Exportar a Excel
    if st.button("Descargar resultados a Excel"):
        with pd.ExcelWriter("resultados_regresiones.xlsx", engine="xlsxwriter") as writer:
            data.to_excel(writer, sheet_name="datos_usados")
            tabla_simple.round(4).to_excel(writer, sheet_name="simples")
            tabla_pesos.round(4).to_excel(writer, sheet_name="simples_ponderado")
            coef.to_excel(writer, sheet_name="multiple_coef")
            pd.DataFrame({
                "Pronostico_crec_ponderado_%":[prom_ponderado],
                "Pronostico_crec_multiple_%":[pred_multi],
                "Ventas_base":[ventas_base],
                f"Ventas_{target_label}_ponderado":[ventas_ponderado],
                f"Ventas_{target_label}_multiple":[ventas_multi],
            }).to_excel(writer, sheet_name="resumen")
        with open("resultados_regresiones.xlsx", "rb") as f:
            st.download_button("Guardar archivo Excel", data=f, file_name="resultados_regresiones.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.caption("Notas metodológicas: 1) Con muy pocas observaciones, la regresión múltiple puede sobreajustarse. 2) El promedio ponderado por R² suele ser más estable cuando hay pocos años.")
