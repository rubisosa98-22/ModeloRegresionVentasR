# Pronóstico de Ventas con Regresión (Streamlit App)

Esta aplicación permite pronosticar ventas usando modelos de regresión lineal simple y un promedio ponderado basado en R² de cada variable macroeconómica.

## 🚀 Funcionalidades
- Carga de archivo Excel/CSV con series históricas.
- Cálculo de crecimiento de ventas histórico (%).
- Estimación de regresiones lineales simples (PIB, Desempleo, Tipo de Cambio, Inflación).
- Cálculo de correlación, pendiente, intercepto, R².
- Pronóstico de crecimiento y ventas absolutas para un periodo proyectado.
- Promedio ponderado de pronósticos con R² como peso.
- Resultados presentados en tabla dentro de Streamlit.

## 📂 Formato esperado del archivo
Columnas requeridas (o equivalentes que la app reconoce automáticamente):
- `Año` (ej. 2000, 2001, 2002...)
- `PIB` (%)
- `Desemp` (%)
- `TipoCam` (%)
- `Inflación` (%)
- `Ventas` (valores absolutos)

Ejemplo:

| Año | PIB  | Desemp | TipoCam | Inflación | Ventas  |
|----:|-----:|-------:|--------:|----------:|--------:|
| 2000| 2.31 |   3.50 |   -3.47 |      3.33 |  4824.00|
| 2001| 4.50 |   4.40 |    0.04 |      4.45 |  7194.00|
| 2002| 2.29 |   4.30 |    0.24 |      3.76 | 10812.00|
| 2003| 0.43 |   3.80 |    1.93 |      6.53 | 14394.00|

## ▶️ Ejecución
1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```

3. Abre el navegador en la URL que te indique Streamlit (por defecto `http://localhost:8501`).

## 📊 Notas
- Los valores macroeconómicos deben estar en **%**.
- Las ventas deben estar en **valor absoluto**.
- El pronóstico se hace para un periodo proyectado (ej. 2004p).

---

Desarrollado para análisis de planeación financiera y pronóstico con modelos econométricos sencillos.
