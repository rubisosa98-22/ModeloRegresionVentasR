# Pronóstico de Ventas por Regresión (Streamlit)

Este proyecto implementa un flujo reproducible para pronosticar el **crecimiento de ventas** a partir de **variables macroeconómicas**, calculando:
- Correlaciones de Pearson
- Regresiones lineales simples (pendiente, intercepto, R²)
- Predicción de crecimiento por variable para `YYYYp`
- **Regresión múltiple ponderada** (promedio ponderado por R² de los pronósticos simples)
- Conversión a pronóstico de ventas en valor absoluto

## Entrada esperada

Acepta dos formatos:

1) **Wide** (filas = variables, columnas = años):
```
Variable,2019,2020,2021,2022,2023,2024p
Ventas,1000,900,1100,1200,1300,
PIB,2.0,-8.0,5.0,3.0,2.6,2.2
Inflacion,2.8,3.4,5.7,7.9,4.8,4.0
TipoCambio,19.2,21.5,20.1,19.3,17.8,18.2
```
> El año a pronosticar se indica como `YYYYp` (ej. `2024p`). La fila **Ventas** debe contener **valores absolutos**.

2) **Long** (columnas = `variable, year, value`):
```
variable,year,value
Ventas,2019,1000
Ventas,2020,900
...
Inflacion,2024p,4.0
```

## Cálculo del objetivo (crecimiento de ventas)

Para años históricos:
\\[
g_t \;=\; \frac{Ventas_t}{Ventas_{t-1}} - 1
\\]

Para `YYYYp`, se predice \\( \hat g_{p} \\) por variable, usando una regresión simple \\( g_t = a + b \\, x_t \\) y el valor \\( x_{p} \\).  
El **promedio ponderado** usa pesos \\( w_i = \\frac{R_i^2}{\sum_j R_j^2} \\).  
El pronóstico final de ventas:  
\\[
\widehat{Ventas}_{p} \;=\; Ventas_{\text{último año}} \times (1 + \bar g)
\\]

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Verificaciones y supuestos

- Mínimo **2 observaciones** para ajustar cada regresión.
- Se alinean años para cada variable (solo intersección donde hay datos).
- Si falta `YYYYp` para alguna variable, esa variable **no** participa en el promedio ponderado.
- Los pesos se normalizan para sumar 1 (entre variables con predicción disponible).

## Licencia

MIT