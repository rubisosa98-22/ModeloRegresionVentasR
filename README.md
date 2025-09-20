# Pronóstico de Ventas (Regresiones + Ponderado por R²)

App de **Streamlit** que:
- Carga un Excel con **años en la fila 1** (Año | 2000 | 2001 | … | XXXXp) y **variables en la columna A** (PIB, Desempleo, Tipo de Cambio, Inflación, Ventas).
- Detecta automáticamente años **históricos (XXXX)** y el **año de pronóstico (XXXXp)**.
- Ejecuta **regresiones simples** (Ventas vs cada macro) y un **modelo ponderado por R²**.
- Permite **editar dinámicamente** los valores del año `p` para las variables macro.
- Exporta **Excel** con 3 hojas: *Regresiones_Simples*, *Ponderado_R2* y *Resumen*.

## Estructura esperada del Excel
```
Año            2000    2001    2002    2003    2004p
PIB            0.0231  0.0450  0.0229  0.0043  0.0250
Desempleo      0.0350  0.0440  0.0430  0.0380  0.0390
Tipo de Cambio -0.0347 0.0004  0.0024  0.0193  0.0028
Inflación      0.0333  0.0405  0.0376  0.0653  0.0480
Ventas         0.4418  0.4913  0.5029  0.3313
```

- Valores **YoY** en **decimales** (0.025 = 2.5%).
- La columna con sufijo **`p`** es el **año de pronóstico**.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requisitos
- Python 3.9+ recomendado

## Notas
- Si no aparece alguna macro (p. ej. *Tipo de Cambio*), la app excluirá esa regresión y recalculará los pesos con las variables disponibles.
- Si hay muy pocas observaciones (históricos), las métricas pueden ser inestables; la app igual reporta el resultado con advertencias implícitas (n bajo).
