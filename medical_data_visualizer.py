import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar los datos de medical_examination.csv
df = pd.read_csv('medical_examination.csv')

# 2. Agregar columna overweight (IMC = peso_kg / (altura_m)**2). Si IMC > 25 -> 1, sino 0
# Nota: La altura está en cm en el dataset, por lo que se divide entre 100 para pasar a metros
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalizar datos: Si cholesterol o gluc es 1 -> 0; si > 1 -> 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Función para dibujar el gráfico categórico
def draw_cat_plot():
    # 5. Crear DataFrame usando pd.melt con las variables especificadas
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Agrupar y reformatear los datos para dividirlos por cardio y mostrar conteos
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Crear el gráfico con sns.catplot()
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    # 8. Obtener la figura para la salida
    fig = g.fig

    # 9. No modificar las siguientes dos líneas
    fig.savefig('catplot.png')
    return fig


# 10. Función para dibujar el mapa de calor
def draw_heat_map():
    # 11. Limpiar datos en df_heat filtrando los segmentos incorrectos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calcular matriz de correlación
    corr = df_heat.corr()

    # 13. Generar máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar la figura de matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Graficar la matriz de correlación con sns.heatmap()
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        vmin=-0.16,
        vmax=0.32,
        cbar_kws={'shrink': 0.5},
        square=True,
        linewidths=0.5,
        ax=ax
    )

    # 16. No modificar las siguientes dos líneas
    fig.savefig('heatmap.png')
    return fig