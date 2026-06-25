# Metodología

Se adopta **CRISP-DM adaptado a investigación experimental con estudio de ablación**.

La comparación principal es:

- E0 = CNN base.
- E1 = CNN + segmentación.
- E2 = CNN + textura explícita.
- E3 = CNN + segmentación + textura explícita.

La métrica principal será Macro-F1 y las métricas secundarias serán accuracy, precision macro, recall macro, matriz de confusión, tiempo de inferencia, parámetros y tamaño del modelo.
