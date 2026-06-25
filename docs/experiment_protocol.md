# Protocolo experimental

Reglas de control:

1. Usar el mismo dataset para todos los setups.
2. Usar el mismo split train/val/test.
3. Mantener semilla fija.
4. Mantener tamaño de imagen 224x224.
5. Mantener MobileNetV2 como CNN base.
6. Evaluar todos los modelos en el mismo test set.
7. No tocar test durante entrenamiento.
8. Reportar Macro-F1 como métrica principal.
