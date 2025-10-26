# Evaluación de Modelos GAN para Señales EMG

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

Script completo para la evaluación cuantitativa y cualitativa de modelos Generativos Adversariales (GAN) aplicados a la reconstrucción de señales electromiográficas (EMG) de la base de datos Ninapro DB3.

## 📋 Descripción

Este proyecto implementa un sistema de evaluación automatizado para modelos GAN que generan señales EMG sintéticas. Incluye:

- **Evaluación cuantitativa** mediante un "Juez de Calidad" pre-entrenado
- **Evaluación cualitativa** mediante visualización comparativa
- **Análisis comparativo** de múltiples modelos a lo largo del entrenamiento
- **Métricas de calidad** de reconstrucción de señales

## 🚀 Características

- ✅ Evaluación automatizada de múltiples modelos GAN
- ✅ Soporte para diferentes estrategias de normalización (Z-score, Tanh)
- ✅ Clasificación de calidad con modelo pre-entrenado
- ✅ Generación de reportes detallados y comparativos
- ✅ Visualización de resultados
- ✅ Procesamiento por lotes eficiente

## 📁 Estructura del Proyecto

```
evaluar_gans_final.py
├── Configuración de modelos
├── Funciones de carga y preprocesamiento
├── Evaluación cuantitativa con Juez de Calidad
├── Generación de reportes comparativos
└── Visualización de resultados
```

## 🛠️ Instalación y Requisitos

### Dependencias

```bash
pip install tensorflow scipy matplotlib pandas scikit-learn numpy
```

### Requisitos del Sistema

- Python 3.7+
- TensorFlow 2.x
- SciPy
- scikit-learn
- pandas
- matplotlib

## ⚙️ Configuración

### Modelos a Evaluar

El script evalúa automáticamente los siguientes modelos:

- **Modelos de epoch 5 en 5** (del epoch 5 al 50)
- **Modelos de epoch consecutivos** (del 55 al 111)
- **Modelos adicionales** (generador_gan.h5 a generador_gan_6.h5)

### Parámetros Principales

```python
MODEL_PATH_JUEZ = 'juez_de_calidad.h5'
DATASET_PATH_DB3 = 'ninapro_db3_data'
SUBJECT_TO_EVALUATE = 11
NUM_EJEMPLOS_A_EVALUAR = 1000
UMBRAL_CALIDAD = 0.8
```

## 🎯 Uso

### Ejecución Básica

```bash
python evaluar_gans_final.py
```

### Flujo de Ejecución

1. **Carga del Juez de Calidad** - Modelo pre-entrenado para evaluar calidad
2. **Preparación de datos** - Carga y preprocesamiento de señales EMG reales
3. **Evaluación por modelo** - Generación y evaluación de señales sintéticas
4. **Generación de reportes** - Métricas comparativas y análisis detallado
5. **Visualización** - Comparación gráfica de reconstrucciones

## 📊 Métricas de Evaluación

### Métricas Principales

- **Índice de Éxito (%)**: Porcentaje de señales clasificadas como "sanas"
- **Reporte de Clasificación**: Precision, recall y F1-score
- **Comparativa Visual**: Reconstrucciones vs señal original

### Ejemplo de Salida

```
--- TABLA COMPARATIVA DE RENDIMIENTO DE GENERADORES ---
Modelo                          Índice de Éxito (%)
generador_gan_epoch_100.h5             92.50
generador_gan_epoch_99.h5              91.80
...
```

## 📈 Resultados y Visualización

El script genera:

1. **Tabla comparativa** ordenada por rendimiento
2. **Reporte detallado** del mejor modelo
3. **Gráfico comparativo** entre:
   - Señal original
   - Reconstrucción del modelo medio
   - Reconstrucción del mejor modelo

## 🔧 Personalización

### Modificar Modelos Evaluados

Editar la lista `MODELOS_GAN_A_EVALUAR`:

```python
MODELOS_GAN_A_EVALUAR = [
    'tu_modelo_1.h5',
    'tu_modelo_2.h5',
    # ... añadir más modelos
]
```

### Ajustar Parámetros

- `UMBRAL_CALIDAD`: Umbral para clasificación binaria (default: 0.8)
- `NUM_EJEMPLOS_A_EVALUAR`: Número de muestras para evaluación
- `SUBJECT_TO_EVALUATE`: Sujeto de la base de datos a evaluar

## 🐛 Solución de Problemas

### Errores Comunes

1. **Modelo no encontrado**: Verificar rutas y nombres de archivos
2. **Datos insuficientes**: Ajustar `NUM_EJEMPLOS_A_EVALUAR`
3. **Error de memoria**: Reducir `BATCH_SIZE`

### Requisitos de Archivos

- Archivos `.h5` de modelos GAN en el directorio de trabajo
- Modelo `juez_de_calidad.h5` disponible
- Datos Ninapro DB3 en estructura correcta

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Distribuido bajo la Licencia MIT. Ver `LICENSE` para más información.

## 📞 Contacto

Para preguntas o soporte sobre este script, por favor abra un issue en el repositorio del proyecto.

---

**Nota**: Asegúrese de tener todos los modelos GAN y el juez de calidad en el directorio correcto antes de ejecutar el script.
