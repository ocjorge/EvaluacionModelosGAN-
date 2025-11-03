# =============================================================================
# SCRIPT FINAL: EVALUACIÓN CUANTITATIVA Y CUALITATIVA DE MODELOS GAN
# =============================================================================
import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Añade aquí TODOS los archivos .h5 de tus generadores GAN

# MODELOS_GAN_A_EVALUAR = [
#     # Modelos de epoch 5 en 5 (del 5 al 50)
#     'generador_gan_epoch_5.h5', 'generador_gan_epoch_10.h5',
#     'generador_gan_epoch_15.h5', 'generador_gan_epoch_20.h5',
#     'generador_gan_epoch_25.h5', 'generador_gan_epoch_30.h5',
#     'generador_gan_epoch_35.h5', 'generador_gan_epoch_40.h5',
#     'generador_gan_epoch_45.h5', 'generador_gan_epoch_50.h5',
#
#     # Modelos de epoch consecutivos (del 55 al 111)
#     'generador_gan_epoch_55.h5', 'generador_gan_epoch_56.h5',
#     'generador_gan_epoch_57.h5', 'generador_gan_epoch_58.h5',
#     'generador_gan_epoch_59.h5', 'generador_gan_epoch_60.h5',
#     'generador_gan_epoch_61.h5', 'generador_gan_epoch_62.h5',
#     'generador_gan_epoch_63.h5', 'generador_gan_epoch_64.h5',
#     'generador_gan_epoch_65.h5', 'generador_gan_epoch_66.h5',
#     'generador_gan_epoch_67.h5', 'generador_gan_epoch_68.h5',
#     'generador_gan_epoch_69.h5', 'generador_gan_epoch_70.h5',
#     'generador_gan_epoch_71.h5', 'generador_gan_epoch_72.h5',
#     'generador_gan_epoch_73.h5', 'generador_gan_epoch_74.h5',
#     'generador_gan_epoch_75.h5', 'generador_gan_epoch_76.h5',
#     'generador_gan_epoch_77.h5', 'generador_gan_epoch_78.h5',
#     'generador_gan_epoch_79.h5', 'generador_gan_epoch_80.h5',
#     'generador_gan_epoch_81.h5', 'generador_gan_epoch_82.h5',
#     'generador_gan_epoch_83.h5', 'generador_gan_epoch_84.h5',
#     'generador_gan_epoch_85.h5', 'generador_gan_epoch_86.h5',
#     'generador_gan_epoch_87.h5', 'generador_gan_epoch_88.h5',
#     'generador_gan_epoch_89.h5', 'generador_gan_epoch_90.h5',
#     'generador_gan_epoch_91.h5', 'generador_gan_epoch_92.h5',
#     'generador_gan_epoch_93.h5', 'generador_gan_epoch_94.h5',
#     'generador_gan_epoch_95.h5', 'generador_gan_epoch_96.h5',
#     'generador_gan_epoch_97.h5', 'generador_gan_epoch_98.h5',
#     'generador_gan_epoch_99.h5', 'generador_gan_epoch_100.h5',
#     'generador_gan_epoch_101.h5', 'generador_gan_epoch_102.h5',
#     'generador_gan_epoch_103.h5', 'generador_gan_epoch_104.h5',
#     'generador_gan_epoch_105.h5', 'generador_gan_epoch_106.h5',
#     'generador_gan_epoch_107.h5', 'generador_gan_epoch_108.h5',
#     'generador_gan_epoch_109.h5', 'generador_gan_epoch_110.h5',
#     'generador_gan_epoch_111.h5',
#
#     # Modelos adicionales
#     'generador_gan.h5', 'generador_gan_2.h5', 'generador_gan_3.h5',
#     'generador_gan_4.h5', 'generador_gan_5.h5', 'generador_gan_6.h5'
# ]

MODELOS_GAN_A_EVALUAR = [
    # Modelos de epoch 5 en 5 (del 5 al 50)
    'generador_gan_epoch_5.h5', 'generador_gan_epoch_10.h5',
    'generador_gan_epoch_15.h5', 'generador_gan_epoch_20.h5',
    'generador_gan_epoch_25.h5', 'generador_gan_epoch_30.h5',
    'generador_gan_epoch_35.h5', 'generador_gan_epoch_40.h5',
    'generador_gan_epoch_45.h5', 'generador_gan_epoch_50.h5',
    'generador_gan_epoch_55.h5', 'generador_gan_epoch_60.h5',
    'generador_gan_epoch_65.h5', 'generador_gan_epoch_70.h5',
    'generador_gan_epoch_75.h5', 'generador_gan_epoch_80.h5',
    'generador_gan_epoch_85.h5', 'generador_gan_epoch_90.h5',
    'generador_gan_epoch_95.h5', 'generador_gan_epoch_100.h5',
    'generador_gan_epoch_105.h5', 'generador_gan_epoch_110.h5',
    'generador_gan_epoch_115.h5', 'generador_gan_epoch_120.h5',
    'generador_gan_epoch_125.h5', 'generador_gan_epoch_130.h5',
    'generador_gan_epoch_135.h5', 'generador_gan_epoch_140.h5',
    'generador_gan_epoch_145.h5', 'generador_gan_epoch_150.h5',
    'generador_gan_epoch_155.h5', 'generador_gan_epoch_160.h5',
    'generador_gan_epoch_165.h5', 'generador_gan_epoch_170.h5',
    'generador_gan_epoch_175.h5', 'generador_gan_epoch_180.h5',
    'generador_gan_epoch_185.h5', 'generador_gan_epoch_190.h5',
    'generador_gan_epoch_195.h5', 'generador_gan_epoch_200.h5',
    # Modelos adicionales
    'generador_gan.h5'
]

MODEL_PATH_JUEZ = 'juez_de_calidad.h5'
DATASET_PATH_DB3 = 'ninapro_db3_data'
SUBJECT_TO_EVALUATE = 11

NUM_EJEMPLOS_A_EVALUAR = 1000
WINDOW_SIZE, NUM_CHANNELS = 200, 12
UMBRAL_CALIDAD = 0.8 #0.5 # Usamos 0.5 para una clasificación binaria estándar
BATCH_SIZE = 32

# --- FUNCIONES DE CARGA Y PREPROCESAMIENTO ---

def load_data_for_subject(base_path, subject_id):
    """
    Función robusta que carga datos encontrando archivos por patrón.
    """
    all_emg, all_gestures = np.array([]), np.array([])
    try:
        all_files_in_dir = os.listdir(base_path)
    except FileNotFoundError:
        print(f"¡ERROR CRÍTICO! La carpeta especificada no existe: {base_path}")
        return None, None

    subject_pattern = f'S{subject_id}_'
    subject_files = [f for f in all_files_in_dir if f.startswith(subject_pattern) and f.endswith('.mat')]

    if not subject_files:
        print(f"¡ADVERTENCIA! No se encontró NINGÚN archivo para el patrón '{subject_pattern}' en '{base_path}'.")
        return None, None

    for filename in sorted(subject_files):
        file_path = os.path.join(base_path, filename)
        try:
            data = loadmat(file_path)
            if 'emg' in data and 'restimulus' in data:
                emg, gestures = data['emg'], data['restimulus']
                if all_emg.size == 0:
                    all_emg, all_gestures = emg, gestures
                else:
                    all_emg = np.vstack((all_emg, emg))
                    all_gestures = np.vstack((all_gestures, gestures))
            else:
                continue
        except Exception:
            continue
    if all_emg.size == 0:
        return None, None
    return all_emg, all_gestures

def create_windows(emg, gestures, window_size, step, normalization_type='zscore'):
    """Crea ventanas y las normaliza según el tipo especificado."""
    X = []
    active_indices = np.where(gestures.flatten() != 0)[0]
    for i in range(0, len(active_indices) - window_size, step):
        window_indices = active_indices[i: i + window_size]
        if window_indices[-1] - window_indices[0] != window_size - 1: continue
        window_emg = emg[window_indices]

        if window_emg.shape[1] < NUM_CHANNELS:
            padding = np.zeros((window_emg.shape[0], NUM_CHANNELS - window_emg.shape[1]))
            window_emg = np.concatenate([window_emg, padding], axis=1)

        if normalization_type == 'zscore':
            mean, std = np.mean(window_emg, axis=0), np.std(window_emg, axis=0)
            window_normalized = (window_emg - mean) / (std + 1e-8)
           
        elif normalization_type == 'tanh':
            min_val, max_val = np.min(window_emg), np.max(window_emg)
            window_normalized = 2 * (window_emg - min_val) / (max_val - min_val + 1e-8) - 1
        else:
            raise ValueError("Tipo de normalización no reconocido.")

        X.append(window_normalized)
    return np.array(X)

# --- SCRIPT PRINCIPAL DE EVALUACIÓN ---
if __name__ == "__main__":
    print("Cargando modelo 'Juez de Calidad'...")
    try:
        juez_calidad = load_model(MODEL_PATH_JUEZ, compile=False)
    except IOError:
        print(f"ERROR: No se pudo cargar el modelo '{MODEL_PATH_JUEZ}'.")
        exit()

    print(f"Cargando y preparando datos de prueba del sujeto {SUBJECT_TO_EVALUATE}...")
    emg_data, gesture_labels = load_data_for_subject(DATASET_PATH_DB3, SUBJECT_TO_EVALUATE)

    if emg_data is None:
        print(f"No se pudieron cargar los datos para el sujeto {SUBJECT_TO_EVALUATE}. Saliendo.")
        exit()

    X_test_zscore = create_windows(emg_data, gesture_labels, WINDOW_SIZE, 50, normalization_type='zscore')
    X_test_tanh = create_windows(emg_data, gesture_labels, WINDOW_SIZE, 50, normalization_type='tanh')

    if len(X_test_zscore) < NUM_EJEMPLOS_A_EVALUAR:
        print(f"Advertencia: Solo se encontraron {len(X_test_zscore)} muestras, se usarán todas.")
        NUM_EJEMPLOS_A_EVALUAR = len(X_test_zscore)

    X_test_zscore = X_test_zscore[:NUM_EJEMPLOS_A_EVALUAR]
    X_test_tanh = X_test_tanh[:NUM_EJEMPLOS_A_EVALUAR]

    print(f"Se usarán {NUM_EJEMPLOS_A_EVALUAR} muestras fijas para la evaluación.")

    resultados = []

    for gan_model_file in MODELOS_GAN_A_EVALUAR:
        if not os.path.exists(gan_model_file):
            print(f"Saltando {gan_model_file}, no se encontró.")
            continue

        print(f"\n--- EVALUANDO MODELO: {gan_model_file} ---")
        generador = load_model(gan_model_file, compile=False)

        senales_reconstruidas_tanh = generador.predict(X_test_tanh, batch_size=BATCH_SIZE)

        # senal_0_1 = (senales_reconstruidas_tanh + 1) / 2.0
        # mean = np.mean(senal_0_1, axis=(1, 2), keepdims=True)
        # std = np.std(senal_0_1, axis=(1, 2), keepdims=True)
        # senales_reconstruidas_zscore = (senal_0_1 - mean) / (std + 1e-8)
        senales_reconstruidas_zscore = generador.predict(X_test_zscore, batch_size=BATCH_SIZE)

        puntuaciones_calidad = juez_calidad.predict(senales_reconstruidas_zscore, batch_size=BATCH_SIZE)
        predicciones_juez = (puntuaciones_calidad > UMBRAL_CALIDAD).astype(int)

        etiquetas_ideales = np.ones(NUM_EJEMPLOS_A_EVALUAR, dtype=int)

        accuracy = accuracy_score(etiquetas_ideales, predicciones_juez)
        report = classification_report(etiquetas_ideales, predicciones_juez,
                                       target_names=['Fallo (Pred. 0)', 'Éxito (Pred. 1)'],
                                       labels=[0, 1], zero_division=0)

        resultados.append({
            'Modelo': gan_model_file,
            'Índice de Éxito (%)': accuracy * 100,
            'Reporte': report
        })
        print(f"Resultado: {accuracy * 100:.2f}% de las señales reconstruidas fueron clasificadas como 'Sanas' por el Juez.")
        print("Reporte de Calidad de Reconstrucción:")
        print(report)

    if resultados:
        print("\n\n--- TABLA COMPARATIVA DE RENDIMIENTO DE GENERADORES ---")
        df_resumen = pd.DataFrame([{'Modelo': r['Modelo'], 'Índice de Éxito (%)': r['Índice de Éxito (%)']} for r in resultados])
        df_resumen = df_resumen.sort_values(by='Índice de Éxito (%)', ascending=False)
        print(df_resumen.to_string(index=False))

        mejor_resultado_row = df_resumen.iloc[0]
        mejor_modelo_nombre = mejor_resultado_row['Modelo']

        print(f"\n--- ANÁLISIS DETALLADO DEL MEJOR MODELO: {mejor_modelo_nombre} ---")
        # Encontrar el reporte completo del mejor modelo
        mejor_reporte = next((r['Reporte'] for r in resultados if r['Modelo'] == mejor_modelo_nombre), "No encontrado")
        print(mejor_reporte)

        # Visualización
        indice_medio = len(df_resumen) // 2
        medio_modelo_nombre = df_resumen.iloc[indice_medio]['Modelo']

        print(f"\nVisualizando comparación:")
        print(f"  - Mejor Modelo: {mejor_modelo_nombre} ({mejor_resultado_row['Índice de Éxito (%)']:.2f}%)")
        print(f"  - Modelo Medio: {medio_modelo_nombre} ({df_resumen.iloc[indice_medio]['Índice de Éxito (%)']:.2f}%)")

        mejor_generador = load_model(mejor_modelo_nombre, compile=False)
        medio_generador = load_model(medio_modelo_nombre, compile=False)

        muestra_original_tanh = np.expand_dims(X_test_tanh[0], axis=0)

        reconstruccion_mejor = mejor_generador.predict(muestra_original_tanh, verbose=0)
        reconstruccion_media = medio_generador.predict(muestra_original_tanh, verbose=0)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("Comparación de Calidad de Reconstrucción de la GAN", fontsize=16)

        axes[0].plot(muestra_original_tanh[0, :, 0])
        axes[0].set_title("Señal Original de Amputado (Canal 0)")
        axes[0].set_ylabel("Amplitud Normalizada [-1, 1]")
        axes[0].grid(True, alpha=0.3); axes[0].set_ylim(-1.1, 1.1)

        axes[1].plot(reconstruccion_media[0, :, 0])
        axes[1].set_title(f"Reconstrucción - Modelo Medio\n({medio_modelo_nombre})")
        axes[1].grid(True, alpha=0.3); axes[1].set_ylim(-1.1, 1.1)

        axes[2].plot(reconstruccion_mejor[0, :, 0])
        axes[2].set_title(f"Reconstrucción - Mejor Modelo\n({mejor_modelo_nombre})")
        axes[2].grid(True, alpha=0.3); axes[2].set_ylim(-1.1, 1.1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("\nNo se encontraron modelos para evaluar.")
