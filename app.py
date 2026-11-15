
from pathlib import Path
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import torch
import torch.nn.functional as F


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(
    page_title="Proyecto de Ciencia de Datos",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device("cpu")


# ============================================================
# 2. INFORMACIÓN DEL PROYECTO
# ============================================================

UNIVERSIDAD = "Universidad de La Salle"
FACULTAD = "Facultad de Ingeniería"
PROGRAMA = "Maestría en Inteligencia Artificial"
ASIGNATURA = "Ciencia de Datos"

TITULO_PROYECTO = (
    "ANÁLISIS DE DATOS DE SENSORES AMBIENTALES "
    "MEDIANTE TÉCNICAS DE CIENCIA DE DATOS"
)

ESTUDIANTE = "Juan Andres Lopez Cubides"
DOCENTE = "Fabián Camilo Castro Riveros"
CIUDAD = "Bogotá D.C., Colombia"
PERIODO = "2026"

DATASET_NOMBRE = "Environmental Sensor Telemetry Data"
DATASET_AUTOR = "Gary A. Stafford"

DATASET_URL = (
    "https://www.kaggle.com/datasets/"
    "garystafford/environmental-sensor-data-132k"
)


# ============================================================
# 3. CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():

    df = pd.read_csv(DATA_DIR / "processed_data.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


@st.cache_data
def cargar_datos_clustering():

    df_cluster = pd.read_csv(DATA_DIR / "data_clusters.csv")
    df_cluster["datetime"] = pd.to_datetime(
        df_cluster["datetime"]
    )

    pca = pd.read_csv(DATA_DIR / "pca_clusters.csv")

    perfiles = pd.read_csv(
        DATA_DIR / "cluster_profiles.csv"
    )

    # Se agrega el identificador de cada perfil.
    perfiles.insert(0, "cluster", range(len(perfiles)))

    return df_cluster, pca, perfiles


@st.cache_data
def cargar_resultados():

    archivos = {
        "clasificacion": "classification_metrics.csv",
        "clustering": "clustering_metrics.csv",
        "mlp": "mlp_metrics.csv",
        "mlp_loss": "mlp_loss.csv",
        "mlp_loss_ponderado": "mlp_weighted_loss.csv",
        "rf_regresion": "random_forest_regression.csv",
        "ridge": "ridge_metrics.csv"
    }

    resultados = {}

    for nombre, archivo in archivos.items():

        ruta = RESULTS_DIR / archivo

        if ruta.exists():
            resultados[nombre] = pd.read_csv(ruta)
        else:
            resultados[nombre] = pd.DataFrame()

    return resultados


# ============================================================
# 4. CARGA DE MODELOS
# ============================================================

@st.cache_resource
def cargar_modelos():

    archivos = {
        "regresor": "random_forest_regressor_compressed.pkl",
        "codificador_reg": "encoder_reg.pkl",
        "escalador_reg": "scaler_reg.pkl",
        "codificador_clf": "encoder_clf.pkl",
        "decision_tree": "decision_tree_light.pkl",
        "random_forest_clf": "random_forest_light.pkl",
        "kmeans": "kmeans.pkl",
        "escalador_cluster": "scaler_cluster.pkl"
    }

    modelos = {}

    for nombre, archivo in archivos.items():

        ruta = MODEL_DIR / archivo

        if ruta.exists():
            modelos[nombre] = joblib.load(ruta)
        else:
            modelos[nombre] = None

    return modelos


# ============================================================
# 5. CARGA DE LA CONFIGURACIÓN Y LOS MLP
# ============================================================

@st.cache_data
def cargar_configuracion_mlp():

    ruta = MODEL_DIR / "mlp_config.json"

    if not ruta.exists():
        return None

    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


class MLPFlexible:

    # Se recuperan las capas lineales guardadas en el state_dict.

    def __init__(self, state_dict):

        self.capas = []

        for nombre, tensor in state_dict.items():

            if nombre.endswith("weight") and tensor.ndim == 2:

                nombre_bias = nombre.replace(
                    "weight",
                    "bias"
                )

                if nombre_bias not in state_dict:
                    continue

                peso = tensor.detach().cpu()
                bias = state_dict[nombre_bias].detach().cpu()

                self.capas.append((peso, bias))

        if not self.capas:
            raise ValueError(
                "No se encontraron capas lineales en el MLP."
            )

        self.input_size = self.capas[0][0].shape[1]

    def predict_proba(self, X):

        x = torch.tensor(
            np.asarray(X),
            dtype=torch.float32
        )

        with torch.no_grad():

            for i, (peso, bias) in enumerate(self.capas):

                x = F.linear(x, peso, bias)

                # ReLU en las capas ocultas.
                if i < len(self.capas) - 1:
                    x = F.relu(x)

            salida = torch.sigmoid(x)

        return salida.numpy().reshape(-1)


@st.cache_resource
def cargar_mlps():

    archivos = {
        "mlp_original": "mlp_original.pt",
        "mlp_ponderado": "mlp_weighted.pt"
    }

    mlps = {}

    for nombre, archivo in archivos.items():

        ruta = MODEL_DIR / archivo

        if not ruta.exists():
            mlps[nombre] = None
            continue

        state_dict = torch.load(
            ruta,
            map_location=DEVICE,
            weights_only=True
        )

        mlps[nombre] = MLPFlexible(state_dict)

    return mlps


# ============================================================
# 6. INICIALIZACIÓN DEL PROYECTO
# ============================================================

try:

    df = cargar_datos()

    df_cluster, df_pca, perfiles = cargar_datos_clustering()

    resultados = cargar_resultados()

    modelos = cargar_modelos()

    mlps = cargar_mlps()

    config_mlp = cargar_configuracion_mlp()

except Exception as error:

    st.error(
        f"No fue posible cargar los archivos del proyecto: {error}"
    )

    st.stop()


# ============================================================
# 7. FUNCIONES PARA MOSTRAR RESULTADOS
# ============================================================

def mostrar_tabla(tabla):

    # Las tablas estáticas se muestran sin controles de selección.
    if tabla is not None and not tabla.empty:
        st.table(tabla)


def mostrar_metricas_clasificacion(nombre_modelo):

    tabla = resultados["clasificacion"]

    if tabla.empty:
        st.warning("No hay métricas de clasificación disponibles.")
        return

    datos = tabla[
        tabla["Modelo"].astype(str).str.lower()
        == nombre_modelo.lower()
    ]

    datos = datos[
        datos["Conjunto"].astype(str).str.lower().isin(
            ["prueba", "test"]
        )
    ]

    if datos.empty:

        st.info(
            f"No se encontraron métricas de prueba para {nombre_modelo}."
        )

        return

    fila = datos.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{fila['Accuracy']:.4f}")
    c2.metric("Precision", f"{fila['Precision']:.4f}")
    c3.metric("Recall", f"{fila['Recall']:.4f}")
    c4.metric("F1-score", f"{fila['F1-Score']:.4f}")
    c5.metric("AUC-ROC", f"{fila['AUC-ROC']:.4f}")


def mostrar_metricas_regresion(tabla):

    if tabla.empty:
        st.warning("No hay métricas de regresión disponibles.")
        return

    datos = tabla.copy()

    if "Conjunto" in datos.columns:

        datos = datos[
            datos["Conjunto"].astype(str).str.lower().isin(
                ["prueba", "test"]
            )
        ]

    if datos.empty:
        mostrar_tabla(tabla)
        return

    fila = datos.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("R²", f"{fila['R2']:.4f}")
    c2.metric("RMSE", f"{fila['RMSE']:.4f}")
    c3.metric("MAE", f"{fila['MAE']:.4f}")


def preparar_entrada_clasificacion(
    co,
    humidity,
    lpg,
    smoke,
    temp,
    hora,
    dispositivo
):

    # Se preparan las mismas variables usadas por los clasificadores.

    entrada = pd.DataFrame([{
        "co": co,
        "humidity": humidity,
        "lpg": lpg,
        "smoke": smoke,
        "temp": temp,
        "hour_sin": np.sin(2 * np.pi * hora / 24),
        "hour_cos": np.cos(2 * np.pi * hora / 24)
    }])

    codificador = modelos["codificador_clf"]

    if codificador is None:
        raise ValueError("No se encontró encoder_clf.pkl.")

    device_encoded = codificador.transform(
        pd.DataFrame({"device": [dispositivo]})
    )

    if hasattr(device_encoded, "toarray"):
        device_encoded = device_encoded.toarray()

    device_df = pd.DataFrame(
        device_encoded,
        columns=codificador.get_feature_names_out(["device"])
    )

    entrada = pd.concat(
        [entrada, device_df],
        axis=1
    )

    return entrada


def predecir_clasificador(modelo, entrada):

    # Se respeta el orden de variables guardado en el modelo.

    columnas = getattr(
        modelo,
        "feature_names_in_",
        None
    )

    if columnas is not None:

        entrada = entrada.reindex(
            columns=columnas,
            fill_value=0
        )

    prediccion = int(
        modelo.predict(entrada)[0]
    )

    probabilidades = None

    if hasattr(modelo, "predict_proba"):

        probabilidades = modelo.predict_proba(
            entrada
        )[0]

    return prediccion, probabilidades


def preparar_entrada_mlp(
    co,
    humidity,
    lpg,
    smoke,
    temp,
    hora,
    dispositivo,
    input_size
):

    # Se preparan las 11 entradas de la red neuronal.
    # El orden debe coincidir con el usado durante el entrenamiento.

    codificador = modelos["codificador_clf"]

    device_encoded = codificador.transform(
        pd.DataFrame({"device": [dispositivo]})
    )

    if hasattr(device_encoded, "toarray"):
        device_encoded = device_encoded.toarray()

    entrada = np.array([[
        co,
        humidity,
        lpg,
        smoke,
        temp,
        hora,
        np.sin(2 * np.pi * hora / 24),
        np.cos(2 * np.pi * hora / 24),
        *device_encoded.reshape(-1)
    ]], dtype=np.float32)

    if entrada.shape[1] != input_size:

        raise ValueError(
            f"El MLP espera {input_size} características, "
            f"pero se construyeron {entrada.shape[1]}. "
            "Revisar el orden de variables del entrenamiento."
        )

    return entrada


def mostrar_resultado_luz(prediccion, probabilidades=None):

    st.divider()

    if prediccion == 1:
        st.success("Resultado: LUZ DETECTADA")
    else:
        st.warning("Resultado: LUZ NO DETECTADA")

    if probabilidades is not None:

        c1, c2 = st.columns(2)

        c1.metric(
            "Probabilidad de no detectar luz",
            f"{probabilidades[0] * 100:.2f}%"
        )

        c2.metric(
            "Probabilidad de detectar luz",
            f"{probabilidades[1] * 100:.2f}%"
        )


def formulario_sensores(prefijo):

    # Se ingresan las mediciones para la nueva observación.

    st.subheader("Datos de entrada")

    c1, c2 = st.columns(2)

    with c1:

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=float(df["co"].median()),
            format="%.6f",
            key=f"{prefijo}_co"
        )

        humidity = st.number_input(
            "Humedad (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(df["humidity"].median()),
            key=f"{prefijo}_humidity"
        )

        lpg = st.number_input(
            "LPG",
            min_value=0.0,
            value=float(df["lpg"].median()),
            format="%.6f",
            key=f"{prefijo}_lpg"
        )

    with c2:

        smoke = st.number_input(
            "Humo",
            min_value=0.0,
            value=float(df["smoke"].median()),
            format="%.6f",
            key=f"{prefijo}_smoke"
        )

        temp = st.number_input(
            "Temperatura (°C)",
            value=float(df["temp"].median()),
            key=f"{prefijo}_temp"
        )

        hora = st.slider(
            "Hora del día",
            min_value=0,
            max_value=23,
            value=12,
            key=f"{prefijo}_hora"
        )

    dispositivo = st.selectbox(
        "Dispositivo",
        modelos["codificador_clf"].categories_[0].tolist(),
        key=f"{prefijo}_device"
    )

    return (
        co,
        humidity,
        lpg,
        smoke,
        temp,
        hora,
        dispositivo
    )


# ============================================================
# 8. PORTADA ACADÉMICA
# ============================================================

def mostrar_portada():

    st.markdown(
        """
        <style>
        .academic-cover {
            text-align: center;
            padding: 35px 20px 20px 20px;
        }

        .university {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: 1px;
        }

        .faculty {
            font-size: 16px;
            margin-top: 8px;
        }

        .program {
            font-size: 15px;
            color: #8b949e;
            margin-top: 5px;
        }

        .project-title {
            font-size: 32px;
            font-weight: 800;
            line-height: 1.35;
            text-align: center;
            max-width: 900px;
            margin: 45px auto 20px auto;
        }

        .project-subtitle {
            font-size: 17px;
            color: #8b949e;
            text-align: center;
            margin-bottom: 35px;
        }

        .academic-info {
            max-width: 750px;
            margin: auto;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.3);
        }

        .info-label {
            font-size: 12px;
            color: #8b949e;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }

        .info-value {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 20px;
        }

        .location {
            text-align: center;
            color: #8b949e;
            margin-top: 30px;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="academic-cover">
            <div class="university">{UNIVERSIDAD.upper()}</div>
            <div class="faculty">{FACULTAD}</div>
            <div class="program">{PROGRAMA}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        f"""
        <div class="project-title">
            {TITULO_PROYECTO}
        </div>

        <div class="project-subtitle">
            Dashboard interactivo · Proyecto Integrador
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class="info-grid">
            <div>
                <div class="info-label">PRESENTADO POR</div>
                <div class="info-value">Juan Andres Lopez Cubides</div>
            </div>
            <div>
                <div class="info-label">ASIGNATURA</div>
                <div class="info-value">Ciencia de Datos</div>
            </div>
            <div>
                <div class="info-label">DOCENTE</div>
                <div class="info-value">Fabián Camilo Castro Riveros</div>
            </div>
            <div>
                <div class="info-label">PERIODO ACADÉMICO</div>
                <div class="info-value">2026</div>
            </div>
        </div>
        <div class="cover-location">
            Bogotá D.C., Colombia
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Fuente de información")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(f"**Conjunto de datos:** {DATASET_NOMBRE}")
        st.write(f"**Autor:** {DATASET_AUTOR}")
        st.write("**Archivo:** iot_telemetry_data.csv")

    with c2:

        st.markdown("**Repositorio:**")
        st.markdown(
            f"[Consultar conjunto de datos en Kaggle]({DATASET_URL})"
        )

    st.divider()

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:

        if st.button(
            "Ingresar al dashboard",
            type="primary",
            use_container_width=True
        ):

            st.session_state["dashboard_iniciado"] = True
            st.rerun()


# ============================================================
# 9. NAVEGACIÓN
# ============================================================

if "dashboard_iniciado" not in st.session_state:
    st.session_state["dashboard_iniciado"] = False


if not st.session_state["dashboard_iniciado"]:

    st.title("Proyecto de Ciencia de Datos")

    mostrar_portada()

    st.stop()


st.title("🌡️ Dashboard de Sensores Ambientales")

st.sidebar.title("Navegación")

if st.sidebar.button(
    "← Volver a portada",
    use_container_width=True
):

    st.session_state["dashboard_iniciado"] = False
    st.rerun()


seccion = st.sidebar.radio(
    "Seleccione una sección:",
    [
        "Resumen general",
        "Exploración de datos",
        "Predicción de temperatura",
        "Clasificación de luz",
        "Análisis de agrupamiento",
        "Evaluación de modelos"
    ]
)

st.sidebar.divider()

st.sidebar.caption(UNIVERSIDAD)
st.sidebar.caption(PROGRAMA)
st.sidebar.caption(f"Asignatura: {ASIGNATURA}")
st.sidebar.caption(f"Periodo: {PERIODO}")


# ============================================================
# 10. RESUMEN GENERAL
# ============================================================

if seccion == "Resumen general":

    st.header("Resumen general")

    # Se presentan los datos principales del conjunto.

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Registros",
        f"{len(df):,}"
    )

    c2.metric(
        "Dispositivos",
        df["device"].nunique()
    )

    c3.metric(
        "Variables originales",
        "9"
    )

    c4.metric(
        "Grupos identificados",
        perfiles.shape[0]
    )

    st.divider()

    st.subheader("Descripción del conjunto de datos")

    st.write(
        """
        El conjunto contiene registros de telemetría ambiental
        obtenidos mediante dispositivos IoT.

        Se incluyen mediciones de temperatura, humedad,
        concentraciones de gases, detección de luz ambiental
        y movimiento.
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        conteo_luz = (
            df["light"]
            .astype(int)
            .value_counts()
            .rename_axis("Luz")
            .reset_index(name="Registros")
        )

        conteo_luz["Luz"] = conteo_luz["Luz"].map({
            0: "No detectada",
            1: "Detectada"
        })

        fig = px.pie(
            conteo_luz,
            names="Luz",
            values="Registros",
            title="Detección de luz ambiental",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        conteo_dispositivos = (
            df["device"]
            .value_counts()
            .rename_axis("Dispositivo")
            .reset_index(name="Registros")
        )

        fig = px.bar(
            conteo_dispositivos,
            x="Dispositivo",
            y="Registros",
            title="Registros por dispositivo"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Periodo de observación")

    st.write(
        f"Desde **{df['datetime'].min()}** "
        f"hasta **{df['datetime'].max()}**"
    )


# ============================================================
# 11. EXPLORACIÓN DE DATOS
# ============================================================

elif seccion == "Exploración de datos":

    st.header("Exploración de datos")

    # Se filtran los registros por dispositivo y periodo del día.

    dispositivos = sorted(
        df["device"].dropna().unique().tolist()
    )

    dispositivos_seleccionados = st.sidebar.multiselect(
        "Dispositivos",
        dispositivos,
        default=dispositivos
    )

    periodos = sorted(
        df["day_period"].dropna().unique().tolist()
    )

    periodos_seleccionados = st.sidebar.multiselect(
        "Periodos del día",
        periodos,
        default=periodos
    )

    datos = df[
        df["device"].isin(dispositivos_seleccionados)
        & df["day_period"].isin(periodos_seleccionados)
    ].copy()

    c1, c2 = st.columns(2)

    c1.metric(
        "Registros filtrados",
        f"{len(datos):,}"
    )

    c2.metric(
        "Dispositivos seleccionados",
        len(dispositivos_seleccionados)
    )

    st.divider()

    st.subheader("Estadísticas descriptivas")

    variables = [
        "co",
        "humidity",
        "lpg",
        "smoke",
        "temp"
    ]

    st.table(
        datos[variables].describe().round(4)
    )

    st.subheader("Comportamiento temporal")

    variable = st.selectbox(
        "Seleccione una variable",
        variables,
        format_func=lambda x: {
            "co": "Monóxido de carbono (CO)",
            "humidity": "Humedad",
            "lpg": "Gas LPG",
            "smoke": "Humo",
            "temp": "Temperatura"
        }[x]
    )

    if not datos.empty:

        temporal = (
            datos.set_index("datetime")[variable]
            .resample("1h")
            .mean()
            .reset_index()
        )

        fig = px.line(
            temporal,
            x="datetime",
            y=variable,
            title="Promedio horario",
            labels={
                "datetime": "Fecha y hora",
                variable: "Valor promedio"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No hay registros para los filtros seleccionados."
        )

    st.subheader("Muestra de registros")

    st.table(
        datos.head(50)
    )


# ============================================================
# 12. PREDICCIÓN DE TEMPERATURA
# ============================================================

elif seccion == "Predicción de temperatura":

    st.header("Predicción de temperatura")

    st.write(
        """
        El modelo Random Forest Regressor estima la temperatura
        a partir de las mediciones de los sensores, la hora
        y el dispositivo.
        """
    )

    if modelos["regresor"] is None:

        st.error(
            "No se encontró random_forest_regressor.pkl."
        )

        st.stop()

    c1, c2 = st.columns(2)

    with c1:

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=float(df["co"].median()),
            format="%.6f"
        )

        humidity = st.number_input(
            "Humedad (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(df["humidity"].median())
        )

        lpg = st.number_input(
            "LPG",
            min_value=0.0,
            value=float(df["lpg"].median()),
            format="%.6f"
        )

    with c2:

        smoke = st.number_input(
            "Humo",
            min_value=0.0,
            value=float(df["smoke"].median()),
            format="%.6f"
        )

        hora = st.slider(
            "Hora del día",
            min_value=0,
            max_value=23,
            value=12
        )

        dispositivo = st.selectbox(
            "Dispositivo",
            modelos["codificador_reg"].categories_[0].tolist()
        )

    if st.button(
        "Predecir temperatura",
        type="primary"
    ):

        try:

            # Se escalan las cuatro variables numéricas.

            entrada_numerica = pd.DataFrame([{
                "co": co,
                "humidity": humidity,
                "lpg": lpg,
                "smoke": smoke
            }])

            escalados = modelos[
                "escalador_reg"
            ].transform(entrada_numerica)

            escalados_df = pd.DataFrame(
                escalados,
                columns=modelos[
                    "escalador_reg"
                ].feature_names_in_
            )

            # Se codifica el dispositivo.

            device_encoded = modelos[
                "codificador_reg"
            ].transform(
                pd.DataFrame({
                    "device": [dispositivo]
                })
            )

            if hasattr(device_encoded, "toarray"):
                device_encoded = device_encoded.toarray()

            device_df = pd.DataFrame(
                device_encoded,
                columns=modelos[
                    "codificador_reg"
                ].get_feature_names_out(["device"])
            )

            # Se calculan las variables cíclicas de la hora.

            temporal = pd.DataFrame([{
                "hour_sin": np.sin(2 * np.pi * hora / 24),
                "hour_cos": np.cos(2 * np.pi * hora / 24)
            }])

            entrada = pd.concat(
                [
                    escalados_df,
                    temporal,
                    device_df
                ],
                axis=1
            )

            # Se conserva el orden de variables del modelo.

            entrada = entrada.reindex(
                columns=modelos[
                    "regresor"
                ].feature_names_in_,
                fill_value=0
            )

            prediccion = modelos[
                "regresor"
            ].predict(entrada)[0]

            st.success("Predicción realizada correctamente.")

            st.metric(
                "Temperatura estimada",
                f"{prediccion:.2f} °C"
            )

            st.divider()

            st.subheader("Desempeño del modelo")

            mostrar_metricas_regresion(
                resultados["rf_regresion"]
            )

        except Exception as error:

            st.error(
                f"No fue posible realizar la predicción: {error}"
            )


# ============================================================
# 13. CLASIFICACIÓN DE LUZ
# ============================================================

elif seccion == "Clasificación de luz":

    st.header("Clasificación de luz ambiental")

    st.write(
        """
        Se ingresan las mediciones de los sensores para
        identificar si existe detección de luz ambiental.
        """
    )

    # Se muestran solamente los modelos que están guardados.

    modelos_disponibles = []

    if modelos["decision_tree"] is not None:
        modelos_disponibles.append("Decision Tree")

    if modelos["random_forest_clf"] is not None:
        modelos_disponibles.append("Random Forest")

    if mlps["mlp_original"] is not None:
        modelos_disponibles.append("MLP original")

    if mlps["mlp_ponderado"] is not None:
        modelos_disponibles.append("MLP ponderado")

    if not modelos_disponibles:

        st.error(
            "No se encontraron modelos de clasificación."
        )

        st.stop()

    modelo_elegido = st.selectbox(
        "Seleccione el modelo",
        modelos_disponibles
    )

    valores = formulario_sensores("clasificacion")

    if st.button(
        "Clasificar detección de luz",
        type="primary"
    ):

        try:

            entrada = preparar_entrada_clasificacion(
                *valores
            )

            if modelo_elegido == "Decision Tree":

                prediccion, probabilidades = (
                    predecir_clasificador(
                        modelos["decision_tree"],
                        entrada
                    )
                )

                nombre_metricas = "Decision Tree"

            elif modelo_elegido == "Random Forest":

                prediccion, probabilidades = (
                    predecir_clasificador(
                        modelos["random_forest_clf"],
                        entrada
                    )
                )

                nombre_metricas = "Random Forest"

            else:

                nombre_mlp = (
                    "mlp_original"
                    if modelo_elegido == "MLP original"
                    else "mlp_ponderado"
                )

                modelo_mlp = mlps[nombre_mlp]

                # Se construye la entrada con el tamaño del MLP.

                entrada_mlp = preparar_entrada_mlp(
                    *valores,
                    input_size=modelo_mlp.input_size
                )

                probabilidad = float(
                    modelo_mlp.predict_proba(
                        entrada_mlp
                    )[0]
                )

                prediccion = int(
                    probabilidad >= 0.5
                )

                probabilidades = np.array([
                    1 - probabilidad,
                    probabilidad
                ])

                nombre_metricas = (
                    "MLP"
                    if nombre_mlp == "mlp_original"
                    else "MLP ponderado"
                )

            mostrar_resultado_luz(
                prediccion,
                probabilidades
            )

            st.subheader("Desempeño del modelo")

            if nombre_metricas in [
                "Decision Tree",
                "Random Forest"
            ]:

                mostrar_metricas_clasificacion(
                    nombre_metricas
                )

            else:

                # Se muestran las métricas guardadas del MLP.

                st.info(
                    "Estas métricas corresponden a la evaluación "
                    "realizada durante el entrenamiento."
                )

                if not resultados["mlp"].empty:
                    mostrar_tabla(resultados["mlp"])
                else:
                    st.warning(
                        "No se encontró mlp_metrics.csv."
                    )

        except Exception as error:

            st.error(
                f"No fue posible realizar la clasificación: {error}"
            )

    st.divider()

    st.subheader("Comparación de modelos")

    tabla = resultados["clasificacion"]

    if not tabla.empty:

        mostrar_tabla(tabla.round(4))

        metrica = st.selectbox(
            "Seleccione una métrica",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "AUC-ROC"
            ],
            key="metrica_clasificacion"
        )

        fig = px.bar(
            tabla,
            x="Modelo",
            y=metrica,
            color="Conjunto",
            barmode="group",
            title=f"Comparación según {metrica}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 14. ANÁLISIS DE AGRUPAMIENTO
# ============================================================

elif seccion == "Análisis de agrupamiento":

    st.header("Análisis de agrupamiento")

    st.write(
        """
        K-Means agrupa las observaciones según la similitud
        de sus características ambientales.
        """
    )

    if modelos["kmeans"] is None:

        st.error("No se encontró kmeans.pkl.")
        st.stop()

    # Los grupos se describen según sus perfiles promedio.

    interpretaciones = {

        0: {
            "nombre": "Mayor temperatura y detección de luz",
            "descripcion": (
                "Presenta la temperatura promedio más elevada, "
                "detección de luz en casi todos los registros "
                "y humedad intermedia."
            )
        },

        1: {
            "nombre": "Mayores concentraciones de gases",
            "descripcion": (
                "Presenta los mayores valores promedio de CO, "
                "LPG y humo, junto con una humedad más baja."
            )
        },

        2: {
            "nombre": "Mayor humedad y menor temperatura",
            "descripcion": (
                "Presenta la humedad promedio más elevada, "
                "la temperatura más baja y menores "
                "concentraciones de gases."
            )
        }
    }

    st.subheader("Asignar una nueva observación")

    c1, c2 = st.columns(2)

    with c1:

        co_nueva = st.number_input(
            "CO",
            min_value=0.0,
            value=float(df["co"].median()),
            format="%.6f",
            key="cluster_co"
        )

        humidity_nueva = st.number_input(
            "Humedad (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(df["humidity"].median()),
            key="cluster_humidity"
        )

        lpg_nueva = st.number_input(
            "LPG",
            min_value=0.0,
            value=float(df["lpg"].median()),
            format="%.6f",
            key="cluster_lpg"
        )

        smoke_nueva = st.number_input(
            "Humo",
            min_value=0.0,
            value=float(df["smoke"].median()),
            format="%.6f",
            key="cluster_smoke"
        )

    with c2:

        temp_nueva = st.number_input(
            "Temperatura (°C)",
            value=float(df["temp"].median()),
            key="cluster_temp"
        )

        luz_nueva = st.selectbox(
            "¿Se detecta luz ambiental?",
            [False, True],
            format_func=lambda x: "Sí" if x else "No",
            key="cluster_luz"
        )

        hora_nueva = st.slider(
            "Hora del registro",
            min_value=0,
            max_value=23,
            value=12,
            key="cluster_hora"
        )

    if st.button(
        "Identificar grupo",
        type="primary"
    ):

        try:

            # Se preparan las ocho variables del agrupamiento.

            entrada = pd.DataFrame([{
                "co": co_nueva,
                "humidity": humidity_nueva,
                "lpg": lpg_nueva,
                "smoke": smoke_nueva,
                "temp": temp_nueva,
                "light": int(luz_nueva),
                "hour_sin": np.sin(
                    2 * np.pi * hora_nueva / 24
                ),
                "hour_cos": np.cos(
                    2 * np.pi * hora_nueva / 24
                )
            }])

            entrada_escalada = modelos[
                "escalador_cluster"
            ].transform(entrada)

            grupo = int(
                modelos["kmeans"].predict(
                    entrada_escalada
                )[0]
            )

            info = interpretaciones[grupo]

            st.divider()

            st.success(
                f"La observación pertenece al GRUPO {grupo}"
            )

            st.subheader(info["nombre"])
            st.write(info["descripcion"])

            perfil = perfiles.loc[
                perfiles["cluster"] == grupo
            ].iloc[0]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Temperatura promedio",
                f"{perfil['temp']:.2f} °C"
            )

            c2.metric(
                "Humedad promedio",
                f"{perfil['humidity']:.2f} %"
            )

            c3.metric(
                "CO promedio",
                f"{perfil['co']:.5f}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "LPG promedio",
                f"{perfil['lpg']:.5f}"
            )

            c2.metric(
                "Humo promedio",
                f"{perfil['smoke']:.5f}"
            )

            c3.metric(
                "Luz promedio",
                f"{perfil['light']:.4f}"
            )

        except Exception as error:

            st.error(
                f"No fue posible asignar el grupo: {error}"
            )

    st.divider()

    st.subheader("Distribución de los grupos")

    conteo = (
        df_cluster["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    conteo.columns = [
        "Grupo",
        "Registros"
    ]

    conteo["Porcentaje"] = (
        conteo["Registros"] / len(df_cluster) * 100
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.pie(
            conteo,
            names="Grupo",
            values="Registros",
            title="Distribución de observaciones",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.bar(
            conteo,
            x="Grupo",
            y="Porcentaje",
            text="Porcentaje",
            title="Porcentaje por grupo"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Visualización mediante PCA")

    muestra_pca = df_pca.sample(
        n=min(10000, len(df_pca)),
        random_state=42
    )

    fig = px.scatter(
        muestra_pca,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Distribución de grupos en el espacio PCA",
        opacity=0.6,
        labels={
            "PC1": "Componente principal 1",
            "PC2": "Componente principal 2",
            "cluster": "Grupo"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Perfiles de los grupos")

    perfiles_es = perfiles.rename(
        columns={
            "cluster": "Grupo",
            "co": "CO",
            "humidity": "Humedad",
            "lpg": "LPG",
            "smoke": "Humo",
            "temp": "Temperatura",
            "light": "Luz",
            "hour_sin": "Hora (seno)",
            "hour_cos": "Hora (coseno)"
        }
    )

    st.table(
        perfiles_es.round(4)
    )

    for grupo, info in interpretaciones.items():

        with st.expander(
            f"Grupo {grupo}: {info['nombre']}"
        ):

            st.write(info["descripcion"])

    st.divider()

    st.subheader("Evaluación del agrupamiento")

    tabla_k = resultados["clustering"]

    if not tabla_k.empty:

        st.table(tabla_k)

        mejor = tabla_k.loc[
            tabla_k["Silhouette"].idxmax()
        ]

        st.metric(
            "Mejor Silhouette registrado",
            f"{mejor['Silhouette']:.4f}",
            f"K = {int(mejor['K'])}"
        )


# ============================================================
# 15. EVALUACIÓN DE MODELOS
# ============================================================

elif seccion == "Evaluación de modelos":

    st.header("Evaluación de modelos")

    # Se presentan las métricas obtenidas durante el entrenamiento.

    # --------------------------------------------------------
    # REGRESIÓN
    # --------------------------------------------------------

    st.subheader("Modelos de regresión")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("**Regresión Ridge**")

        if not resultados["ridge"].empty:
            mostrar_tabla(resultados["ridge"])
        else:
            st.info("No hay resultados de Ridge.")

    with c2:

        st.markdown("**Random Forest Regressor**")

        if not resultados["rf_regresion"].empty:
            mostrar_tabla(resultados["rf_regresion"])
        else:
            st.info("No hay resultados de Random Forest.")

    st.divider()

    # --------------------------------------------------------
    # CLASIFICACIÓN
    # --------------------------------------------------------

    st.subheader("Modelos de clasificación")

    tabla_clasificacion = resultados["clasificacion"]

    if not tabla_clasificacion.empty:

        mostrar_tabla(
            tabla_clasificacion.round(4)
        )

        metrica = st.selectbox(
            "Seleccione una métrica",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "AUC-ROC"
            ],
            key="evaluacion_clasificacion"
        )

        fig = px.bar(
            tabla_clasificacion,
            x="Modelo",
            y=metrica,
            color="Conjunto",
            barmode="group",
            title=f"Comparación según {metrica}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No se encontraron resultados de clasificación."
        )

    st.divider()

    # --------------------------------------------------------
    # CLUSTERING
    # --------------------------------------------------------

    st.subheader("Evaluación del agrupamiento")

    tabla_k = resultados["clustering"]

    if not tabla_k.empty:

        fig = px.line(
            tabla_k,
            x="K",
            y="Silhouette",
            markers=True,
            title="Coeficiente Silhouette según K",
            labels={
                "K": "Número de grupos",
                "Silhouette": "Coeficiente Silhouette"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.table(tabla_k)

    else:

        st.warning(
            "No se encontraron resultados de clustering."
        )

    st.divider()

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    st.subheader("Evaluación de la red neuronal MLP")

    if not resultados["mlp"].empty:

        st.markdown("**Métricas registradas del MLP**")

        st.table(
            resultados["mlp"]
        )

    else:

        st.warning(
            "No se encontró mlp_metrics.csv."
        )

    # Curva del MLP original.

    perdida = resultados["mlp_loss"]

    if not perdida.empty:

        st.markdown("**MLP original: pérdida de entrenamiento y validación**")

        perdida = perdida.copy()

        perdida["Época"] = range(
            1,
            len(perdida) + 1
        )

        fig = px.line(
            perdida,
            x="Época",
            y=["train_loss", "validation_loss"],
            markers=True,
            title="Pérdida del MLP original",
            labels={
                "value": "Pérdida",
                "variable": "Conjunto"
            }
        )

        fig.for_each_trace(
            lambda trace: trace.update(
                name={
                    "train_loss": "Entrenamiento",
                    "validation_loss": "Validación"
                }.get(trace.name, trace.name)
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Curva del MLP ponderado.

    perdida_ponderada = resultados["mlp_loss_ponderado"]

    if not perdida_ponderada.empty:

        st.markdown("**MLP ponderado: pérdida de entrenamiento y validación**")

        perdida_ponderada = perdida_ponderada.copy()

        perdida_ponderada["Época"] = range(
            1,
            len(perdida_ponderada) + 1
        )

        fig = px.line(
            perdida_ponderada,
            x="Época",
            y=["train_loss", "validation_loss"],
            markers=True,
            title="Pérdida del MLP ponderado",
            labels={
                "value": "Pérdida",
                "variable": "Conjunto"
            }
        )

        fig.for_each_trace(
            lambda trace: trace.update(
                name={
                    "train_loss": "Entrenamiento",
                    "validation_loss": "Validación"
                }.get(trace.name, trace.name)
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 16. PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    f"{UNIVERSIDAD} | {PROGRAMA} | {ASIGNATURA} | {PERIODO}"
)
