# Importar librerías necesarias
import os                # Para manejar variables de entorno y rutas de archivos
import openai            # Para interactuar con la API de OpenAI
import faiss             # Para crear el índice de búsqueda vectorial
import numpy as np       # Para operaciones numéricas y manejo de matrices
import logging           # Para registrar eventos y depurar
import streamlit as st   # Para crear la interfaz web interactiva

# Decorar la función para optimizar su ejecución
@st.cache_resource
def cargar_texto(nombre_archivo):
    """
    Lee el contenido de un archivo de texto.

    Parámetros:
    - nombre_archivo (str): Nombre del archivo a leer.

    Retorna:
    - texto (str): Contenido del archivo de texto.
    """
    # Verificar si el archivo existe
    if not os.path.exists(nombre_archivo):
        raise FileNotFoundError(f"El archivo {nombre_archivo} no se encontró.")

    # Leer el archivo con codificación UTF-8
    with open(nombre_archivo, "r", encoding="utf-8") as file:
        texto = file.read()
    
    return texto

# Cargar el archivo "Leo.txt"
archivo_texto = "Leo.txt"
texto = cargar_texto(archivo_texto)

# Decorar la función para mejorar la eficiencia
@st.cache_resource
def dividir_texto_en_trozos(texto, palabras_por_trozo=500):
    """
    Divide un texto en trozos de un número específico de palabras.

    Parámetros:
    - texto (str): Texto completo a dividir.
    - palabras_por_trozo (int): Cantidad de palabras por trozo.

    Retorna:
    - trozos (list): Lista de trozos de texto.
    """
    # Dividir el texto en palabras individuales
    palabras = texto.split()
    
    # Crear los trozos de texto con la cantidad especificada
    trozos = [' '.join(palabras[i:i + palabras_por_trozo]) for i in range(0, len(palabras), palabras_por_trozo)]
    return trozos

# Crear los trozos de texto
trozos = dividir_texto_en_trozos(texto)

# Configurar el registro de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cargar las variables de entorno desde el archivo .env
from dotenv import load_dotenv
load_dotenv()

# Configurar la clave API de OpenAI
openai.api_key = os.getenv('API_KEY')

# Verificar que la clave API está configurada
if not openai.api_key:
    raise ValueError("La clave API de OpenAI no está configurada. Establece la variable de entorno 'OPENAI_API_KEY'.")

# Decorar la función para optimizar la ejecución
@st.cache_resource
def generar_embeddings_y_configurar_faiss(trozos):
    """
    Genera embeddings para cada trozo y configura un índice FAISS.

    Parámetros:
    - trozos (list): Lista de trozos de texto.

    Retorna:
    - index (faiss.IndexFlatL2): Índice FAISS configurado.
    - embeddings (list): Lista de embeddings generados para los trozos.
    """
    embeddings = []
    for idx, trozo in enumerate(trozos):
        try:
            # Generar el embedding del trozo
            embedding = openai.Embedding.create(
                input=trozo,
                model="text-embedding-ada-002"
            )["data"][0]["embedding"]
            embeddings.append(embedding)
            logging.info(f"Embedding creado para el trozo {idx+1}/{len(trozos)}")
        except Exception as e:
            logging.error(f"Error al crear embedding para el trozo {idx+1}: {e}")
    
    # Configurar FAISS
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings

# Configurar FAISS y generar embeddings
index, embeddings = generar_embeddings_y_configurar_faiss(trozos)

def obtener_contexto(pregunta, index, trozos, k=5):
    """
    Genera el contexto relevante para una pregunta utilizando embeddings y FAISS.

    Parámetros:
    - pregunta (str): La pregunta del usuario.
    - index (faiss.IndexFlatL2): Índice FAISS configurado.
    - trozos (list): Lista de trozos de texto.
    - k (int): Número de trozos relevantes a recuperar.

    Retorna:
    - contexto (str): Trozos más relevantes concatenados en un solo texto.
    """
    try:
        # Generar el embedding de la pregunta
        pregunta_embedding = openai.Embedding.create(
            input=pregunta,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]

        # Convertir el embedding a un array de NumPy
        pregunta_embedding_array = np.array([pregunta_embedding]).astype("float32")

        # Buscar en el índice FAISS
        D, I = index.search(pregunta_embedding_array, k)

        # Recuperar y concatenar los trozos relevantes
        return ' '.join(trozos[i] for i in I[0])

    except Exception as e:
        logging.error(f"Error al obtener el contexto: {e}")
        return "Error al obtener el contexto."

def generar_respuesta(pregunta, contexto):
    """
    Genera una respuesta a una pregunta basada en el contexto proporcionado.

    Parámetros:
    - pregunta (str): La pregunta del usuario.
    - contexto (str): El contexto relevante obtenido.

    Retorna:
    - respuesta (str): La respuesta generada por el modelo.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un experto en la película 'Leo'. "
                "Proporciona respuestas detalladas basadas únicamente en el contexto proporcionado. "
                "Si no encuentras la respuesta en el contexto, indica que no tienes esa información."
            )
        },
        {
            "role": "user",
            "content": f"Contexto: {contexto}\n\nPregunta: {pregunta}"
        }
    ]
    try:
        # Generar la respuesta utilizando la API de ChatCompletion
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.5,
            n=1,
            stop=None
        )
        # Extraer el contenido de la respuesta
        respuesta_texto = respuesta['choices'][0]['message']['content'].strip()

        # Registrar la generación exitosa
        logging.info(f"Respuesta generada para la pregunta: '{pregunta}'")

        return respuesta_texto
    except Exception as e:
        # Registrar el error
        logging.error(f"Error al generar la respuesta: {e}")
        return "Lo siento, ocurrió un error al generar la respuesta."

cache_respuestas = {}  # Caché para respuestas previas

def responder_pregunta(pregunta, index, trozos):
    """
    Responde a la pregunta del usuario utilizando RAG.

    Parámetros:
    - pregunta (str): La pregunta del usuario.
    - index (faiss.IndexFlatL2): Índice FAISS configurado.
    - trozos (list): Lista de trozos de texto.

    Retorna:
    - respuesta (str): La respuesta generada.
    """
    # Verificar si la pregunta ya está en la caché
    if pregunta in cache_respuestas:
        logging.info(f"Respuesta obtenida de la caché para la pregunta: '{pregunta}'")
        return cache_respuestas[pregunta]

    # Obtener el contexto relevante
    contexto = obtener_contexto(pregunta, index, trozos)

    # Verificar si se obtuvo contexto
    if not contexto:
        logging.warning(f"No se encontró contexto para la pregunta: '{pregunta}'")
        return "No pude encontrar información relevante para responder tu pregunta."

    # Generar la respuesta basada en el contexto
    respuesta = generar_respuesta(pregunta, contexto)

    # Almacenar la respuesta en la caché
    cache_respuestas[pregunta] = respuesta

    return respuesta

if __name__ == "__main__":
    # Configurar el título y la descripción de la aplicación
    st.title("Chatbot de la película 'Leo'")
    st.write("Hacé preguntas sobre la película y obtené respuestas basadas en el contenido de 'Leo.txt'.")

    # Inicializar la sesión para guardar el historial
    if "historial" not in st.session_state:
        st.session_state.historial = []  # Lista para almacenar el historial de preguntas y respuestas

    # Inicializar el estado del input
    if "input_pregunta" not in st.session_state:
        st.session_state.input_pregunta = ""  # Valor inicial vacío para el campo de entrada

    # Función para procesar la pregunta
    def procesar_pregunta():
        if st.session_state.input_pregunta.strip():  # Verificar si el input no está vacío
            # Generar la respuesta
            respuesta = responder_pregunta(st.session_state.input_pregunta, index, trozos)

            # Guardar la pregunta y la respuesta en el historial
            st.session_state.historial.append({
                "pregunta": st.session_state.input_pregunta,
                "respuesta": respuesta
            })

            # Limpiar el campo de entrada
            st.session_state.input_pregunta = ""  # Restablecer a vacío
        else:
            st.warning("Por favor, ingresá una pregunta válida.")

    # Campo de entrada para la pregunta del usuario
    with st.form("my_form"):
        st.text_input(
            "Hacé una pregunta sobre la película:",
            key="input_pregunta",
            placeholder="Tu pregunta aquí..."
        )
        st.form_submit_button("Enviar", on_click=procesar_pregunta)

    # Mostrar el historial de preguntas y respuestas
    # st.write("### Historial de Preguntas y Respuestas")
    for intercambio in reversed(st.session_state.historial):
        st.write(f"**Pregunta:** {intercambio['pregunta']}")
        with st.spinner('Generando respuesta...'):
            st.write(f"**Respuesta:** {intercambio['respuesta']}")
        st.markdown("---")  # Separador entre interacciones

    # # Aplicar un estilo personalizado
    # st.markdown("""
    # <style>
    # .stApp {
    # font-family: 'Helvetica Neue', sans-serif;
    # color: #333;
    # background-color: #f5f5f5;
    # }
    # .stTextInput > div > div > input {
    # border: 1px solid #ccc;
    # border-radius: 5px;
    # padding: 10px;
    # }
    # </style>
    # """, unsafe_allow_html=True)