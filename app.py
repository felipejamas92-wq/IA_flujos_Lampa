import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ==== FUNCIONES ====
def crear_embeddings(documentos, modelo_embeddings):
    return modelo_embeddings.encode(documentos)

def buscar_contexto(pregunta, documentos, embeddings, modelo_embeddings, top_k=2):
    embedding_pregunta = modelo_embeddings.encode([pregunta])
    similitudes = cosine_similarity(embedding_pregunta, embeddings)[0]
    indices = similitudes.argsort()[-top_k:][::-1]
    contexto = "\n\n".join([documentos[i][:1000] for i in indices])
    return contexto

def responder_pregunta(pregunta, documentos, embeddings, modelo_embeddings, generador):
    contexto = buscar_contexto(pregunta, documentos, embeddings, modelo_embeddings)
    prompt = f"Responde en español basándote SOLO en el siguiente texto:\n\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"
    respuesta = generador(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )[0]["generated_text"]
    return respuesta

# ==== INTERFAZ WEB ====
st.set_page_config(page_title="Chat con tus documentos", layout="wide")
st.title("📖 Chat con tus documentos")
st.write("Esta app permite subir archivos .txt y hacer preguntas basadas en ellos.")

# ---- ROLES ----
rol = st.sidebar.radio("Selecciona rol:", ["Usuario", "Administrador"])

if rol == "Administrador":
    password = st.sidebar.text_input("🔑 Clave de administrador", type="password")
    if password == "mi_clave_segura":  # <<--- cámbiala por una propia
        st.sidebar.success("✅ Acceso como Administrador")
        archivos = st.file_uploader("Sube tus archivos de texto", type="txt", accept_multiple_files=True)

        if archivos:
            documentos = [archivo.read().decode("utf-8") for archivo in archivos]
            st.session_state["documentos"] = documentos
            st.success("📂 Archivos cargados correctamente. Ahora los usuarios pueden hacer preguntas.")
    else:
        st.sidebar.error("❌ Clave incorrecta")

# ---- USUARIO ----
if rol == "Usuario":
    if "documentos" in st.session_state:
        documentos = st.session_state["documentos"]

        # Crear embeddings (una sola vez)
        if "embeddings" not in st.session_state:
            st.write("🔎 Generando embeddings...")
            modelo_embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = crear_embeddings(documentos, modelo_embeddings)
            st.session_state["modelo_embeddings"] = modelo_embeddings
            st.session_state["embeddings"] = embeddings
        else:
            modelo_embeddings = st.session_state["modelo_embeddings"]
            embeddings = st.session_state["embeddings"]

        # Cargar modelo de lenguaje (solo 1 vez)
        if "generador" not in st.session_state:
            st.write("⏳ Cargando modelo de lenguaje (Mistral 7B Instruct)...")
            st.session_state["generador"] = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2",
                device_map="auto"
            )

        generador = st.session_state["generador"]

        # Pregunta del usuario
        pregunta = st.text_input("❓ Escribe tu pregunta aquí:")
        if pregunta:
            respuesta = responder_pregunta(pregunta, documentos, embeddings, modelo_embeddings, generador)
            st.subheader("🧠 Respuesta:")
            st.write(respuesta)

    else:
        st.warning("⚠️ Aún no hay documentos cargados. Pide al administrador que suba archivos.")
