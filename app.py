import streamlit as st
import os
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ==== CONFIGURACIÓN ====
CARPETA_DOCS = "documentos"
os.makedirs(CARPETA_DOCS, exist_ok=True)

st.set_page_config(page_title="Chat con tus documentos", layout="wide")
st.title("📖 Chat con tus documentos (multiusuario)")

# ==== FUNCIONES DE LECTURA DE ARCHIVOS ====
def leer_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def leer_pdf(file_path):
    texto = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            texto += page.extract_text() + "\n"
    return texto

def leer_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def cargar_archivo(file_path):
    if file_path.endswith(".txt"):
        return leer_txt(file_path)
    elif file_path.endswith(".pdf"):
        return leer_pdf(file_path)
    elif file_path.endswith(".doc") or file_path.endswith(".docx"):
        return leer_docx(file_path)
    else:
        return None

# ==== IA ====
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

# ==== ROLES ====
rol = st.sidebar.radio("Selecciona rol:", ["Usuario", "Administrador"])

if rol == "Administrador":
    password = st.sidebar.text_input("🔑 Clave de administrador", type="password")
    if password == "mi_clave_segura":  # <<--- cámbiala por tu propia clave
        st.sidebar.success("✅ Acceso como Administrador")

        uploaded_file = st.file_uploader("Sube tus archivos (.txt, .pdf, .docx)", type=["txt", "pdf", "doc", "docx"])
        if uploaded_file:
            file_path = os.path.join(CARPETA_DOCS, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Archivo '{uploaded_file.name}' guardado en carpeta local ✅")
            st.info("👉 Recuerda hacer `git add . && git commit && git push` para guardarlo en GitHub.")
    else:
        st.sidebar.error("❌ Clave incorrecta")

# ==== USUARIO ====
st.header("📄 Consultar documentos")

archivos = [os.path.join(CARPETA_DOCS, f) for f in os.listdir(CARPETA_DOCS)]

if not archivos:
    st.warning("⚠️ No hay documentos. Espera que el administrador suba archivos y los sincronice con GitHub.")
else:
    documentos = [cargar_archivo(a) for a in archivos if cargar_archivo(a)]
    nombres = [os.path.basename(a) for a in archivos]

    # Crear embeddings
    if "modelo_embeddings" not in st.session_state:
        st.write("🔎 Generando embeddings de los documentos...")
        modelo_embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = crear_embeddings(documentos, modelo_embeddings)
        st.session_state["modelo_embeddings"] = modelo_embeddings
        st.session_state["embeddings"] = embeddings
    else:
        modelo_embeddings = st.session_state["modelo_embeddings"]
        embeddings = st.session_state["embeddings"]

    # Cargar modelo de lenguaje
    if "generador" not in st.session_state:
        st.write("⏳ Cargando modelo de lenguaje (Mistral 7B Instruct)...")
        st.session_state["generador"] = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            device_map="auto"
        )
    generador = st.session_state["generador"]

    # Selección de documentos (opcional)
    seleccionados = st.multiselect("📂 Selecciona documentos para consultar:", nombres, default=nombres)

    documentos_filtrados = [documentos[i] for i in range(len(nombres)) if nombres[i] in seleccionados]
    embeddings_filtrados = crear_embeddings(documentos_filtrados, modelo_embeddings)

    # Pregunta del usuario
    pregunta = st.text_input("❓ Escribe tu pregunta aquí:")
    if pregunta and documentos_filtrados:
        respuesta = responder_pregunta(pregunta, documentos_filtrados, embeddings_filtrados, modelo_embeddings, generador)
        st.subheader("🧠 Respuesta:")
        st.write(respuesta)

