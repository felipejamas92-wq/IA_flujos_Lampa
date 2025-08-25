import streamlit as st
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import PyPDF2
import docx

# ==== AUTENTICACIÓN GOOGLE DRIVE ====
st.set_page_config(page_title="Chat con tus documentos", layout="wide")
st.title("📖 Chat con tus documentos (multiusuario)")

gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Esto abrirá el navegador para autenticar
drive = GoogleDrive(gauth)

# ==== FUNCIONES DE LECTURA DE ARCHIVOS ====
def leer_txt(file):
    return file.read().decode("utf-8")

def leer_pdf(file):
    reader = PyPDF2.PdfReader(file)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"
    return texto

def leer_docx(file):
    doc = docx.Document(file)
    texto = "\n".join([p.text for p in doc.paragraphs])
    return texto

def cargar_archivo(file):
    if file.name.endswith(".txt"):
        return leer_txt(file)
    elif file.name.endswith(".pdf"):
        return leer_pdf(file)
    elif file.name.endswith(".doc") or file.name.endswith(".docx"):
        return leer_docx(file)
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
            contenido = cargar_archivo(uploaded_file)
            if contenido:
                gfile = drive.CreateFile({'title': uploaded_file.name})
                gfile.SetContentString(contenido)
                gfile.Upload()
                st.success(f"Archivo '{uploaded_file.name}' subido y guardado permanentemente en Google Drive!")
            else:
                st.error("Formato no soportado")
    else:
        st.sidebar.error("❌ Clave incorrecta")

# ==== USUARIO ====
st.header("📄 Consultar documentos")

# Listar todos los archivos de Google Drive
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
if not file_list:
    st.warning("⚠️ No hay documentos en Drive. Espera que el administrador suba archivos.")
else:
    documentos = [f.GetContentString() for f in file_list]
    nombres = [f['title'] for f in file_list]

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

    # Pregunta del usuario
    pregunta = st.text_input("❓ Escribe tu pregunta aquí:")
    if pregunta:
        respuesta = responder_pregunta(pregunta, documentos, embeddings, modelo_embeddings, generador)
        st.subheader("🧠 Respuesta:")
        st.write(respuesta)
