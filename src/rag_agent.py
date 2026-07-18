from pathlib import Path
import re
import unicodedata

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ============================================================
# Configuración general
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Modelo ligero para una respuesta más rápida en OCI.
OLLAMA_MODEL = "gemma3:1b"

# Optimización del RAG.
MAX_RETRIEVED_DOCUMENTS = 4
RETRIEVAL_CANDIDATES = 8
MAX_CHARS_PER_DOCUMENT = 1200

MAX_SOURCES_TO_DISPLAY = 3
SHOW_RETRIEVED_SOURCES_IN_TERMINAL = True

NO_INFORMATION_BASE = (
    "No encontré esa información en la documentación disponible."
)

NO_INFORMATION_MESSAGE = (
    f"{NO_INFORMATION_BASE} "
    "Por favor, ponte en contacto con el equipo de soporte "
    "por correo electrónico."
)


# ============================================================
# Utilidades de texto
# ============================================================

def normalize_text(text):
    """
    Convierte el texto a minúsculas, elimina acentos,
    signos innecesarios y espacios duplicados.
    """

    normalized = unicodedata.normalize(
        "NFKD",
        str(text).strip().lower(),
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    normalized = re.sub(
        r"[¿?¡!.,;:()\"']",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def answer_indicates_no_information(answer):
    """
    Detecta distintas formas en que el modelo puede indicar
    que no encontró información.
    """

    normalized_answer = normalize_text(answer)

    variants = [
        "no encontre esa informacion en la documentacion disponible",
        "no encontre informacion en la documentacion disponible",
        "no se encontro esa informacion en la documentacion disponible",
        "la documentacion disponible no contiene esa informacion",
        "no hay informacion suficiente en el contexto",
        "el contexto no contiene informacion suficiente",
        "no encontre informacion relevante",
        "no se proporciona informacion suficiente",
    ]

    return any(
        variant in normalized_answer
        for variant in variants
    )


# ============================================================
# Carga de recursos
# ============================================================

def load_embeddings():
    """Carga el modelo utilizado para generar los embeddings."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def load_vector_store(embeddings):
    """Carga el índice FAISS generado previamente."""

    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            "No se encontró la base vectorial. "
            "Ejecuta primero: python src/create_vector_store.py"
        )

    return FAISS.load_local(
        folder_path=str(VECTOR_STORE_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def load_language_model():
    """
    Inicializa Gemma 3 mediante Ollama.

    keep_alive conserva el modelo cargado para evitar
    recargarlo completamente en cada consulta.
    """

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=220,
        keep_alive="30m",
    )


# ============================================================
# Expansión de consultas
# ============================================================

def expand_search_query(question):
    """
    Agrega términos equivalentes a ciertas preguntas para mejorar
    la recuperación semántica.

    La pregunta original se conserva para que Gemma responda
    exactamente a lo solicitado. La versión ampliada se utiliza
    únicamente para buscar en FAISS.
    """

    normalized = normalize_text(question)

    # --------------------------------------------------------
    # Creación o registro de usuarios
    # --------------------------------------------------------

    user_creation_patterns = [
        r"\bcomo creo un usuario\b",
        r"\bcomo crear un usuario\b",
        r"\bcomo registro un usuario\b",
        r"\bcomo registrar un usuario\b",
        r"\bcrear un usuario\b",
        r"\bcrear usuario\b",
        r"\bregistrar un usuario\b",
        r"\bregistrar usuario\b",
        r"\bnuevo usuario\b",
        r"\bagregar una cuenta de usuario\b",
        r"\bcrear una cuenta de usuario\b",
        r"\bcuenta de usuario\b",
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in user_creation_patterns
    ):
        return (
            f"{question}\n\n"
            "Registrar un nuevo usuario en el Dashboard. "
            "Administración de Usuarios. "
            "Seleccionar la opción Nuevo Usuario. "
            "Capturar la información solicitada. "
            "Seleccionar el rol correspondiente. "
            "Guardar la información."
        )

    # --------------------------------------------------------
    # Registro de nuevas prendas
    # --------------------------------------------------------

    garment_registration_patterns = [
        r"\bregistrar una nueva prenda\b",
        r"\bregistrar nuevas prendas\b",
        r"\bdar de alta una prenda\b",
        r"\bagregar una prenda nueva\b",
        r"\bcrear una prenda\b",
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in garment_registration_patterns
    ):
        return (
            f"{question}\n\n"
            "Registrar nuevas prendas en la aplicación Android. "
            "Seleccionar el tipo de prenda. "
            "Iniciar la lectura RFID. "
            "Validar las etiquetas detectadas. "
            "Confirmar el registro."
        )

    # --------------------------------------------------------
    # Localización de prendas
    # --------------------------------------------------------

    garment_location_patterns = [
        r"\blocalizar una prenda\b",
        r"\bbuscar una prenda\b",
        r"\bencontrar una prenda\b",
        r"\bdonde esta una prenda\b",
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in garment_location_patterns
    ):
        return (
            f"{question}\n\n"
            "Localizar prendas mediante RFID. "
            "Iniciar lectura. "
            "Validar etiquetas. "
            "Consultar nombre, ubicación y RFID."
        )

    # --------------------------------------------------------
    # Problemas de lectura RFID
    # --------------------------------------------------------

    rfid_detection_patterns = [
        r"\bno detecta etiquetas\b",
        r"\bno se detectan etiquetas\b",
        r"\blector no detecta\b",
        r"\brfid no lee\b",
        r"\bno lee las prendas\b",
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in rfid_detection_patterns
    ):
        return (
            f"{question}\n\n"
            "El lector RFID no detecta etiquetas. "
            "Verificar funcionamiento del lector. "
            "Revisar potencia, distancia, alcance y estado "
            "de las etiquetas."
        )

    return question


# ============================================================
# Recuperación de documentos
# ============================================================

def _document_key(document):
    """Genera una clave para evitar fragmentos duplicados."""

    file_name = document.metadata.get(
        "file_name",
        "",
    )

    page = document.metadata.get("page")

    normalized_content = normalize_text(
        document.page_content
    )[:500]

    return (
        file_name,
        page,
        normalized_content,
    )


def retrieve_documents(vector_store, question):
    """
    Recupera fragmentos utilizando tanto la pregunta original
    como una versión ampliada.

    Después combina, ordena y elimina resultados repetidos.
    """

    expanded_question = expand_search_query(question)

    search_queries = [question]

    if expanded_question != question:
        search_queries.append(expanded_question)

    combined_results = {}

    for query_index, search_query in enumerate(search_queries):
        results_with_scores = (
            vector_store.similarity_search_with_score(
                query=search_query,
                k=RETRIEVAL_CANDIDATES,
            )
        )

        for document, score in results_with_scores:
            key = _document_key(document)
            numeric_score = float(score)

            if (
                key not in combined_results
                or numeric_score < combined_results[key][1]
            ):
                document.metadata[
                    "retrieval_score"
                ] = numeric_score

                document.metadata[
                    "retrieval_query"
                ] = (
                    "original"
                    if query_index == 0
                    else "expanded"
                )

                combined_results[key] = (
                    document,
                    numeric_score,
                )

    ordered_results = sorted(
        combined_results.values(),
        key=lambda item: item[1],
    )

    documents = [
        document
        for document, _score in ordered_results[
            :MAX_RETRIEVED_DOCUMENTS
        ]
    ]

    return documents


# ============================================================
# Construcción del contexto
# ============================================================

def get_page_number(document):
    """Obtiene el número de página visible para el usuario."""

    page = document.metadata.get("page")

    if isinstance(page, int):
        return page + 1

    return None


def build_context(documents):
    """
    Construye el contexto enviado a Gemma.

    Limita cada fragmento para reducir el tiempo de evaluación
    del prompt sin modificar el índice FAISS ni los documentos.
    """

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )

        page = get_page_number(document)

        page_text = (
            str(page)
            if page is not None
            else "no disponible"
        )

        source_header = (
            f"[Fuente {index}: "
            f"{file_name}, página {page_text}]"
        )

        content = document.page_content.strip()[
            :MAX_CHARS_PER_DOCUMENT
        ]

        context_parts.append(
            f"{source_header}\n{content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# Prompts
# ============================================================

def build_prompt(question, context):
    """Construye el prompt principal."""

    return f"""
Eres un asistente especializado en el sistema IDLinens HA.

Responde preguntas sobre la aplicación Android, el Dashboard
y los procesos operativos utilizando exclusivamente el contexto
recuperado de los manuales oficiales.

INSTRUCCIONES OBLIGATORIAS:

1. Responde siempre en español.
2. Analiza todos los fragmentos recuperados.
3. Reconoce sinónimos y expresiones equivalentes.
4. La pregunta no necesita utilizar exactamente las palabras
   empleadas en los manuales.
5. Ejemplos:
   - "crear un usuario" equivale a "registrar un nuevo usuario";
   - "buscar una prenda" equivale a "localizar una prenda";
   - "dar de alta una prenda" equivale a
     "registrar una nueva prenda".
6. Cuando se solicite un procedimiento, identifica los pasos,
   acciones o instrucciones relacionados.
7. Presenta los procedimientos con una lista numerada.
8. Responde únicamente la operación solicitada.
9. No confundas:
   - iniciar sesión con crear un usuario;
   - registrar una prenda con ponerla en circulación;
   - buscar una prenda con editar un activo.
10. No inventes botones, campos, funciones, precios,
    requisitos ni resultados.
11. No utilices conocimientos externos.
12. No agregues fuentes. El sistema las añadirá.
13. Si ninguno de los fragmentos contiene información útil,
    responde exactamente:

"{NO_INFORMATION_BASE}"

CONTEXTO:

{context}

PREGUNTA:

{question}

RESPUESTA:
""".strip()


def build_retry_prompt(question, context):
    """Construye un segundo prompt más directo."""

    return f"""
Responde la pregunta usando exclusivamente el contexto.

Reconoce sinónimos y expresiones equivalentes.
Busca pasos, procedimientos y acciones relacionados.

No inventes información.
Responde siempre en español.
No agregues fuentes.

Si verdaderamente no existe información útil, responde:

"{NO_INFORMATION_BASE}"

CONTEXTO:

{context}

PREGUNTA:

{question}

RESPUESTA:
""".strip()


# ============================================================
# Generación de respuestas
# ============================================================

def generate_answer(model, prompt):
    """Envía el prompt a Gemma y devuelve texto limpio."""

    response = model.invoke(prompt)

    content = getattr(
        response,
        "content",
        "",
    )

    return str(content).strip()


# ============================================================
# Fuentes
# ============================================================

def format_sources(documents):
    """Genera una lista de fuentes sin duplicados."""

    sources = []
    seen = set()

    for document in documents:
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )

        page = get_page_number(document)

        source_key = (
            file_name,
            page,
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        page_text = (
            str(page)
            if page is not None
            else "no disponible"
        )

        sources.append(
            f"- {file_name}, página {page_text}"
        )

        if len(sources) >= MAX_SOURCES_TO_DISPLAY:
            break

    if not sources:
        return ""

    return (
        "\n\n### Fuentes consultadas\n"
        + "\n".join(sources)
    )


def show_retrieved_sources(documents):
    """Muestra en terminal las fuentes recuperadas."""

    if not SHOW_RETRIEVED_SOURCES_IN_TERMINAL:
        return

    print("\nFragmentos recuperados:")

    for index, document in enumerate(
        documents,
        start=1,
    ):
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )

        page = get_page_number(document)

        page_text = (
            str(page)
            if page is not None
            else "no disponible"
        )

        score = document.metadata.get(
            "retrieval_score"
        )

        query_type = document.metadata.get(
            "retrieval_query",
            "original",
        )

        score_text = (
            f" | distancia: {score:.4f}"
            if isinstance(score, float)
            else ""
        )

        print(
            f"{index}. {file_name} - "
            f"página {page_text}"
            f"{score_text} | búsqueda: {query_type}"
        )


# ============================================================
# Flujo completo del agente
# ============================================================

def answer_question(
    vector_store,
    model,
    question,
):
    """
    Ejecuta el flujo completo:

    pregunta → recuperación → contexto → Gemma → fuentes.
    """

    clean_question = str(question).strip()

    if not clean_question:
        return "Debes escribir una pregunta."

    documents = retrieve_documents(
        vector_store=vector_store,
        question=clean_question,
    )

    if not documents:
        return NO_INFORMATION_MESSAGE

    show_retrieved_sources(documents)

    context = build_context(documents)

    prompt = build_prompt(
        question=clean_question,
        context=context,
    )

    answer = generate_answer(
        model=model,
        prompt=prompt,
    )

    # Solo realiza un segundo intento cuando el modelo afirma
    # que la documentación no contiene información.
    if answer_indicates_no_information(answer):
        reduced_documents = documents[:3]

        reduced_context = build_context(
            reduced_documents
        )

        retry_prompt = build_retry_prompt(
            question=clean_question,
            context=reduced_context,
        )

        retry_answer = generate_answer(
            model=model,
            prompt=retry_prompt,
        )

        if not answer_indicates_no_information(
            retry_answer
        ):
            answer = retry_answer
            documents = reduced_documents

    if answer_indicates_no_information(answer):
        return NO_INFORMATION_MESSAGE

    sources = format_sources(documents)

    return answer.strip() + sources


# ============================================================
# Ejecución desde terminal
# ============================================================

def main():
    try:
        print("\n" + "=" * 70)
        print("IDLINENS RAG ASSISTANT")
        print("=" * 70)
        print("Escribe una pregunta sobre IDLinens HA.")
        print("Escribe 'salir' para finalizar.\n")

        print("Cargando modelo de embeddings...")
        embeddings = load_embeddings()

        print("Cargando base vectorial...")
        vector_store = load_vector_store(
            embeddings
        )

        print(
            "Conectando con Gemma 3 "
            "mediante Ollama..."
        )

        model = load_language_model()

        print("\nAgente listo.\n")

        while True:
            question = input(
                "Tu pregunta:\n> "
            ).strip()

            if not question:
                print(
                    "\nDebes escribir "
                    "una pregunta.\n"
                )
                continue

            if normalize_text(question) in {
                "salir",
                "exit",
                "quit",
            }:
                print("\nAgente finalizado.")
                break

            print(
                "\nBuscando información "
                "y generando respuesta...\n"
            )

            answer = answer_question(
                vector_store=vector_store,
                model=model,
                question=question,
            )

            print("\n" + "=" * 70)
            print("RESPUESTA")
            print("=" * 70)
            print(answer)
            print("=" * 70 + "\n")

    except FileNotFoundError as error:
        print(f"\nError: {error}")

    except KeyboardInterrupt:
        print("\n\nAgente finalizado.")

    except Exception as error:
        print(
            "\nOcurrió un error "
            "al ejecutar el agente."
        )
        print(f"Detalle: {error}")


if __name__ == "__main__":
    main()