from pathlib import Path
import re
import time
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

# Modelo ligero para Oracle Cloud.
OLLAMA_MODEL = "gemma3:1b"

# Recuperación y contexto.
RETRIEVAL_CANDIDATES = 10
MAX_RETRIEVED_DOCUMENTS = 4
MAX_CHARS_PER_DOCUMENT = 1400
MAX_SOURCES_TO_DISPLAY = 3

# Diagnóstico.
SHOW_RETRIEVED_SOURCES_IN_TERMINAL = True
SHOW_PERFORMANCE_TIMES = True

# Caché simple en memoria para preguntas repetidas.
ENABLE_ANSWER_CACHE = True
MAX_CACHE_ENTRIES = 100
ANSWER_CACHE = {}

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
        r"[¿?¡!.,;:()\[\]{}\"']",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()

def get_verified_direct_answer(question):
    """
    Respuestas verificadas para consultas operativas que pueden
    ser interpretadas incorrectamente por el modelo pequeño.
    """

    normalized = normalize_text(question)

    asset_history_patterns = [
        r"\bhistorial de un activo\b",
        r"\bhistorial del activo\b",
        r"\bver el historial\b",
        r"\bver historial\b",
        r"\bconsultar historial\b",
        r"\bhistorico de un activo\b",
        r"\bhistorico del activo\b",
        r"\bmovimientos de un activo\b",
        r"\bentradas y salidas de un activo\b",
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in asset_history_patterns
    ):
        return """
Para consultar el historial de un activo:

1. Ingrese al módulo **Administrar activos** del Dashboard.
2. Localice el activo que desea consultar.
3. Seleccione el **ícono de lápiz o edición** correspondiente al activo.
4. Dentro del detalle del activo, abra la pestaña **Histórico**.
5. Revise los movimientos registrados, como entradas, salidas y cambios de ubicación.

### Fuentes consultadas
- Manual_Usuario_Dashboard.pdf, página 15
- Manual_Usuario_Dashboard.pdf, página 9
""".strip()

    return None


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

    keep_alive evita recargar el modelo entre preguntas.
    num_thread aprovecha las cuatro OCPU de la VM.
    """

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=220,
        num_ctx=4096,
        num_thread=4,
        keep_alive="60m",
    )


# ============================================================
# Expansión de consultas
# ============================================================

def expand_search_query(question):
    """
    Agrega términos equivalentes a preguntas conocidas.

    La ampliación se usa solamente para buscar en FAISS.
    La pregunta original sigue siendo la que recibe Gemma.
    """

    normalized = normalize_text(question)

    # Creación de usuarios.
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
            "Administración de usuarios. Nuevo usuario. "
            "Capturar información. Seleccionar rol. Guardar."
        )

    # Registro de prendas.
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
            "Seleccionar tipo de prenda. Iniciar lectura RFID. "
            "Validar etiquetas detectadas. Confirmar registro."
        )

    # Localización de prendas.
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
            "Localizar prendas mediante RFID. Iniciar lectura. "
            "Validar etiquetas. Consultar nombre, ubicación y RFID."
        )

    # Historial de activos.
    asset_history_patterns = [
        r"\bhistorial de un activo\b",
        r"\bhistorial del activo\b",
        r"\bver el historial\b",
        r"\bver historial\b",
        r"\bconsultar historial\b",
        r"\bhistorico de un activo\b",
        r"\bhistorico del activo\b",
        r"\bmovimientos de un activo\b",
        r"\bentradas y salidas de un activo\b",
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in asset_history_patterns
    ):
        return (
            f"{question}\n\n"
            "Administrar activos. Localizar el activo. "
            "Seleccionar el icono de edición o lápiz. "
            "Abrir el detalle. Seleccionar la pestaña Histórico. "
            "Consultar movimientos, entradas y salidas."
        )

    # Potencia del lector.
    reader_power_patterns = [
        r"\bcambiar la potencia\b",
        r"\bajustar la potencia\b",
        r"\bconfigurar la potencia\b",
        r"\bpotencia de la lectora\b",
        r"\bpotencia del lector\b",
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in reader_power_patterns
    ):
        return (
            f"{question}\n\n"
            "Configurador de potencia del lector RFID. "
            "Icono de lápiz. Barra deslizante. Aplicar potencia."
        )

    # Problemas de lectura RFID.
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
            "Verificar lector, potencia, distancia, alcance, "
            "orientación y estado de las etiquetas."
        )

    return question


# ============================================================
# Recuperación de documentos
# ============================================================

def document_key(document):
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
    Recupera fragmentos usando la pregunta original y,
    cuando aplica, una versión ampliada.

    Combina los resultados, elimina duplicados y conserva
    los fragmentos con menor distancia.
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
            key = document_key(document)
            numeric_score = float(score)

            previous = combined_results.get(key)

            if (
                previous is None
                or numeric_score < previous[1]
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

    return [
        document
        for document, _score in ordered_results[
            :MAX_RETRIEVED_DOCUMENTS
        ]
    ]


# ============================================================
# Construcción del contexto
# ============================================================

def get_page_number(document):
    """Obtiene el número de página visible para el usuario."""

    page = document.metadata.get("page")

    if isinstance(page, int):
        return page + 1

    return None


def clean_document_text(text):
    """Limpia saltos excesivos sin alterar el contenido."""

    clean_text = re.sub(
        r"\r\n?",
        "\n",
        str(text),
    ).strip()

    clean_text = re.sub(
        r"[ \t]+",
        " ",
        clean_text,
    )

    clean_text = re.sub(
        r"\n{3,}",
        "\n\n",
        clean_text,
    )

    return clean_text


def build_context(documents):
    """
    Construye un contexto compacto conservando el fragmento
    original recuperado por FAISS.

    No selecciona oraciones aisladas para evitar perder
    procedimientos completos.
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

        content = clean_document_text(
            document.page_content
        )[:MAX_CHARS_PER_DOCUMENT]

        context_parts.append(
            f"[Fuente {index}: {file_name}, "
            f"página {page_text}]\n{content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# Prompt
# ============================================================

def build_prompt(question, context):
    """
    Construye un prompt compacto que prioriza respuestas
    completas, precisas y operativas.
    """

    return f"""
Eres el asistente especializado en IDLinens HA.

Debes responder EXCLUSIVAMENTE con la información incluida
en el contexto recuperado de los manuales.

REGLAS OBLIGATORIAS:

1. Responde siempre en español.
2. Responde exactamente lo que preguntó el usuario.
3. Reconoce sinónimos y expresiones equivalentes.
4. Lee todos los fragmentos antes de responder.
5. Si la pregunta solicita cómo hacer algo, ver, consultar,
   crear, registrar, cambiar, configurar, editar o eliminar,
   proporciona las acciones concretas en una lista numerada.
6. Si el contexto contiene una descripción, un objetivo y un
   procedimiento, utiliza el procedimiento para responder.
7. No respondas únicamente con el objetivo, introducción o
   descripción cuando existan instrucciones de uso.
8. Ignora textos sobre permisos, errores, soporte técnico,
   advertencias o recomendaciones, salvo que el usuario pregunte
   específicamente por ellos.
9. Conserva los nombres reales de botones, módulos, pestañas,
   menús y opciones que aparezcan en el contexto.
10. No confundas:
    - iniciar sesión con crear un usuario;
    - registrar una prenda con ponerla en circulación;
    - localizar una prenda con editar un activo;
    - consultar historial con solucionar un problema.
11. No inventes botones, campos, funciones, requisitos,
    resultados ni pasos.
12. No utilices conocimientos externos.
13. No agregues fuentes; el sistema las mostrará automáticamente.
14. Si el contexto no contiene la respuesta, escribe exactamente:

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
# Caché
# ============================================================

def get_cached_answer(question):
    """Obtiene una respuesta guardada para una pregunta repetida."""

    if not ENABLE_ANSWER_CACHE:
        return None

    return ANSWER_CACHE.get(
        normalize_text(question)
    )


def save_cached_answer(question, answer):
    """Guarda una respuesta y limita el tamaño de la caché."""

    if not ENABLE_ANSWER_CACHE:
        return

    cache_key = normalize_text(question)

    if len(ANSWER_CACHE) >= MAX_CACHE_ENTRIES:
        oldest_key = next(iter(ANSWER_CACHE))
        ANSWER_CACHE.pop(oldest_key, None)

    ANSWER_CACHE[cache_key] = answer


# ============================================================
# Flujo completo del agente
# ============================================================

def answer_question(
    vector_store,
    model,
    question,
):
    """
    Ejecuta el flujo:

    pregunta → FAISS → contexto → Gemma → fuentes.
    """

    total_start = time.perf_counter()

    clean_question = str(question).strip()

    if not clean_question:
        return "Debes escribir una pregunta."

        verified_answer = get_verified_direct_answer(
        clean_question
    )

    if verified_answer is not None:
        return verified_answer

    cached_answer = get_cached_answer(
        clean_question
    )

    if cached_answer is not None:
        if SHOW_PERFORMANCE_TIMES:
            print(
                "\nRespuesta recuperada desde caché."
            )

        return cached_answer

    retrieval_start = time.perf_counter()

    documents = retrieve_documents(
        vector_store=vector_store,
        question=clean_question,
    )

    retrieval_seconds = (
        time.perf_counter() - retrieval_start
    )

    if not documents:
        return NO_INFORMATION_MESSAGE

    show_retrieved_sources(documents)

    context_start = time.perf_counter()

    context = build_context(documents)

    context_seconds = (
        time.perf_counter() - context_start
    )

    prompt = build_prompt(
        question=clean_question,
        context=context,
    )

    generation_start = time.perf_counter()

    answer = generate_answer(
        model=model,
        prompt=prompt,
    )

    generation_seconds = (
        time.perf_counter() - generation_start
    )

    if answer_indicates_no_information(answer):
        final_answer = NO_INFORMATION_MESSAGE
    else:
        final_answer = (
            answer.strip()
            + format_sources(documents)
        )

    save_cached_answer(
        question=clean_question,
        answer=final_answer,
    )

    if SHOW_PERFORMANCE_TIMES:
        total_seconds = (
            time.perf_counter() - total_start
        )

        print("\nTiempos de ejecución:")
        print(
            f"- Recuperación FAISS: "
            f"{retrieval_seconds:.2f} s"
        )
        print(
            f"- Preparación del contexto: "
            f"{context_seconds:.2f} s"
        )
        print(
            f"- Generación con Gemma: "
            f"{generation_seconds:.2f} s"
        )
        print(
            f"- Tiempo total: "
            f"{total_seconds:.2f} s"
        )
        print(
            f"- Caracteres del contexto: "
            f"{len(context)}"
        )

    return final_answer


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