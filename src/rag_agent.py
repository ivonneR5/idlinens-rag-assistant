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

# Modelo ligero utilizado en Oracle Cloud.
OLLAMA_MODEL = "gemma3:1b"

# Recuperación:
# buscamos varios candidatos, pero solo enviamos los mejores.
RETRIEVAL_CANDIDATES = 6
MAX_RETRIEVED_DOCUMENTS = 3

# Cantidad máxima enviada a Gemma por cada fragmento.
MAX_CHARS_PER_DOCUMENT = 750

MAX_SOURCES_TO_DISPLAY = 3
SHOW_RETRIEVED_SOURCES_IN_TERMINAL = True
SHOW_PERFORMANCE_TIMES = True

# Caché en memoria para preguntas repetidas.
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

# Palabras demasiado comunes que no ayudan a seleccionar contexto.
STOPWORDS = {
    "a",
    "al",
    "algo",
    "como",
    "con",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "el",
    "ella",
    "en",
    "es",
    "esa",
    "ese",
    "esta",
    "este",
    "hacer",
    "hay",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "su",
    "sus",
    "un",
    "una",
    "y",
}


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


def extract_keywords(text):
    """
    Obtiene términos útiles para identificar las partes
    del contexto más relacionadas con la pregunta.
    """

    normalized = normalize_text(text)

    words = re.findall(
        r"\b[a-z0-9_-]{3,}\b",
        normalized,
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
    }


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

    keep_alive evita que el modelo se descargue de memoria
    entre preguntas. num_thread aprovecha las cuatro OCPU.
    """

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=180,
        num_ctx=2048,
        num_thread=4,
        keep_alive="60m",
    )


# ============================================================
# Expansión de consultas
# ============================================================

def expand_search_query(question):
    """
    Agrega términos equivalentes solamente cuando son necesarios.

    La pregunta original se conserva para que Gemma responda
    exactamente a lo solicitado. La ampliación solo ayuda a FAISS.
    """

    normalized = normalize_text(question)

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
            f"{question}. "
            "Registrar nuevo usuario en Dashboard, "
            "administración de usuarios, nuevo usuario, "
            "datos, rol y guardar."
        )

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
            f"{question}. "
            "Registrar prendas nuevas en aplicación Android, "
            "tipo de prenda, lectura RFID y confirmar."
        )

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
            f"{question}. "
            "Localizar prendas, lectura RFID, validar etiquetas, "
            "nombre y ubicación."
        )

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
            f"{question}. "
            "Lector RFID no detecta etiquetas, potencia, "
            "distancia, alcance y estado de etiquetas."
        )

    return question


# ============================================================
# Recuperación de documentos
# ============================================================

def document_key(document):
    """Genera una clave para eliminar fragmentos duplicados."""

    file_name = document.metadata.get(
        "file_name",
        "",
    )

    page = document.metadata.get("page")

    normalized_content = normalize_text(
        document.page_content
    )[:400]

    return (
        file_name,
        page,
        normalized_content,
    )


def retrieve_documents(vector_store, question):
    """
    Recupera candidatos mediante FAISS, elimina duplicados
    y conserva únicamente los fragmentos más cercanos.
    """

    expanded_question = expand_search_query(question)

    queries = [question]

    if expanded_question != question:
        queries.append(expanded_question)

    combined_results = {}

    for query_index, search_query in enumerate(queries):
        results = vector_store.similarity_search_with_score(
            query=search_query,
            k=RETRIEVAL_CANDIDATES,
        )

        for document, score in results:
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
# Selección de contenido relevante
# ============================================================

def split_text_units(text):
    """
    Divide un fragmento en párrafos, instrucciones,
    títulos y oraciones.
    """

    clean_text = re.sub(
        r"\r\n?",
        "\n",
        str(text),
    )

    raw_units = re.split(
        r"\n+|(?<=[.!?])\s+",
        clean_text,
    )

    return [
        re.sub(r"\s+", " ", unit).strip()
        for unit in raw_units
        if unit and unit.strip()
    ]


def score_text_unit(unit, keywords):
    """
    Asigna relevancia a una oración o línea según
    las palabras de la pregunta.
    """

    normalized_unit = normalize_text(unit)

    unit_words = set(
        re.findall(
            r"\b[a-z0-9_-]{3,}\b",
            normalized_unit,
        )
    )

    common_words = keywords.intersection(unit_words)

    score = len(common_words) * 3

    # Las instrucciones y pasos suelen contener la respuesta.
    if re.match(
        r"^(\d+[\.\)]|paso\s+\d+|primero|despues|luego|finalmente)",
        normalized_unit,
    ):
        score += 2

    # Priorizamos líneas que parecen acciones de interfaz.
    action_terms = {
        "seleccione",
        "seleccionar",
        "presione",
        "presionar",
        "ingrese",
        "ingresar",
        "capture",
        "capturar",
        "guardar",
        "confirme",
        "confirmar",
        "inicie",
        "iniciar",
        "detener",
        "reiniciar",
        "ajustar",
        "potencia",
        "usuario",
        "ubicacion",
        "prenda",
        "lector",
        "rfid",
    }

    score += len(action_terms.intersection(unit_words))

    return score


def select_relevant_excerpt(text, question):
    """
    Extrae del fragmento únicamente las líneas más relacionadas
    con la pregunta, conservando su orden original.
    """

    units = split_text_units(text)

    if not units:
        return str(text).strip()[
            :MAX_CHARS_PER_DOCUMENT
        ]

    search_text = expand_search_query(question)
    keywords = extract_keywords(search_text)

    scored_units = [
        (
            index,
            unit,
            score_text_unit(unit, keywords),
        )
        for index, unit in enumerate(units)
    ]

    relevant_units = [
        item
        for item in scored_units
        if item[2] > 0
    ]

    if not relevant_units:
        return " ".join(units)[
            :MAX_CHARS_PER_DOCUMENT
        ]

    # Tomamos las mejores líneas y algunas líneas contiguas
    # para no perder instrucciones relacionadas.
    best_units = sorted(
        relevant_units,
        key=lambda item: item[2],
        reverse=True,
    )[:5]

    selected_indexes = set()

    for index, _unit, _score in best_units:
        selected_indexes.add(index)

        if index > 0:
            selected_indexes.add(index - 1)

        if index + 1 < len(units):
            selected_indexes.add(index + 1)

    ordered_units = [
        units[index]
        for index in sorted(selected_indexes)
    ]

    excerpt = "\n".join(ordered_units)

    return excerpt[:MAX_CHARS_PER_DOCUMENT]


# ============================================================
# Construcción del contexto
# ============================================================

def get_page_number(document):
    """Obtiene el número de página visible para el usuario."""

    page = document.metadata.get("page")

    if isinstance(page, int):
        return page + 1

    return None


def build_context(documents, question):
    """
    Construye un contexto compacto con las partes más relacionadas
    con la pregunta.
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

        excerpt = select_relevant_excerpt(
            text=document.page_content,
            question=question,
        )

        context_parts.append(
            f"[Fuente {index}: {file_name}, "
            f"página {page_text}]\n{excerpt}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# Prompt
# ============================================================

def build_prompt(question, context):
    """
    Construye un prompt compacto para reducir el tiempo
    de procesamiento del modelo.
    """

    return f"""
Eres el asistente de IDLinens HA.

Usa solamente el contexto de los manuales.
Responde en español y no inventes información.

Reglas:
- Reconoce sinónimos y expresiones equivalentes.
- Responde únicamente lo que se preguntó.
- Si es un procedimiento, presenta pasos numerados.
- No confundas crear usuarios, iniciar sesión, registrar prendas,
  poner prendas en circulación, localizar prendas o editar activos.
- No incluyas fuentes; el sistema las agrega.
- Si el contexto no contiene la respuesta, escribe exactamente:
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

    cache_key = normalize_text(question)

    return ANSWER_CACHE.get(cache_key)


def save_cached_answer(question, answer):
    """Guarda una respuesta y limita el tamaño de la caché."""

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

    pregunta → FAISS → contexto relevante → Gemma → fuentes.
    """

    total_start = time.perf_counter()

    clean_question = str(question).strip()

    if not clean_question:
        return "Debes escribir una pregunta."

    cached_answer = get_cached_answer(clean_question)

    if cached_answer is not None:
        if SHOW_PERFORMANCE_TIMES:
            print(
                "\nRespuesta recuperada desde caché: "
                "0 llamadas al modelo."
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

    context = build_context(
        documents=documents,
        question=clean_question,
    )

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
        sources = format_sources(documents)
        final_answer = answer.strip() + sources

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
            f"- Caracteres enviados como contexto: "
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
                    "\nDebes escribir una pregunta.\n"
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