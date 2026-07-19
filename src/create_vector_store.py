from pathlib import Path
import shutil

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from chunk_documents import load_documents, split_documents


# ============================================================
# Configuración
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store"
TEMP_VECTOR_STORE_PATH = BASE_DIR / "vector_store_temp"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# Embeddings
# ============================================================

def create_embeddings():
    """
    Inicializa el mismo modelo de embeddings utilizado
    posteriormente por el agente RAG.
    """

    print("\nCargando modelo de embeddings...")
    print(f"Modelo: {EMBEDDING_MODEL}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ============================================================
# Construcción del índice
# ============================================================

def build_vector_store(chunks, embeddings):
    """
    Genera el índice FAISS utilizando exactamente los fragmentos
    producidos por chunk_documents.py.
    """

    if not chunks:
        raise ValueError(
            "No existen fragmentos para crear la base vectorial."
        )

    print("\nGenerando embeddings...")
    print(f"Fragmentos que serán indexados: {len(chunks)}")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


# ============================================================
# Guardado seguro
# ============================================================

def save_vector_store(vector_store):
    """
    Guarda primero el índice en una carpeta temporal.

    Solo después de guardarlo correctamente reemplaza el índice
    anterior, evitando perder la base funcional si ocurre un error.
    """

    if TEMP_VECTOR_STORE_PATH.exists():
        shutil.rmtree(TEMP_VECTOR_STORE_PATH)

    TEMP_VECTOR_STORE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nGuardando índice temporal...")

    vector_store.save_local(
        str(TEMP_VECTOR_STORE_PATH)
    )

    required_files = [
        TEMP_VECTOR_STORE_PATH / "index.faiss",
        TEMP_VECTOR_STORE_PATH / "index.pkl",
    ]

    missing_files = [
        path.name
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise RuntimeError(
            "El índice no se guardó correctamente. "
            f"Archivos faltantes: {missing_files}"
        )

    if VECTOR_STORE_PATH.exists():
        shutil.rmtree(VECTOR_STORE_PATH)

    TEMP_VECTOR_STORE_PATH.rename(
        VECTOR_STORE_PATH
    )

    print("\nÍndice vectorial guardado correctamente.")
    print(f"Ubicación: {VECTOR_STORE_PATH}")


# ============================================================
# Validación
# ============================================================

def validate_chunks(chunks):
    """
    Comprueba que los fragmentos tengan contenido y metadatos
    básicos antes de construir el índice.
    """

    valid_chunks = []

    empty_chunks = 0
    missing_file_name = 0
    missing_page = 0

    for chunk in chunks:
        content = str(
            chunk.page_content or ""
        ).strip()

        if not content:
            empty_chunks += 1
            continue

        if not chunk.metadata.get("file_name"):
            missing_file_name += 1

        if chunk.metadata.get("page") is None:
            missing_page += 1

        valid_chunks.append(chunk)

    print("\nValidación de fragmentos")
    print("-" * 60)
    print(f"Fragmentos recibidos: {len(chunks)}")
    print(f"Fragmentos válidos: {len(valid_chunks)}")
    print(f"Fragmentos vacíos descartados: {empty_chunks}")
    print(
        "Fragmentos sin nombre de archivo: "
        f"{missing_file_name}"
    )
    print(
        "Fragmentos sin número de página: "
        f"{missing_page}"
    )

    if not valid_chunks:
        raise ValueError(
            "No quedaron fragmentos válidos para indexar."
        )

    return valid_chunks


# ============================================================
# Ejecución principal
# ============================================================

def main():
    try:
        print("\n" + "=" * 70)
        print("CREACIÓN DE LA BASE VECTORIAL")
        print("=" * 70)

        print(
            "\nCargando los manuales y utilizando "
            "la fragmentación oficial..."
        )

        documents = load_documents()
        chunks = split_documents(documents)

        chunks = validate_chunks(chunks)

        embeddings = create_embeddings()

        vector_store = build_vector_store(
            chunks=chunks,
            embeddings=embeddings,
        )

        save_vector_store(vector_store)

        print("\n" + "=" * 70)
        print("PROCESO COMPLETADO")
        print("=" * 70)
        print(f"Páginas procesadas: {len(documents)}")
        print(f"Fragmentos indexados: {len(chunks)}")
        print(f"Modelo utilizado: {EMBEDDING_MODEL}")
        print("=" * 70)

    except FileNotFoundError as error:
        print(f"\nError de archivos: {error}")
        raise

    except ImportError as error:
        print(
            "\nNo fue posible importar las funciones "
            "de chunk_documents.py."
        )
        print(f"Detalle: {error}")
        raise

    except Exception as error:
        print(
            "\nOcurrió un error durante la creación "
            "de la base vectorial."
        )
        print(f"Detalle: {error}")
        raise

    finally:
        if TEMP_VECTOR_STORE_PATH.exists():
            shutil.rmtree(
                TEMP_VECTOR_STORE_PATH,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()