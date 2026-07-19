from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "docs"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180


def clean_page_text(text):
    """Limpia el texto sin destruir listas, títulos ni pasos."""

    clean_text = re.sub(r"\r\n?", "\n", str(text))
    clean_text = re.sub(r"[ \t]+", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text.strip()


def load_documents():
    """Carga los PDF y conserva metadatos útiles."""

    pdf_files = sorted(DOCS_PATH.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron archivos PDF en la carpeta: {DOCS_PATH}"
        )

    documents = []

    for pdf_file in pdf_files:
        print(f"Cargando: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        valid_pages = 0

        for page in pages:
            content = clean_page_text(page.page_content)
            if len(content) < 40:
                continue

            page.page_content = content
            page.metadata["file_name"] = pdf_file.name
            documents.append(page)
            valid_pages += 1

        print(f"  Páginas válidas cargadas: {valid_pages}")

    return documents


def split_documents(documents):
    """
    Divide con tamaño intermedio: suficiente para conservar pasos,
    pero sin mezclar demasiadas secciones u operaciones.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.page_content = clean_page_text(chunk.page_content)

    return chunks


def show_sample(chunks):
    if not chunks:
        print("No se generaron fragmentos.")
        return

    sample = chunks[0]
    page_number = sample.metadata.get("page")
    if isinstance(page_number, int):
        page_number += 1

    print("\n" + "=" * 70)
    print("FRAGMENTO DE EJEMPLO")
    print("=" * 70)
    print(f"Archivo: {sample.metadata.get('file_name', 'Sin nombre')}")
    print(f"Página: {page_number if page_number is not None else 'No disponible'}")
    print(f"Caracteres: {len(sample.page_content)}")
    print("-" * 70)
    print(sample.page_content[:700])
    print("=" * 70)


def main():
    print("\nIniciando extracción y fragmentación...\n")

    documents = load_documents()
    chunks = split_documents(documents)

    print("\nProceso completado correctamente.")
    print("\nESTADÍSTICAS")
    print("-" * 70)
    print(f"Páginas cargadas: {len(documents)}")
    print(f"Fragmentos generados: {len(chunks)}")
    print(f"Tamaño de fragmento: {CHUNK_SIZE}")
    print(f"Superposición: {CHUNK_OVERLAP}")

    show_sample(chunks)


if __name__ == "__main__":
    main()