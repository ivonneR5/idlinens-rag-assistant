from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "docs"


def load_documents():
    """Carga todos los PDF y devuelve sus páginas como documentos."""

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

        # Añadimos un nombre de archivo fácil de consultar.
        for page in pages:
            page.metadata["file_name"] = pdf_file.name

        documents.extend(pages)

        print(f"  Páginas cargadas: {len(pages)}")

    return documents


def split_documents(documents):
    """Divide las páginas en fragmentos conservando sus metadatos."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return text_splitter.split_documents(documents)


def show_sample(chunks):
    """Muestra un fragmento de ejemplo y sus metadatos."""

    if not chunks:
        print("No se generaron fragmentos.")
        return

    sample = chunks[0]

    print("\n" + "=" * 70)
    print("FRAGMENTO DE EJEMPLO")
    print("=" * 70)
    print(f"Archivo: {sample.metadata.get('file_name', 'Sin nombre')}")

    # PyPDFLoader normalmente numera las páginas desde cero.
    page_number = sample.metadata.get("page")
    if isinstance(page_number, int):
        page_number += 1

    print(f"Página: {page_number if page_number is not None else 'No disponible'}")
    print(f"Caracteres: {len(sample.page_content)}")
    print("-" * 70)
    print(sample.page_content[:700])
    print("=" * 70)


def main():
    print("\nIniciando extracción y fragmentación...\n")

    documents = load_documents()
    chunks = split_documents(documents)

    print("\nProceso completado.")
    print(f"Páginas totales cargadas: {len(documents)}")
    print(f"Fragmentos generados: {len(chunks)}")

    show_sample(chunks)


if __name__ == "__main__":
    main()