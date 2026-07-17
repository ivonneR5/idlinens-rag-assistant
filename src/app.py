import streamlit as st

from rag_agent import (
    answer_question,
    load_embeddings,
    load_language_model,
    load_vector_store,
)


st.set_page_config(
    page_title="IDLinens RAG Assistant",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource
def initialize_agent():
    """
    Carga una sola vez los embeddings, la base FAISS y Gemma.
    Streamlit reutiliza estos recursos durante la sesión.
    """

    embeddings = load_embeddings()
    vector_store = load_vector_store(embeddings)
    model = load_language_model()

    return vector_store, model


def initialize_history():
    """Inicializa el historial de conversación."""

    if "messages" not in st.session_state:
        st.session_state.messages = []


def show_sidebar():
    """Muestra información general y controles de la aplicación."""

    with st.sidebar:
        st.title("ℹ️ Acerca del agente")

        st.write(
            "Este asistente utiliza inteligencia artificial para responder "
            "preguntas sobre la aplicación Android, el Dashboard y la "
            "operación de IDLinens HA."
        )

        st.info(
            "Las respuestas se generan utilizando únicamente la "
            "documentación incluida en la base de conocimiento."
        )

        st.markdown("### Base documental")

        st.markdown(
            """
            - Manual Operativo de IDLinens HA  
            - Manual de Usuario de la aplicación Android  
            - Manual de Usuario del Dashboard
            """
        )

        st.markdown("### Tecnología")

        st.markdown(
            """
            - LangChain  
            - FAISS  
            - Ollama  
            - Gemma 3  
            - Streamlit
            """
        )

        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def show_chat_history():
    """Muestra los mensajes guardados durante la sesión."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    initialize_history()
    show_sidebar()

    st.title("🤖 IDLinens RAG Assistant")

    st.caption(
        "Agente de inteligencia artificial para consultar la documentación "
        "oficial de IDLinens HA."
    )

    st.warning(
        "Este es un agente de inteligencia artificial. "
        "Sus respuestas deben validarse con la documentación oficial "
        "cuando se trate de una operación crítica."
    )

    try:
        with st.spinner("Cargando el agente..."):
            vector_store, model = initialize_agent()

    except Exception as error:
        st.error(
            "No fue posible iniciar el agente. Verifica que Ollama esté "
            "ejecutándose y que la base vectorial haya sido creada."
        )

        with st.expander("Detalle técnico"):
            st.code(str(error))

        st.stop()

    show_chat_history()

    question = st.chat_input(
        "Escribe una pregunta sobre IDLinens HA..."
    )

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los manuales y generando respuesta..."):
            try:
                answer = answer_question(
                    vector_store=vector_store,
                    model=model,
                    question=question,
                )

                st.markdown(answer)

            except Exception as error:
                answer = (
                    "Ocurrió un error al generar la respuesta. "
                    "Verifica que Ollama esté disponible e intenta nuevamente."
                )

                st.error(answer)

                with st.expander("Detalle técnico"):
                    st.code(str(error))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


if __name__ == "__main__":
    main()