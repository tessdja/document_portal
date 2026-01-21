import sys
import streamlit as st

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

# at top-level (module global)
_FALLBACK_STORE = {}

def _has_streamlit_context() -> bool:
    try:
        # session_state exists, but ScriptRunContext is missing in bare mode
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

class ConversationalRAG:
    def __init__(self, session_id: str, retriever):
        self.log = CustomLogger().get_logger(__name__)
        self.session_id = session_id
        self.retriever = retriever

        try:
            self._model_loader = ModelLoader()
            self._embeddings = self._model_loader.load_embeddings()
            self.llm = self._model_loader.load_llm()
            self.log.info("LLM loaded successfully", class_name=self.llm.__class__.__name__)

            self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.log.info("Created history-aware retriever", session_id=session_id)

            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
            self.log.info("Created RAG chain", session_id=session_id)

            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            self.log.info("Wrapped chain with message history", session_id=session_id)

        except Exception as e:
            #self.log.error("Error initializing ConversationalRAG", error=str(e), session_id=session_id)
            self.log.exception("Error initializing ConversationalRAG", session_id=session_id)
            raise DocumentPortalException("Failed to initialize ConversationalRAG", sys)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            if _has_streamlit_context():
                if "store" not in st.session_state:
                    st.session_state.store = {}
                store = st.session_state.store
            else:
                store = _FALLBACK_STORE

            #if session_id not in st.session_state.store:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
                self.log.info("New chat session history created", session_id=session_id)

            return store[session_id]
        except Exception as e:
            #self.log.error("Failed to access session history", session_id=session_id, error=str(e))
            self.log.exception("Failed to access session history", session_id=session_id)
            raise DocumentPortalException("Failed to retrieve session history", sys)

    def load_retriever_from_faiss(self,index_path: str, *, k: int = 5, search_type: str = "similarity"):
        try:
            # embeddings should already be cached on self
            vectorstore = FAISS.load_local(index_path, self._embeddings, allow_dangerous_deserialization=True,)  # only if you trust the index
            retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k,"fetch_k": max(20, k * 4),},)
            self.log.info("FAISS retriever loaded", index_path=index_path, search_type=search_type, k=k,)
            return retriever
        
        except Exception:
            self.log.exception("Failed to load FAISS retriever", index_path=index_path,)
            raise DocumentPortalException("Failed to load FAISS retriever", sys,)   
    
    def invoke(self, user_input: str) -> str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )

            docs = response.get("context") or response.get("documents") or []
            if docs:
                # log minimal but useful info; avoid dumping full text
                self.log.info(
                "Retrieved docs",
                session_id=self.session_id,
                count=len(docs),
                top_sources=[(d.metadata or {}).get("source") for d in docs[:3]],
                top_pages=[(d.metadata or {}).get("page") for d in docs[:3]],
                )
            else:
                self.log.warning("No retrieved docs returned in response payload", session_id=self.session_id)

            answer = response.get("answer", "No answer.")

            if not answer:
                self.log.warning("Empty answer received", session_id=self.session_id)

            self.log.info("Chain invoked successfully", session_id=self.session_id, user_input=user_input, answer_preview=answer[:150])
            return answer

        except Exception as e:
            #self.log.error("Failed to invoke conversational RAG", error=str(e), session_id=self.session_id)
            self.log.exception("Failed to invoke conversational RAG", session_id=self.session_id)
            raise DocumentPortalException("Failed to invoke RAG chain", sys)
