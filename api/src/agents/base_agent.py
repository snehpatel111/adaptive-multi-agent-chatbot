import os
from typing import List, Tuple
import chromadb
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from wikipediaapi import Wikipedia
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

class BaseAgent:
    def __init__(self, collection_name: str):
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_URL"),
            model="llama3.2",
            temperature=0
        )
        self.client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH"))
        self.wikipedia = Wikipedia(
            user_agent=os.getenv('USER_AGENT'), language='en')
        self.memories = {}  # {session_id: ConversationBufferMemory}
        self.collection_name = collection_name
        self.vector_store = Chroma(
            collection_name=collection_name,
            client=self.client,
            persist_directory=os.getenv("CHROMA_DB_PATH"),
            embedding_function=OllamaEmbeddings(
                model="llama3.2",
                base_url=os.getenv("OLLAMA_URL"),
            )
        )

        # Create contextualization chain
        self.history_aware_retriever = self._create_contextual_retriever()
        self.qa_chain = self._create_qa_chain()
        self.qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know.")

    def _create_contextual_retriever(self):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        return create_history_aware_retriever(
            self.llm,
            self.vector_store.as_retriever(search_kwargs={"k": 5}),
            contextualize_q_prompt
        )

    def _create_qa_chain(self):
        qa_system_prompt = """You are a knowledgeable assistant for question-answering tasks. Make sure to answer according to the current context of chat. Do not answer out of the context. Combine information from:
        - Contextual knowledge base (priority)
        - Wikipedia supplements
        - Conversation history
        
        {context}
        
        Guidelines:
        1. Acknowledge if sources conflict
        2. Maintain conversation context"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )

        return create_retrieval_chain(self.history_aware_retriever, document_chain)

    def respond(self, query: str, action: int) -> tuple[str, dict]:
        try:

            # Get Wikipedia results
            wiki_content = self._get_wikipedia_result(query)

            # Get Vector DB results
            vector_content, vector_ids = self._get_vector_db_result(query)

            # Construct context based on action
            if action == 0:  # Only vector DB
                context = [vector_content]
            elif action == 1:  # Only Wikipedia
                context = [Document(page_content=wiki_content, metadata={"source": "Wikipedia"})] if wiki_content else []
            elif action == 2:  # Both, vector DB first
                context = [vector_content] + ([Document(page_content=wiki_content, metadata={"source": "Wikipedia"})] if wiki_content else [])
            elif action == 3:  # Both, Wikipedia first
                context = ([Document(page_content=wiki_content, metadata={"source": "Wikipedia"})] if wiki_content else []) + [vector_content]

            # Fallback: if context is empty and wiki_content is available, use wiki_content
            if not context and wiki_content:
                context = [Document(page_content=wiki_content, metadata={"source": "Wikipedia"})]

            # Convert the context into a list of strings. For each item, if it's a Document, use its page_content.
            context_str_list = [item if isinstance(item, str) else item.page_content for item in context]
            context_str = "\n\n".join(context_str_list)

            memory = self._get_memory()

            # Invoke the full chain
            result = self.qa_chain.invoke({
                "input": query,
                "chat_history": memory.load_memory_variables({})["chat_history"],
                "context": context_str,
                "wikipedia": wiki_content or "No Wikipedia content"
            })

            memory.save_context({"input": query}, {"output": result["answer"]})

            return result["answer"], {
                "sources": self._get_sources(result.get("context", []), bool(wiki_content)),
                "vector_ids": vector_ids,
                "wikipedia_used": bool(wiki_content),
            }
        except Exception as e:
            raise Exception(f"Error in respond: {str(e)}")

    def _get_wikipedia_result(self, query: str) -> str:
        try:
            wiki_page = self.wikipedia.page(query)
            return wiki_page.summary[:2000] if wiki_page.exists() else ""
        except Exception as e:
            return ""

    def _get_vector_db_result(self, query: str) -> Tuple[str, List[str]]:
        try:
            docs = self.vector_store.similarity_search(query, k=10)
            return "\n".join(d.page_content for d in docs), [d.metadata["id"] for d in docs]
        except:
            return "", []
    
    def _get_sources(self, context_docs: list, wiki_used: bool) -> list:
        sources = set()
        for doc in context_docs:
            sources.add(doc.metadata.get("source", "KB"))
        if wiki_used:
            sources.add("Wikipedia")
        return list(sources)
    
    def _get_memory(self) -> ConversationBufferMemory:
        """Get or create memory buffer for session"""
        if not hasattr(self, 'session_id'):
            raise ValueError("Session ID must be set using set_session_id() first")
    
        if self.session_id not in self.memories:
            self.memories[self.session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.memories[self.session_id]
    
    def set_session_id(self, session_id: str):
        self.session_id = session_id