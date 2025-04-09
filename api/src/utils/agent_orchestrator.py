from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama 
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from typing import Literal
from ollama import Client
import os

OLLAMA_URL=os.getenv("OLLAMA_URL")

class Orchestrator:
    def __init__(self, context_window_size: int = 2):
        self.session_memory = {}  # {session_id: ConversationBufferMemory}
        self.classification_chain = self._create_classification_chain()
        self.domain_history = {}  # {session_id: list[domain]}
        self.action_history = {}  # {session_id: {response_id: (domain, action)}}
        self.context_window_size = context_window_size  # Number of recent exchanges to consider
        self.ollama_client = Client(host=OLLAMA_URL)
        
    def _create_classification_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the user's query and conversation history to determine the most appropriate domain.
            Domains:
            - admissions: admissions requirements, deadlines, deadlines, campuses, tuition fees, application fees, application procedures for Concordia University's Computer Science program
            - ai_knowledge: technical topics such as Artificial Intelligence, Machine Learning, Deep Learning, Large Language Models, Natural Language Processing, Computer Vision, Robotics, coding, algorithm and more or any technical terms in those topics.
            - general: Everything else
            
            Current conversation history:
            {history}
            
            Query: {query}
            
            Respond ONLY with one of: admissions, ai_knowledge, general
            Do not include any other text or explanations.
            """
        )
        
        return prompt | ChatOllama(model="llama3.2", base_url=os.getenv("OLLAMA_URL")) | StrOutputParser()

    def _get_session_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.session_memory:
            self.session_memory[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            self.domain_history[session_id] = []
        return self.session_memory[session_id]

    def classify_domain(self, query: str, session_id: str) -> Literal["admissions", "ai_knowledge", "general"]:
        memory = self._get_session_memory(session_id)
        full_history = memory.load_memory_variables({})["chat_history"]
        
        # Apply context window: Take the last 'context_window_size' exchanges
        # Each exchange is a pair (user query + assistant response), so we take 2 * context_window_size messages
        context_window = full_history[-2 * self.context_window_size:] if len(full_history) > 2 * self.context_window_size else full_history
        
        # Format the context window into a readable string
        history_str = "\n".join([f"{('User' if msg.type == 'human' else 'Assistant')}: {msg.content}" for msg in context_window])

        # Get classification with context awareness
        classification = self.classification_chain.invoke({
            "query": query,
            "history": history_str
        }).strip().lower()

        # Context continuation logic
        previous_domains = self.domain_history[session_id]
        if previous_domains:
            last_domain = previous_domains[-1]
            # Maintain context if continuing same topic
            if any(keyword in query.lower() for keyword in ["also", "and what about", "following up"]):
                return last_domain
            
            switch_check = self.ollama_client.generate(
                model="llama3.2",
                prompt=f"Does this query continue discussing {last_domain}? Does it refer to {last_domain} domain? Query: {query} Answer YES or NO. Do not include any other text or explanations."
            )["response"].strip().upper()

            if switch_check == "NO":
                self.domain_history[session_id].append(classification)
                return classification
            return last_domain
        
        self.domain_history[session_id].append(classification)
        return classification

    def route_query(self, query: str, session_id: str) -> str:
        return self.classify_domain(query, session_id)