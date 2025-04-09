from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import re
import uuid
from ollama import Client

from rl.bandit import Bandit

from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from agents.admission_agent import AdmissionsAgent
from agents.ai_agent import AIAgent
from agents.general_agent import GeneralAgent
from utils.agent_orchestrator import Orchestrator
from utils.etl import ETLPipeline
from langchain_core.messages import convert_to_openai_messages

import os

OLLAMA_URL=os.getenv("OLLAMA_URL")

# Define request and response models using Pydantic
class ChatRequest(BaseModel):
    text: str
    session_id: str

class ChatResponse(BaseModel):
    response_id: str
    response: str

class FeedbackRequest(BaseModel):
    session_id: str
    response_id: str
    feedback_score: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    etl = ETLPipeline()
    etl.perform_etl()
    asyncio.create_task(asyncio.to_thread(run_test_evaluation))
    yield


app = FastAPI(
    title="Multi-Agent Chatbot API",
    description="API endpoint for the multi-agent chatbot.",
    version="1.0",
    lifespan=lifespan
)

ollama_client = Client(host=OLLAMA_URL)
orchestrator = Orchestrator()

agents = {
    "admissions": AdmissionsAgent(),
    "ai_knowledge": AIAgent(),
    "general": GeneralAgent()
}


# Initialize bandits tailored to each domain.
# Actions:
# - 0: Only vector DB
# - 1: Only Wikipedia
# - 2: Both, vector DB first
# - 3: Both, Wikipedia first
bandits = {
    "admissions": Bandit(
        num_arms=4,
        initial_values=[1.0, 0.0, 0.5, 0.5]  # Higher value for vector DB (action 0)
    ),
    "ai_knowledge": Bandit(
        num_arms=4,
        initial_values=[0.0, 1.0, 0.5, 0.5]  # Higher value for Wikipedia (action 1)
    ),
    "general": Bandit(
        num_arms=4,
        initial_values=[0.0, 1.0, 0.5, 0.5]  # Higher value for Wikipedia (action 1)
    )
}

# Global user-driven metrics (across all sessions)
user_metrics = {
    "total_good": 0,
    "total_bad": 0,
    "total_unknown": 0,
    "count": 0,
    "accuracy": 0.0,
    "coherence": 0.0,
    "user_satisfaction": 0.0
}

# System-driven (session-based) metrics: {session_id: {accuracy, coherence, user_satisfaction, count}}
system_metrics = {}

# Global test-driven metrics
test_metrics = {
    "accuracy": 0.0,
    "coherence": 0.0,
    "user_satisfaction": 0.0,
    "count": 0
}

@app.get("/")
async def get_status():
    return {"status": "running"}


def evaluate_system_metrics(session_id: str, conversation: str):
    """
    Ask the chatbot (via ollama) to score the conversation for accuracy,
    coherence, and user satisfaction. Update the system_metrics for the session.
    """
    prompt = f"""
                Evaluate this conversation considering different dimensions:
                1. Accuracy: For factual questions, is the answer correct? For social interactions, is the response appropriate?
                2. Coherence: Is the response logically connected to the conversation?
                3. User Satisfaction: How likely would the user be to continue the conversation?

                Now evaluate this conversation:
                {conversation}

                Provide your answer ONLY as a JSON object like:
                {{"accuracy": 0.8, "coherence": 0.7, "user_satisfaction": 0.9}}
                Do not include any other text or explanations.
            """
    result = ollama_client.generate(model="llama3.2", prompt=prompt)
    try:
        json_str = re.search(r'\{.*?\}', result["response"], re.DOTALL).group()
        scores = json.loads(json_str)
    except Exception as e:
        print(f"Error parsing system evaluation: {str(e)}")
        scores = {"accuracy": 0.0, "coherence": 0.0, "user_satisfaction": 0.0}

    # Update per-session system metrics
    if session_id not in system_metrics:
        system_metrics[session_id] = {"accuracy": 0.0, "coherence": 0.0, "user_satisfaction": 0.0, "count": 0}
    system_metrics[session_id]["accuracy"] += scores["accuracy"]
    system_metrics[session_id]["coherence"] += scores["coherence"]
    system_metrics[session_id]["user_satisfaction"] += scores["user_satisfaction"]
    system_metrics[session_id]["count"] += 1


def run_test_evaluation():
    """
    Load test prompts from a custom JSON file (test_prompts.json), simulate conversations,
    and update the test-driven metrics.
    """
    global test_metrics
    try:
        with open("test_prompts.json", "r") as f:
            test_prompts = json.load(f)
    except Exception as e:
        print("Test prompts file not found or error reading file.")
        return

    conversation = ""
    count = 0
    session_id = str(uuid.uuid4())

    for test in test_prompts:
        domain = test.get("domain", "general")
        bandit = bandits[domain]
        action = bandit.select_action()
        agent = agents.get(domain, agents["general"])
        agent.set_session_id(session_id)

        # Simulate a conversation: a single Q&A using the test question.
        response, _ = agent.respond(test["question"], action)

        orchestrator_memory = orchestrator._get_session_memory(session_id)
        orchestrator_memory.save_context({"input": test['question']}, {"output": response})

        conversation += f"User: {test['question']}\nAssistant: {response}\n"
        count += 1
    
    prompt = f"""
                Evaluate the following conversation for accuracy, coherence, and user satisfaction on a 0-1 scale.
                Conversation:
                {conversation}

                Provide your answer as a JSON object like:
                {{"accuracy": 0.8, "coherence": 0.7, "user_satisfaction": 0.9}}
                Do not include any other text or explanations.
            """
    result = ollama_client.generate(model="llama3.2", prompt=prompt)
    try:
        scores = json.loads(result["response"])
    except Exception as e:
        print(f"Error parsing test evaluation: {str(e)}")
        scores = {"accuracy": 0.0, "coherence": 0.0, "user_satisfaction": 0.0}

    test_metrics["accuracy"] = round(scores["accuracy"] * 100, 2)
    test_metrics["coherence"] = round(scores["coherence"] * 100, 2)
    test_metrics["user_satisfaction"] = round(scores["user_satisfaction"] * 100, 2)
    test_metrics["count"] = count


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # Determine domain with context awareness
        domain = orchestrator.route_query(request.text, request.session_id)

        # Get appropriate agent
        agent = agents[domain]
        agent.set_session_id(request.session_id)

        bandit = bandits[domain]
        action = bandit.select_action()

        # Get response with agent's memory
        response, _ = agent.respond(request.text, action)

        response_id = str(uuid.uuid4())
        if request.session_id not in orchestrator.action_history:
            orchestrator.action_history[request.session_id] = {}
        orchestrator.action_history[request.session_id][response_id] = (domain, action)

        orchestrator_memory = orchestrator._get_session_memory(request.session_id)
        orchestrator_memory.save_context({"input": request.text}, {"output": response})

        # Launch system-driven performance evaluation as a background job.
        conversation = convert_to_openai_messages(orchestrator_memory.load_memory_variables({})["chat_history"])

        # Convert conversation history to string for evaluation.
        conversation_text = "\n".join(
            [f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in conversation]
        )
        background_tasks.add_task(evaluate_system_metrics, request.session_id, conversation_text)

        return ChatResponse(
            response_id=response_id,
            response=response
        )
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}", response_id="")
    

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Update user-driven metrics based on the user's rating.
    Rating meanings:
      1  -> Good (accurate and context aware)
      -1 -> Bad
      0  -> Neutral (partial credit: 0.5 for both accuracy and coherence)
    """
    global user_metrics
    session_id = feedback.session_id
    response_id = feedback.response_id
    rating = feedback.feedback_score

    if (session_id in orchestrator.action_history and 
        response_id in orchestrator.action_history[session_id]):
        domain, action = orchestrator.action_history[session_id][response_id]
        bandits[domain].update(action, feedback.feedback_score)
    else:
        raise HTTPException(status_code=404, detail="Response not found")
    
    user_metrics["count"] += 1
    if rating == 1:
        user_metrics["total_good"] += 1
        user_metrics["accuracy"] += 1
        user_metrics["coherence"] += 1
        user_metrics["user_satisfaction"] += 1
    elif rating == -1:
        user_metrics["total_bad"] += 1
    elif rating == 0:
        user_metrics["total_unknown"] += 1
        user_metrics["accuracy"] += 0.5
        user_metrics["coherence"] += 0.5
        user_metrics["user_satisfaction"] += 0.5
        
    
    return {"status": "Feedback recorded"}


@app.get("/api/metrics")
async def get_metrics():
    """Return all metrics (user, system, and test). For system metrics, compute average scores across sessions."""
    overall_system = {"accuracy": 0.0, "coherence": 0.0, "user_satisfaction": 0.0, "count": 0}
    session_count = len(system_metrics)
    for metrics in system_metrics.values():
        if metrics["count"] > 0:
            overall_system["accuracy"] += metrics["accuracy"] / metrics["count"]
            overall_system["coherence"] += metrics["coherence"] / metrics["count"]
            overall_system["user_satisfaction"] += metrics["user_satisfaction"] / metrics["count"]
            overall_system["count"] += metrics["count"]
    if session_count > 0:
        overall_system["accuracy"] = round((overall_system["accuracy"] / session_count) * 100, 2)
        overall_system["coherence"] = round((overall_system["coherence"] / session_count) * 100, 2)
        overall_system["user_satisfaction"] = round((overall_system["user_satisfaction"] / session_count) * 100, 2)
    
    overall_user_metric = {"accuracy": 0.0, "coherence": 0.0, "user_satisfaction": 0.0, "count": 0}
    if user_metrics["count"] > 0:
        overall_user_metric["accuracy"] = round((user_metrics["accuracy"] / user_metrics["count"]) * 100, 2)
        overall_user_metric["coherence"] = round((user_metrics["coherence"] / user_metrics["count"]) * 100, 2)
        overall_user_metric["user_satisfaction"] = round((user_metrics["user_satisfaction"] / user_metrics["count"]) * 100, 2)
        overall_user_metric["count"] = user_metrics["count"]
    
    return {
        "user_metrics": overall_user_metric,
        "system_metrics": overall_system,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
