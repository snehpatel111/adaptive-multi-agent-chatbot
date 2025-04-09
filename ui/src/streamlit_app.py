import time
import uuid
import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()

CHATBOT_URL = os.getenv("CHATBOT_URL")
FEEDBACK_URL = os.getenv("FEEDBACK_URL")
METRICS_URL = os.getenv("METRICS_URL")

st.set_page_config(page_title="Multi-Agent Chatbot",
                   page_icon="🤖", layout="wide")


# Initialize session IDs
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0
if "last_updated_at" not in st.session_state:
    st.session_state.last_updated_at = 0

# Initialize session state
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'user_metrics': {'accuracy': 0.0, 'coherence': 0.0, 'user_satisfaction': 0.0, 'count': 0},
        'system_metrics': {'accuracy': 0.0, 'coherence': 0.0, 'user_satisfaction': 0.0, 'count': 0},
        'test_metrics': {'accuracy': 0.0, 'coherence': 0.0, 'user_satisfaction': 0.0, 'count': 0},
    }


# Add model description section
def model_description():
    with st.expander("ℹ️ **Model Architecture Details**"):
        st.markdown("""
        Our chatbot uses a sophisticated multi-agent system with context-aware routing:
        
        - 🧠 **3 Specialized Agents**
            | Agent          | Function                          | Knowledge Source          |
            |----------------|-----------------------------------|---------------------------|
            | Admissions     | Concordia CS admissions info     | University web data       |
            | AI Knowledge   | Technical AI/ML concepts         | Wikipedia + Research data |
            | General        | Broad domain queries             | Wikipedia                 |
          
        - 🔀 **Smart Routing System**
          - Context-aware query classification
          - Conversation history tracking
          - Domain-specific knowledge retrieval
          
        - 📈 **Continuous Learning**
          - Real-time user feedback integration
          - Reinforcement learning from interactions
          - Automatic performance benchmarking
          
        **Technical Stack**
        ```python
        - Ollama + Llama3.2 (Base LLM)
        - ChromaDB (Vector Storage)
        - FastAPI (Backend)
        - LangChain (Orchestration)
        - Streamlit (Frontend)
        ```
        """)


def main_content():
    # Main UI
    st.title("Adaptive Multi-Agent Chatbot")
    model_description()
    st.info("Ask about admissions, AI, or general topics. Your conversation context is maintained and evaluated.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history with feedback
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                feedback_key = f"feedback_{i}_{st.session_state.feedback_key}"
                with st.form(key=feedback_key):
                    st.write("Rate this response:")
                    rating = st.radio(
                        "Rating",
                        ["👍", "👎", "🤷"],
                        key=f"rating_{i}",
                        horizontal=True
                    )
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        score = {"👍": 1, "👎": -1, "🤷": 0}[rating]
                        feedback_data = {
                            "session_id": st.session_state.session_id,
                            "response_text": message["content"],
                            "response_id": message["response_id"],
                            "feedback_score": score
                        }
                        try:
                            requests.post(FEEDBACK_URL, json=feedback_data)
                            st.success("Feedback recorded!")
                            st.session_state.feedback_key += 1
                        except Exception as e:
                            st.error(f"Error submitting feedback: {str(e)}")

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # API request with error handling
        with st.spinner("Processing your request..."):
            try:
                response = requests.post(
                    CHATBOT_URL,
                    json={
                        "text": prompt,
                        "session_id": st.session_state.session_id
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    response_content = result["response"]
                    response_id = result["response_id"]
                else:
                    response_content = "Error processing request"
                    response_id = None
            except Exception as e:
                response_content = f"Connection error: {str(e)}"

        # Display response
        with st.chat_message("assistant"):
            st.markdown(response_content)

        # Update chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_content,
            "response_id": response_id
        })
        st.rerun()


def sidebar_content():
    # get_latest_metrics()
    with st.sidebar:
        st.header("Performance Metrics")
        st.info("Three-layered evaluation system tracking different aspects of chatbot performance:")
        
        with st.expander("📊 Metric Definitions"):
            st.markdown("""
            **User Metrics**
            - Collected from explicit user feedback (👍/👎/🤷 ratings)
            - Measures: Immediate user satisfaction
            
            **System Metrics**
            - Automated conversation quality assessment
            - Evaluates: Accuracy, Coherence, Context maintenance
            
            **Test Metrics**
            - Benchmark results from predefined test cases
            - Simulates: Real-world usage scenarios
            """)

        # st.subheader("User Metrics")
        # col1, col2 = st.columns(2)
        # user_metrics_total_responses_placeholder = col1.empty()
        # user_metrics_acc_placeholder = col1.empty()
        # user_metrics_coh_placeholder = col2.empty()
        # user_metrics_user_satisfaction_placeholder = col2.empty()

        # st.subheader("System Metrics")
        # col1, col2 = st.columns(2)
        # system_metric_total_responses_placeholder = col1.empty()
        # system_metric_acc_placeholder = col1.empty()
        # system_metric_coh_placeholder = col2.empty()
        # system_metric_user_satisfaction_placeholder = col2.empty()

        # st.subheader("Test Metrics")
        # col1, col2 = st.columns(2)
        # test_metric_total_responses_placeholder = col1.empty()
        # test_metric_acc_placeholder = col1.empty()
        # test_metric_coh_placeholder = col2.empty()
        # test_metric_user_satisfaction_placeholder = col2.empty()

        # user_metrics_total_responses_placeholder.metric("Total Responses", f"{st.session_state.metrics['user_metrics'].get('count', 0)}")
        # user_metrics_acc_placeholder.metric("Accuracy", f"{st.session_state.metrics['user_metrics'].get('accuracy', 0)}%")
        # user_metrics_coh_placeholder.metric("Coherence", f"{st.session_state.metrics['user_metrics'].get('coherence', 0)}%")
        # user_metrics_user_satisfaction_placeholder.metric("User Satisfaction", f"{st.session_state.metrics['user_metrics'].get('user_satisfaction', 0)}%")

        # system_metric_total_responses_placeholder.metric("Total Responses", f"{st.session_state.metrics['system_metrics'].get('count', 0)}")
        # system_metric_acc_placeholder.metric("Accuracy", f"{st.session_state.metrics['system_metrics'].get('accuracy', 0)}%")
        # system_metric_coh_placeholder.metric("Coherence", f"{st.session_state.metrics['system_metrics'].get('coherence', 0)}%")
        # system_metric_user_satisfaction_placeholder.metric("User Satisfaction", f"{st.session_state.metrics['system_metrics'].get('user_satisfaction', 0)}%")

        # test_metric_total_responses_placeholder.metric("Total Responses", f"{st.session_state.metrics['test_metrics'].get('count', 0)}")
        # test_metric_acc_placeholder.metric("Accuracy", f"{st.session_state.metrics['test_metrics'].get('accuracy', 0)}%")
        # test_metric_coh_placeholder.metric("Coherence", f"{st.session_state.metrics['test_metrics'].get('coherence', 0)}%")
        # test_metric_user_satisfaction_placeholder.metric("User Satisfaction", f"{st.session_state.metrics['test_metrics'].get('user_satisfaction', 0)}%")

        st.subheader("Real-Time Metrics")
        col1, col2 = st.columns(2)
        
        # User Metrics
        with st.container(border=True):
            st.markdown("### 👤 User Feedback")
            col1, col2 = st.columns(2)
            col1.metric("Total Responses", st.session_state.metrics['user_metrics']['count'])
            col1.metric("Accuracy", f"{st.session_state.metrics['user_metrics']['accuracy']}%")
            col2.metric("Coherence", f"{st.session_state.metrics['user_metrics']['coherence']}%")
            col2.metric("Satisfaction", f"{st.session_state.metrics['user_metrics']['user_satisfaction']}%")

        # System Metrics
        with st.container(border=True):
            st.markdown("### 🤖 System Evaluation")
            col1, col2 = st.columns(2)
            col1.metric("Conversations", st.session_state.metrics['system_metrics']['count'])
            col1.metric("Accuracy", f"{st.session_state.metrics['system_metrics']['accuracy']}%")
            col2.metric("Coherence", f"{st.session_state.metrics['system_metrics']['coherence']}%")
            col2.metric("Satisfaction", f"{st.session_state.metrics['system_metrics']['user_satisfaction']}%")

        # Test Metrics
        with st.container(border=True):
            st.markdown("### 🧪 Benchmark Tests")
            col1, col2 = st.columns(2)
            col1.metric("Test Cases", st.session_state.metrics['test_metrics']['count'])
            col1.metric("Accuracy", f"{st.session_state.metrics['test_metrics']['accuracy']}%")
            col2.metric("Coherence", f"{st.session_state.metrics['test_metrics']['coherence']}%")
            col2.metric("Satisfaction", f"{st.session_state.metrics['test_metrics']['user_satisfaction']}%")

        st.markdown("---")
        st.markdown("**Legend**")
        st.caption("📈 Accuracy: Factual correctness of responses")
        st.caption("🔗 Coherence: Logical flow consistency")
        st.caption("😊 Satisfaction: User-reported experience quality")
        

@st.fragment(run_every="3s")
def get_latest_metrics():
    current_time = time.time()
    last_updated = st.session_state.get("last_updated_at", 0)
    if current_time - last_updated > 3:
        response = requests.get(METRICS_URL)
        if response.status_code == 200:
            data = response.json()
            st.session_state.metrics = data
            st.session_state.last_updated_at = current_time
            st.rerun()
        else:
            st.error(f"Failed to fetch metrics. Status code: {response.status_code}")


if __name__ == "__main__":
    main_content()
    sidebar_content()
    get_latest_metrics()