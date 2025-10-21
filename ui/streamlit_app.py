"""
Streamlit UI for Insurance Policy Copilot
"""

import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import time

# Page configuration
st.set_page_config(
    page_title="Insurance Policy Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8001"

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fc;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .main-subtitle {
        color: #e0e7ff;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e0e7ff;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: white;
        margin-right: 20%;
        border: 1px solid #e5e7eb;
    }
    .citation {
        background-color: #fef3c7;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .decision-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .risk-high {
        border-left: 4px solid #ef4444;
    }
    .risk-medium {
        border-left: 4px solid #f59e0b;
    }
    .risk-low {
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "policy_id" not in st.session_state:
    st.session_state.policy_id = None

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API call to backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def display_decision_card(decision: Dict[str, Any]):
    """Display decision card with risk assessment"""
    risk_level = decision.get("risk_level", "unknown")
    risk_class = f"risk-{risk_level}"
    
    st.markdown(f"""
    <div class="decision-card {risk_class}">
        <h4>🎯 Surrender Risk Assessment</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
            <div>
                <strong>Risk Level:</strong> {risk_level.upper()}<br/>
                <strong>Probability:</strong> {decision.get('probability', 0):.1%}<br/>
                <strong>Action:</strong> {decision.get('action', 'N/A')}
            </div>
            <div>
                <strong>Recommendation:</strong><br/>
                {decision.get('recommendation', 'No recommendation available')}
            </div>
        </div>
        <div style="margin-top: 1rem;">
            <strong>Top Risk Factors:</strong>
            <ul style="margin-top: 0.5rem;">
                {"".join([f"<li>{factor}</li>" for factor in decision.get('top_factors', [])])}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_citations(citations: List[str]):
    """Display document citations"""
    if citations:
        st.markdown("""
        <div class="citation">
            <strong>📚 Sources:</strong> {}
        </div>
        """.format(", ".join(citations)), unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🛡️ Insurance Policy Copilot</h1>
        <p class="main-subtitle">AI-powered assistant for policy inquiries and surrender risk analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Customer/Policy context
        st.subheader("Context")
        customer_id = st.text_input(
            "Customer ID", 
            value=st.session_state.customer_id or "",
            placeholder="e.g., C000001",
            help="Enter customer ID for personalized assistance"
        )
        if customer_id:
            st.session_state.customer_id = customer_id
            
        policy_id = st.text_input(
            "Policy ID",
            value=st.session_state.policy_id or "",
            placeholder="e.g., P00000001",
            help="Enter policy ID for specific policy queries"
        )
        if policy_id:
            st.session_state.policy_id = policy_id
            
        # Quick Actions
        st.subheader("Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Check Risk", use_container_width=True):
                if st.session_state.customer_id:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Check surrender propensity for customer {st.session_state.customer_id}"
                    })
                else:
                    st.warning("Please enter a Customer ID")
                    
        with col2:
            if st.button("📋 Policy Info", use_container_width=True):
                if st.session_state.policy_id:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Get summary for policy {st.session_state.policy_id}"
                    })
                else:
                    st.warning("Please enter a Policy ID")
                    
        if st.button("💰 Calculate Surrender", use_container_width=True):
            if st.session_state.policy_id:
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Calculate surrender value for policy {st.session_state.policy_id}"
                })
            else:
                st.warning("Please enter a Policy ID")
                
        # Session Management
        st.subheader("Session")
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.session_id:
                call_api(f"/sessions/{st.session_state.session_id}", method="DELETE")
            st.session_state.session_id = None
            st.rerun()
            
        # Statistics
        st.subheader("📊 Statistics")
        if st.session_state.messages:
            st.metric("Messages", len(st.session_state.messages))
            st.metric("Session Duration", f"{len(st.session_state.messages) * 2} min")
            
        # Sample Questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is the surrender charge in year 2?",
            "What is the lock-in period for ULIP?",
            "How do I file a claim?",
            "What are the exclusions in health insurance?",
            "What is the grace period for premium payment?"
        ]
        
        for question in sample_questions:
            if st.button(f"❓ {question[:30]}...", key=question, use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br/>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br/>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display citations if available
                    if "citations" in message:
                        display_citations(message["citations"])
                        
                    # Display decision card if available
                    if "decision" in message and message["decision"]:
                        display_decision_card(message["decision"])
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your message",
                placeholder="Ask about policies, claims, surrender values, or risk assessment...",
                height=100
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                submit = st.form_submit_button("Send 📤", use_container_width=True)
            with col2:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
                
        if submit and user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Prepare API request
            request_data = {
                "messages": [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ],
                "temperature": temperature
            }
            
            if st.session_state.customer_id:
                request_data["customer_id"] = st.session_state.customer_id
            if st.session_state.policy_id:
                request_data["policy_id"] = st.session_state.policy_id
            if st.session_state.session_id:
                request_data["session_id"] = st.session_state.session_id
                
            # Show spinner
            with st.spinner("Thinking..."):
                # Call API
                response = call_api("/chat", method="POST", data=request_data)
                
                if response:
                    # Update session ID
                    st.session_state.session_id = response.get("session_id")
                    
                    # Add assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": response.get("response", ""),
                        "citations": response.get("citations", []),
                        "decision": response.get("decision"),
                        "tools_used": response.get("tools_used", [])
                    }
                    st.session_state.messages.append(assistant_message)
                    
            st.rerun()
    
    with col2:
        st.subheader("📈 Insights")
        
        # Risk Distribution (Mock Data)
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Count': [65, 25, 10],
            'Color': ['#10b981', '#f59e0b', '#ef4444']
        })
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_data['Risk Level'],
                values=risk_data['Count'],
                marker_colors=risk_data['Color'],
                hole=0.4
            )
        ])
        
        fig.update_layout(
            title="Portfolio Risk Distribution",
            showlegend=True,
            height=250,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Metrics
        st.markdown("### 📊 Key Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>78%</h4>
                <p>Retention Rate</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>2.5%</h4>
                <p>Surrender Rate</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Recent Activity
        st.markdown("### 🕐 Recent Activity")
        activities = [
            "Risk assessment completed",
            "Policy summary generated",
            "Surrender value calculated",
            "Documents retrieved"
        ]
        
        for activity in activities[:3]:
            st.info(f"✓ {activity}", icon="ℹ️")

if __name__ == "__main__":
    main()