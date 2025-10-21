"""
FastAPI Backend for Insurance Policy Copilot
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid
import json
import os
import asyncio
from contextlib import asynccontextmanager

# Azure OpenAI
import openai
from azure.identity import DefaultAzureCredential

# Internal modules
from app.rag_pipeline import get_rag_pipeline
from app.guardrails import PII_Redactor, OutputValidator, PolicyFilter
from app.tools import InsuranceTools
from config.databricks_config import databricks_config

# Logging and monitoring
import logging
from prometheus_client import Counter, Histogram, generate_latest
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
chat_requests = Counter('chat_requests_total', 'Total chat requests')
tool_calls = Counter('tool_calls_total', 'Total tool calls', ['tool_name'])
response_time = Histogram('response_time_seconds', 'Response time')
errors = Counter('errors_total', 'Total errors', ['error_type'])

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-12-01-preview"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Insurance Copilot API...")
    
    # Initialize RAG pipeline
    rag = get_rag_pipeline()
    if not rag.chunks:
        logger.info("No existing index found. Processing documents...")
        from data.policy_document_generator import PolicyDocumentGenerator
        doc_gen = PolicyDocumentGenerator()
        doc_gen.generate_all_documents()
        rag.process_documents()
    
    # Initialize Databricks connection
    try:
        databricks_config.create_catalog_schema()
        logger.info("✓ Databricks connected")
    except Exception as e:
        logger.warning(f"Databricks connection failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Insurance Copilot API...")

# Initialize FastAPI
app = FastAPI(
    title="Insurance Policy Copilot API",
    description="AI-powered insurance policy assistant with RAG and decision tools",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pii_redactor = PII_Redactor()
output_validator = OutputValidator()
policy_filter = PolicyFilter()
insurance_tools = InsuranceTools()

# Request/Response Models
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    customer_id: Optional[str] = None
    policy_id: Optional[str] = None
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    temperature: float = Field(default=0.3, ge=0, le=1)
    
class ToolCall(BaseModel):
    name: str
    parameters: Dict[str, Any]
    
class ChatResponse(BaseModel):
    response: str
    citations: List[str] = []
    tools_used: List[str] = []
    decision: Optional[Dict[str, Any]] = None
    session_id: str
    trace_id: str
    timestamp: datetime
    
class ToolResponse(BaseModel):
    tool: str
    result: Any
    execution_time: float
    
# Audit Log Model
class AuditLog(BaseModel):
    trace_id: str
    session_id: str
    timestamp: datetime
    user_input: str
    user_input_redacted: str
    retrieved_chunks: List[Dict[str, Any]]
    llm_prompt: str
    llm_response: str
    tools_called: List[ToolCall]
    final_response: str
    latency_ms: int
    customer_id: Optional[str]
    policy_id: Optional[str]

# Session storage (use Redis in production)
sessions = {}
audit_logs = []

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """Main chat endpoint with RAG and tool calling"""
    
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    chat_requests.inc()
    
    try:
        # Get or create session
        if request.session_id not in sessions:
            sessions[request.session_id] = {
                "messages": [],
                "context": {},
                "created_at": datetime.now()
            }
        
        session = sessions[request.session_id]
        
        # Get user message
        user_message = request.messages[-1].content if request.messages else ""
        
        # PII Redaction
        user_message_redacted = pii_redactor.redact(user_message)
        
        # Intent Classification
        intent = classify_intent(user_message_redacted)
        
        # Initialize response components
        response_text = ""
        citations = []
        tools_used = []
        decision = None
        retrieved_chunks = []
        
        if intent == "policy_question":
            # RAG-based response
            rag = get_rag_pipeline()
            rag_result = rag.answer_question(user_message_redacted)
            
            if rag_result['confidence'] > 0.3:
                retrieved_chunks = rag_result['chunks_used']
                context = rag_result['context']
                citations = rag_result['citations']
                
                # Generate response using LLM with context
                response_text = await generate_llm_response(
                    user_message_redacted,
                    context,
                    citations,
                    request.temperature
                )
            else:
                response_text = "I couldn't find specific information about that in our policy documents. Please contact customer service for assistance."
                
        elif intent == "tool_required":
            # Determine which tools to call
            tool_calls = determine_tool_calls(user_message_redacted, request)
            
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_params = tool_call["parameters"]
                
                # Execute tool
                tool_result = await execute_tool(tool_name, tool_params)
                tools_used.append(tool_name)
                
                # Special handling for surrender propensity
                if tool_name == "get_surrender_propensity":
                    decision = format_surrender_decision(tool_result)
                    response_text = generate_decision_response(decision)
                else:
                    # Format tool result into response
                    response_text += format_tool_response(tool_name, tool_result)
                    
                tool_calls_metric.labels(tool_name=tool_name).inc()
                
        else:
            # General conversation
            response_text = await generate_llm_response(
                user_message_redacted,
                context="",
                citations=[],
                temperature=request.temperature
            )
        
        # Validate output
        response_text = output_validator.validate(response_text)
        
        # Apply policy filters
        response_text = policy_filter.apply(response_text)
        
        # Update session
        session["messages"].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response_text}
        ])
        
        # Create response
        response = ChatResponse(
            response=response_text,
            citations=citations,
            tools_used=tools_used,
            decision=decision,
            session_id=request.session_id,
            trace_id=trace_id,
            timestamp=datetime.now()
        )
        
        # Audit logging (async)
        audit_log = AuditLog(
            trace_id=trace_id,
            session_id=request.session_id,
            timestamp=datetime.now(),
            user_input=user_message,
            user_input_redacted=user_message_redacted,
            retrieved_chunks=retrieved_chunks,
            llm_prompt="",  # Would be filled with actual prompt
            llm_response=response_text,
            tools_called=[],  # Would be filled with actual tool calls
            final_response=response_text,
            latency_ms=int((time.time() - start_time) * 1000),
            customer_id=request.customer_id,
            policy_id=request.policy_id
        )
        
        background_tasks.add_task(save_audit_log, audit_log)
        
        # Record metrics
        response_time.observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        errors.labels(error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail="An error occurred processing your request")

@app.post("/tools/{tool_name}", response_model=ToolResponse)
async def tool_endpoint(
    tool_name: str,
    parameters: Dict[str, Any]
) -> ToolResponse:
    """Direct tool execution endpoint"""
    
    start_time = time.time()
    
    try:
        result = await execute_tool(tool_name, parameters)
        
        return ToolResponse(
            tool=tool_name,
            result=result,
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# Tool execution functions
async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Any:
    """Execute a tool by name"""
    
    if tool_name == "get_surrender_propensity":
        return await insurance_tools.get_surrender_propensity(
            customer_id=parameters.get("customer_id")
        )
    elif tool_name == "get_policy_summary":
        return await insurance_tools.get_policy_summary(
            policy_id=parameters.get("policy_id")
        )
    elif tool_name == "get_customer_policies":
        return await insurance_tools.get_customer_policies(
            customer_id=parameters.get("customer_id")
        )
    elif tool_name == "calculate_surrender_value":
        return await insurance_tools.calculate_surrender_value(
            policy_id=parameters.get("policy_id")
        )
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

def classify_intent(message: str) -> str:
    """Classify user intent"""
    
    message_lower = message.lower()
    
    # Tool-related keywords
    tool_keywords = [
        "surrender", "propensity", "risk", "score",
        "my policy", "my policies", "customer id",
        "calculate", "value", "charges"
    ]
    
    # Policy question keywords
    policy_keywords = [
        "what is", "how", "when", "where", "why",
        "explain", "tell me about", "coverage",
        "premium", "claim", "exclusion", "terms"
    ]
    
    if any(keyword in message_lower for keyword in tool_keywords):
        return "tool_required"
    elif any(keyword in message_lower for keyword in policy_keywords):
        return "policy_question"
    else:
        return "general"

def determine_tool_calls(message: str, request: ChatRequest) -> List[Dict[str, Any]]:
    """Determine which tools to call based on message"""
    
    tools = []
    message_lower = message.lower()
    
    if "surrender" in message_lower and "propensity" in message_lower:
        if request.customer_id:
            tools.append({
                "name": "get_surrender_propensity",
                "parameters": {"customer_id": request.customer_id}
            })
            
    if "policy" in message_lower and "summary" in message_lower:
        if request.policy_id:
            tools.append({
                "name": "get_policy_summary",
                "parameters": {"policy_id": request.policy_id}
            })
            
    if "surrender value" in message_lower or "calculate surrender" in message_lower:
        if request.policy_id:
            tools.append({
                "name": "calculate_surrender_value",
                "parameters": {"policy_id": request.policy_id}
            })
            
    return tools

async def generate_llm_response(
    message: str,
    context: str,
    citations: List[str],
    temperature: float
) -> str:
    """Generate response using Azure OpenAI"""
    
    system_prompt = """You are an insurance policy copilot. 
    Answer only from the provided context. 
    If information is not in the context, say "I don't have that information in the policy documents."
    Always cite your sources when answering from documents.
    Be helpful, accurate, and concise."""
    
    user_prompt = f"""Context:\n{context}\n\nQuestion: {message}"""
    
    if citations:
        user_prompt += f"\n\nSources: {', '.join(citations)}"
    
    try:
        # For demo, return a formatted response
        # In production, this would call Azure OpenAI
        if context:
            response = f"Based on the policy documents, {message.lower()}. "
            if citations:
                response += f"\n\nSource: {', '.join(citations)}"
        else:
            response = "I can help you with policy-related questions, surrender analysis, and claim information."
            
        return response
        
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return "I encountered an error generating a response. Please try again."

def format_surrender_decision(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format surrender propensity result into decision"""
    
    prob = result.get("probability", 0)
    factors = result.get("top_factors", [])
    
    if prob >= 0.7:
        action = "escalate_to_retention"
        recommendation = "High risk - Immediate retention intervention recommended"
    elif prob >= 0.3:
        action = "send_education"
        recommendation = "Medium risk - Send personalized education materials"
    else:
        action = "normal_service"
        recommendation = "Low risk - Continue normal service"
    
    return {
        "probability": prob,
        "risk_level": "high" if prob >= 0.7 else "medium" if prob >= 0.3 else "low",
        "action": action,
        "recommendation": recommendation,
        "top_factors": factors
    }

def generate_decision_response(decision: Dict[str, Any]) -> str:
    """Generate natural language response from decision"""
    
    prob = decision["probability"]
    risk_level = decision["risk_level"]
    recommendation = decision["recommendation"]
    factors = decision.get("top_factors", [])
    
    response = f"Surrender Risk Analysis:\n"
    response += f"• Probability: {prob:.1%}\n"
    response += f"• Risk Level: {risk_level.capitalize()}\n"
    response += f"• Recommendation: {recommendation}\n"
    
    if factors:
        response += f"• Key Factors: {', '.join(factors)}\n"
    
    return response

def format_tool_response(tool_name: str, result: Any) -> str:
    """Format tool result into natural language"""
    
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    else:
        return str(result)

async def save_audit_log(log: AuditLog):
    """Save audit log (async)"""
    
    # In production, save to database
    audit_logs.append(log.dict())
    
    # Keep only last 1000 logs in memory
    if len(audit_logs) > 1000:
        audit_logs.pop(0)
    
    logger.info(f"Audit log saved: {log.trace_id}")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session history"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear session history"""
    
    if session_id in sessions:
        del sessions[session_id]
    
    return {"message": "Session cleared"}

@app.get("/audit/logs")
async def get_audit_logs(limit: int = 100):
    """Get recent audit logs"""
    
    return audit_logs[-limit:]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)