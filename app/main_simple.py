"""
Simplified FastAPI Backend for Insurance Policy Copilot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid
import json
import os
import random

# Initialize FastAPI
app = FastAPI(
    title="Insurance Policy Copilot API",
    description="AI-powered insurance policy assistant with RAG and decision tools",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
class ChatResponse(BaseModel):
    response: str
    citations: List[str] = []
    tools_used: List[str] = []
    decision: Optional[Dict[str, Any]] = None
    session_id: str
    trace_id: str
    timestamp: datetime

# Session storage
sessions = {}

# Mock policy data
MOCK_POLICIES = {
    "surrender_charges": {
        "year_1": "10% of total premiums paid",
        "year_2": "8% of total premiums paid",
        "year_3": "6% of total premiums paid",
        "year_4": "4% of total premiums paid",
        "year_5": "No surrender charge"
    },
    "ulip_lockin": "5 years from policy commencement",
    "health_exclusions": [
        "Cosmetic/aesthetic treatments",
        "Dental treatment (unless accident-related)",
        "Self-inflicted injuries",
        "Treatment arising from substance abuse"
    ],
    "claim_process": [
        "Notify within 30 days of claim event",
        "Submit all required documents",
        "Settlement within 30 days of document receipt"
    ],
    "grace_period": "30 days from premium due date"
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Insurance Policy Copilot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "insurance-copilot-api"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint with mock responses"""
    
    trace_id = str(uuid.uuid4())
    
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
    user_message_lower = user_message.lower()
    
    # Initialize response components
    response_text = ""
    citations = []
    tools_used = []
    decision = None
    
    # Simple keyword-based responses
    if "surrender" in user_message_lower and "charge" in user_message_lower:
        if "year 2" in user_message_lower or "second year" in user_message_lower:
            response_text = "The surrender charge in year 2 is 8% of total premiums paid. This charge decreases over time and becomes zero after year 5."
            citations = ["Term Life Insurance Policy (Page 3)"]
        else:
            response_text = "Surrender charges vary by year:\n"
            for year, charge in MOCK_POLICIES["surrender_charges"].items():
                response_text += f"• {year.replace('_', ' ').title()}: {charge}\n"
            citations = ["Term Life Insurance Policy (Page 3)", "General Terms & Conditions (Page 5)"]
            
    elif "lock-in" in user_message_lower or "lockin" in user_message_lower:
        response_text = f"The lock-in period for ULIP policies is {MOCK_POLICIES['ulip_lockin']}. During this period, partial withdrawals are not allowed, though you can switch between funds."
        citations = ["ULIP Insurance Policy (Page 4)"]
        
    elif "health" in user_message_lower and "exclusion" in user_message_lower:
        response_text = "Health insurance exclusions include:\n"
        for exclusion in MOCK_POLICIES["health_exclusions"]:
            response_text += f"• {exclusion}\n"
        citations = ["Health Insurance Policy (Page 6)"]
        
    elif "claim" in user_message_lower:
        response_text = "Claims process:\n"
        for step in MOCK_POLICIES["claim_process"]:
            response_text += f"• {step}\n"
        citations = ["General Terms & Conditions (Page 8)"]
        
    elif "grace period" in user_message_lower:
        response_text = f"The grace period for premium payment is {MOCK_POLICIES['grace_period']}. If premiums are not paid within this period, the policy will lapse."
        citations = ["Term Life Insurance Policy (Page 5)", "General Terms & Conditions (Page 4)"]
        
    elif "surrender" in user_message_lower and "propensity" in user_message_lower:
        # Mock surrender propensity
        if request.customer_id:
            prob = random.uniform(0.1, 0.9)
            risk_level = "high" if prob >= 0.7 else "medium" if prob >= 0.3 else "low"
            
            decision = {
                "customer_id": request.customer_id,
                "probability": round(prob, 3),
                "risk_level": risk_level,
                "action": "escalate_to_retention" if risk_level == "high" else "send_education" if risk_level == "medium" else "normal_service",
                "recommendation": "Immediate retention intervention required" if risk_level == "high" else "Send personalized education materials" if risk_level == "medium" else "Continue regular service",
                "top_factors": ["payment_gaps", "fund_value_change", "sentiment_score"][:random.randint(2, 3)]
            }
            
            response_text = f"Surrender Risk Analysis for Customer {request.customer_id}:\n"
            response_text += f"• Probability: {decision['probability']:.1%}\n"
            response_text += f"• Risk Level: {risk_level.capitalize()}\n"
            response_text += f"• Recommendation: {decision['recommendation']}\n"
            response_text += f"• Key Factors: {', '.join(decision['top_factors'])}"
            
            tools_used = ["get_surrender_propensity"]
        else:
            response_text = "Please provide a customer ID to check surrender propensity."
            
    elif "policy" in user_message_lower and request.policy_id:
        # Mock policy summary
        response_text = f"Policy Summary for {request.policy_id}:\n"
        response_text += f"• Type: Term Life Insurance\n"
        response_text += f"• Status: Active\n"
        response_text += f"• Premium: ₹5,000/month\n"
        response_text += f"• Sum Assured: ₹10,00,000\n"
        response_text += f"• Term: 20 years\n"
        response_text += f"• Surrender Charge: 4% (Year 4)"
        tools_used = ["get_policy_summary"]
        
    else:
        # Default response
        response_text = "I can help you with:\n"
        response_text += "• Policy terms and conditions\n"
        response_text += "• Surrender charges and values\n"
        response_text += "• Claims process\n"
        response_text += "• Health insurance coverage\n"
        response_text += "• ULIP fund options\n"
        response_text += "\nPlease ask me a specific question about your insurance policy."
    
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
    
    return response

@app.post("/tools/get_surrender_propensity")
async def get_surrender_propensity(customer_id: str):
    """Get surrender propensity for a customer"""
    
    # Mock response
    prob = random.uniform(0.1, 0.9)
    risk_level = "high" if prob >= 0.7 else "medium" if prob >= 0.3 else "low"
    
    return {
        "customer_id": customer_id,
        "probability": round(prob, 3),
        "risk_level": risk_level,
        "top_factors": ["payment_gaps", "fund_value_change", "sentiment_score"],
        "recommendation": "Immediate retention intervention" if risk_level == "high" else "Send education materials" if risk_level == "medium" else "Continue normal service",
        "model_version": "v1.0-mock",
        "scored_at": datetime.now().isoformat()
    }

@app.post("/tools/get_policy_summary")
async def get_policy_summary(policy_id: str):
    """Get policy summary"""
    
    return {
        "policy_id": policy_id,
        "customer_name": "John Doe",
        "policy_type": "Term Life",
        "status": "Active",
        "premium_amount": 5000.00,
        "sum_assured": 1000000.00,
        "surrender_charge_pct": 4.0,
        "lock_in_months_remaining": 12,
        "last_premium_paid": datetime.now().date().isoformat()
    }

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

# Databricks connection status endpoint
@app.get("/databricks/status")
async def databricks_status():
    """Check Databricks connection status"""
    
    # For now, return mock status
    return {
        "connected": False,
        "mode": "mock_data",
        "message": "Using mock data. Configure Databricks credentials to connect to live data.",
        "catalog": "insurance_copilot",
        "schemas": ["policy_data", "ml_models", "features"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)