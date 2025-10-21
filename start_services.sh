#!/bin/bash

# Start Insurance Copilot Services
echo "🛡️ Starting Insurance Policy Copilot Services..."
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install minimal required packages if not present
echo -e "${YELLOW}Installing core dependencies...${NC}"
pip install -q fastapi uvicorn streamlit pandas numpy faker pypdf 2>/dev/null

# Start FastAPI backend
echo -e "${GREEN}Starting FastAPI backend on http://localhost:8000${NC}"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start Streamlit frontend
echo -e "${GREEN}Starting Streamlit UI on http://localhost:8501${NC}"
python -m streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Services are starting!${NC}"
echo "=========================================="
echo ""
echo "Access points:"
echo "  • Chat UI: http://localhost:8501"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait