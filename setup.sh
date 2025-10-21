#!/bin/bash

# Insurance Policy Copilot - Setup Script
# This script automates the initial setup process

set -e

echo "🛡️  Insurance Policy Copilot - Setup Script"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_success "Python $python_version is installed"
else
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
print_info "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
print_info "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
print_success "Dependencies installed"

# Create necessary directories
print_info "Creating project directories..."
mkdir -p data/policy_documents
mkdir -p data/vector_index
mkdir -p logs
mkdir -p models
print_success "Directories created"

# Copy .env template if not exists
if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    cp .env.template .env
    print_warning "Please edit .env file with your Azure/Databricks credentials"
else
    print_info ".env file already exists. Skipping..."
fi

# Generate synthetic data
print_info "Generating synthetic insurance data..."
python -c "
from data.synthetic_data_generator import InsuranceDataGenerator
generator = InsuranceDataGenerator(num_customers=1000)
all_data = generator.generate_all_data()
print('✓ Generated synthetic data')
"

# Generate policy documents
print_info "Generating sample policy documents..."
python -c "
from data.policy_document_generator import PolicyDocumentGenerator
generator = PolicyDocumentGenerator()
generator.generate_all_documents()
print('✓ Generated policy documents')
"

# Process documents for RAG
print_info "Processing documents for RAG pipeline..."
python -c "
from app.rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.process_documents()
print('✓ Documents indexed for RAG')
"

# Check Docker installation
print_info "Checking Docker installation..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | grep -Po '\d+\.\d+\.\d+')
    print_success "Docker $docker_version is installed"
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        compose_version=$(docker-compose --version | grep -Po '\d+\.\d+\.\d+')
        print_success "Docker Compose $compose_version is installed"
    else
        print_warning "Docker Compose is not installed. Install it for containerized deployment"
    fi
else
    print_warning "Docker is not installed. Install Docker for containerized deployment"
fi

# Create run scripts
print_info "Creating run scripts..."

# Create run_backend.sh
cat > run_backend.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
EOF
chmod +x run_backend.sh

# Create run_frontend.sh
cat > run_frontend.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
EOF
chmod +x run_frontend.sh

# Create run_all.sh
cat > run_all.sh << 'EOF'
#!/bin/bash
# Run both backend and frontend
./run_backend.sh &
BACKEND_PID=$!
./run_frontend.sh &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
EOF
chmod +x run_all.sh

print_success "Run scripts created"

# Setup completion message
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your credentials:"
echo "   ${YELLOW}nano .env${NC}"
echo ""
echo "2. Run Databricks setup (if using Databricks):"
echo "   - Upload notebooks/01_databricks_setup.py to your Databricks workspace"
echo "   - Run the notebook to create tables and schemas"
echo ""
echo "3. Start the application:"
echo ""
echo "   Option A - Run locally:"
echo "   ${GREEN}./run_backend.sh${NC}    # Terminal 1: Start backend"
echo "   ${GREEN}./run_frontend.sh${NC}   # Terminal 2: Start frontend"
echo ""
echo "   Option B - Run with Docker:"
echo "   ${GREEN}docker-compose up --build${NC}"
echo ""
echo "4. Access the application:"
echo "   - Chat UI: http://localhost:8501"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "For more information, see README.md"
echo ""
print_success "Happy coding! 🚀"