# 🛡️ Insurance Policy Copilot - Running Services

## ✅ Services Status

Your Insurance Policy Copilot is now running with the following services:

| Service | URL | Status |
|---------|-----|--------|
| **Chat UI** | http://localhost:8501 | ✅ Running |
| **API Backend** | http://localhost:8001 | ✅ Running |
| **API Documentation** | http://localhost:8001/docs | ✅ Available |
| **Health Check** | http://localhost:8001/health | ✅ Healthy |

## 🎯 Quick Access Links

### 1. **Chat Interface** - http://localhost:8501
   - Interactive chat UI with the insurance copilot
   - Enter Customer ID: `C000001` for testing
   - Enter Policy ID: `P00000001` for testing
   - Try sample questions from the sidebar

### 2. **API Documentation** - http://localhost:8001/docs
   - Interactive Swagger/OpenAPI documentation
   - Test API endpoints directly
   - View request/response schemas

### 3. **Direct API Access** - http://localhost:8001
   - Base API endpoint for programmatic access
   - Health check: http://localhost:8001/health
   - Databricks status: http://localhost:8001/databricks/status

## 📝 Sample Interactions

### Chat UI Questions to Try:
1. "What is the surrender charge in year 2?"
2. "What is the lock-in period for ULIP?"
3. "What are the exclusions in health insurance?"
4. "How do I file a claim?"
5. "What is the grace period for premium payment?"

### With Customer Context:
- Enter Customer ID: `C000001`
- Ask: "Check surrender propensity for this customer"

### With Policy Context:
- Enter Policy ID: `P00000001`
- Ask: "Get summary for this policy"

## 🔧 Current Configuration

- **Mode**: MOCK DATA (not connected to Databricks)
- **Backend Port**: 8001
- **Frontend Port**: 8501
- **API Base URL**: http://localhost:8001

## 📊 Databricks Integration

To connect to real Databricks data:

1. **Log in to Databricks Community Edition**
   - URL: https://community.cloud.databricks.com
   - Email: smn.mukherjee@gmail.com

2. **Upload Notebooks**
   - Navigate to Workspace → Create → Folder → "insurance-copilot"
   - Upload from `notebooks/` directory:
     - `01_databricks_setup.py`
     - `02_train_surrender_model.py`
     - `03_upload_data.py`

3. **Run Setup**
   - Execute notebooks in order (1, 2, 3)
   - This creates tables, trains models, and loads data

4. **Get Access Token**
   - User Settings → Developer → Access Tokens
   - Generate New Token

5. **Update Configuration**
   ```bash
   # Edit .env file
   DATABRICKS_HOST=https://community.cloud.databricks.com
   DATABRICKS_TOKEN=<your-token>
   USE_MOCK_DATA=false
   ```

6. **Restart Services**
   ```bash
   # Stop current services (Ctrl+C)
   # Restart with new configuration
   ./start_services.sh
   ```

## 🛑 Stopping Services

To stop all services, press `Ctrl+C` in the terminal or run:
```bash
pkill -f "uvicorn.*8001"
pkill -f "streamlit.*8501"
```

## 📁 Project Structure

```
ai-policy-copilot/
├── app/                 # FastAPI backend
│   ├── main_simple.py   # Simplified API (currently running)
│   ├── main.py          # Full API with all features
│   ├── rag_pipeline.py  # RAG implementation
│   ├── guardrails.py    # PII & security
│   └── tools.py         # Business logic
├── ui/                  # Streamlit frontend
│   └── streamlit_app.py # Chat interface
├── data/                # Generated data
│   ├── *.parquet        # Synthetic insurance data
│   └── policy_documents/ # PDF documents
├── notebooks/           # Databricks notebooks
└── .env                 # Configuration

```

## 🚀 Next Steps

1. **Explore the Chat UI** at http://localhost:8501
2. **Test API endpoints** at http://localhost:8001/docs
3. **Set up Databricks** for real data integration
4. **Review generated data** in `data/` folder

## 📞 Troubleshooting

If services are not accessible:

1. **Check if services are running:**
   ```bash
   curl http://localhost:8001/health
   curl http://localhost:8501
   ```

2. **Check for port conflicts:**
   ```bash
   lsof -i :8001
   lsof -i :8501
   ```

3. **Restart services:**
   ```bash
   cd ai-policy-copilot
   ./start_services.sh
   ```

## 📈 Features Available

### In Mock Mode (Current):
- ✅ Policy Q&A with mock responses
- ✅ Surrender propensity simulation
- ✅ Policy summary generation
- ✅ Interactive chat interface
- ✅ API documentation

### With Databricks Connected:
- 🔄 Real ML model predictions
- 🔄 50,000+ customer records
- 🔄 120,000+ policy records
- 🔄 Feature store integration
- 🔄 MLflow experiment tracking
- 🔄 Delta Lake tables

---

**Your Insurance Policy Copilot is ready to use!** 🎉

Visit http://localhost:8501 to start chatting with the copilot.