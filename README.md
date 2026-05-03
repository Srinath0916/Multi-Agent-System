# Multi-Agent Travel Planner System

A sophisticated multi-agent AI system built with LangChain and LangGraph that creates comprehensive travel plans through collaborative agent workflows.

## 🤖 Agent Architecture

The system uses 4 specialized agents working in sequence:

1. **Planner Agent** - Extracts structured information (destination, dates, budget, preferences) from natural language requests
2. **Research Agent** - Gathers destination highlights, attractions, and practical travel tips
3. **Itinerary Builder Agent** - Creates detailed day-by-day travel plans incorporating research findings
4. **Budget Estimator Agent** - Provides cost breakdowns and budget feasibility analysis

## 🏗️ Technical Stack

- **LangGraph**: Workflow orchestration with directed graph
- **LangChain**: LLM integration and prompt management
- **Groq API**: Fast inference with llama-3.3-70b-versatile model
- **Streamlit**: Interactive web interface
- **Python 3.8+**: Core implementation

## 📦 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd multi-agent-travel-planner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

Get a free Groq API key at: https://console.groq.com/keys

## 🚀 Usage

### Streamlit Web Interface (Recommended)
```bash
streamlit run multi_agent_system.py
```

### CLI Mode
```bash
python multi_agent_system.py
```

## 💡 Example Requests

- "I want to visit Tokyo for 7 days in March with a budget of $3000. I love food and temples."
- "Plan a romantic trip to Paris for 5 days in June. Budget is $2500. We enjoy art museums and cafes."
- "Family vacation to Bali for 10 days in December. Budget $5000. Kids love beaches and animals."
- "Solo backpacking trip to Thailand for 2 weeks. Budget $1500. Interested in culture and nightlife."

## 🔄 Workflow

```
User Input → Planner Agent → Research Agent → Itinerary Builder → Budget Estimator → Final Plan
```

Each agent:
- Receives shared state from previous agent
- Performs specialized task
- Updates state with results
- Passes to next agent

## 📊 State Management

The system uses a shared `TravelState` TypedDict that flows through all agents:

```python
{
    "user_input": str,
    "destination": str,
    "travel_dates": str,
    "budget": str,
    "preferences": str,
    "research_notes": str,
    "itinerary": str,
    "budget_estimate": str
}
```

## 🎯 Key Features

- **Natural Language Processing**: Understands free-text travel requests
- **Collaborative Agents**: Each agent specializes in one aspect of travel planning
- **Structured Workflow**: LangGraph ensures proper agent sequencing
- **Rich Output**: Comprehensive travel plans with itineraries and budgets
- **Dual Interface**: Both CLI and web UI available
- **Download Support**: Export travel plans as text files

## 🔧 Configuration

### Local Development
Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
```

### Streamlit Cloud Deployment
Add to App Settings → Secrets:
```toml
GROQ_API_KEY = "your_key_here"
```

## 📝 Assignment Requirements

✅ 3-4 agents with clear roles  
✅ LangGraph nodes and edges  
✅ Shared state/context passing  
✅ main() function  
✅ Dynamic user input  
✅ Single Python file implementation

## 🎥 Demo Video

[Link to demo video - 5-8 minutes with voice explanation]

## 📄 License

MIT License - Feel free to use for educational purposes
