"""
Multi-Agent Travel Planner System
Assignment: Build a Multi-Agent System using LangChain + LangGraph

This system uses 4 specialized agents to create comprehensive travel plans:
1. Planner Agent - Extracts structured information from user requests
2. Research Agent - Gathers destination information and attractions
3. Itinerary Builder Agent - Creates day-by-day travel itineraries
4. Budget Estimator Agent - Provides cost breakdowns and budget analysis

The agents collaborate through a shared state and are orchestrated
using LangGraph's directed workflow.

Usage:
    CLI Mode: python multi_agent_system.py
    Streamlit Mode: streamlit run multi_agent_system.py
"""

import os
import sys
import json
from typing import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# SHARED STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Shared state passed between all agents in the workflow."""
    user_input: str
    destination: str
    travel_dates: str
    budget: str
    preferences: str
    research_notes: str
    itinerary: str
    budget_estimate: str

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

def get_api_key():
    """Get Groq API key from environment variables or Streamlit secrets"""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except:
            pass
    
    return api_key

api_key = get_api_key()
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in:\n"
        "- Local: .env file with GROQ_API_KEY=your_key\n"
        "- Streamlit Cloud: App Settings → Secrets → Add GROQ_API_KEY = \"your_key\""
    )

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=api_key)

# ============================================================================
# AGENT 1: PLANNER AGENT
# ============================================================================

def planner_agent(state: AgentState) -> AgentState:
    """Planner Agent: Extracts structured travel information from user input."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel planning assistant. Extract structured travel intent from the user's request.\n"
         "Return ONLY a valid JSON object with these exact keys: destination, travel_dates, budget, preferences.\n"
         "Do not include any other text, explanations, or markdown formatting.\n"
         "If you cannot determine a value, use an empty string. If destination is unclear, use \"unknown\".\n\n"
         "Example output:\n"
         '{{"destination": "Paris", "travel_dates": "June 15-22", "budget": "$2000", "preferences": "art museums and cafes"}}'),
        ("human", "Travel request: {user_input}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"user_input": state["user_input"]})
    
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        parsed = json.loads(content)
        state["destination"] = parsed.get("destination", "unknown")
        state["travel_dates"] = parsed.get("travel_dates", "")
        state["budget"] = parsed.get("budget", "")
        state["preferences"] = parsed.get("preferences", "")
    except Exception as e:
        print(f"Warning: Failed to parse planner response: {e}")
        state["destination"] = "unknown"
        state["travel_dates"] = ""
        state["budget"] = ""
        state["preferences"] = ""
    
    return state

# ============================================================================
# AGENT 2: RESEARCH AGENT
# ============================================================================

def research_agent(state: AgentState) -> AgentState:
    """Research Agent: Gathers destination information and travel tips."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a destination research specialist. Provide at least 3 highlights, top attractions,\n"
         "and practical travel tips for the given destination and traveller preferences.\n"
         "If destination is \"unknown\", provide general travel tips instead."),
        ("human", "Destination: {destination}\nPreferences: {preferences}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "preferences": state["preferences"],
    })
    
    state["research_notes"] = response.content
    return state

# ============================================================================
# AGENT 3: ITINERARY BUILDER AGENT
# ============================================================================

def itinerary_agent(state: AgentState) -> AgentState:
    """Itinerary Builder Agent: Creates detailed day-by-day travel plans."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert travel itinerary builder. Create a detailed day-by-day itinerary.\n"
         "Each day must include at least one activity drawn directly from the research notes provided.\n"
         "Format clearly with Day 1, Day 2, etc."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Preferences: {preferences}\n"
         "Research Notes: {research_notes}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "preferences": state["preferences"],
        "research_notes": state["research_notes"],
    })
    
    state["itinerary"] = response.content
    return state

# ============================================================================
# AGENT 4: BUDGET ESTIMATOR AGENT
# ============================================================================

def budget_agent(state: AgentState) -> AgentState:
    """Budget Estimator Agent: Provides cost analysis and budget breakdown."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel budget analyst. Provide a cost breakdown covering accommodation,\n"
         "transport, food, and activities. If a numeric budget is provided, explicitly state\n"
         "whether the estimated total is within, at, or over that budget."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Budget: {budget}\n"
         "Itinerary: {itinerary}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "budget": state["budget"],
        "itinerary": state["itinerary"],
    })
    
    state["budget_estimate"] = response.content
    return state

# ============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ============================================================================

def create_workflow():
    """Build the LangGraph workflow connecting all agents."""
    
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("researcher", research_agent)
    workflow.add_node("itinerary_builder", itinerary_agent)
    workflow.add_node("budget_estimator", budget_agent)
    
    # Define workflow edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "itinerary_builder")
    workflow.add_edge("itinerary_builder", "budget_estimator")
    workflow.add_edge("budget_estimator", END)
    
    return workflow.compile()

# ============================================================================
# EXECUTION FUNCTION
# ============================================================================

def run_travel_planner(user_input: str) -> AgentState:
    """Execute the multi-agent travel planning workflow."""
    
    initial_state = AgentState(
        user_input=user_input,
        destination="",
        travel_dates="",
        budget="",
        preferences="",
        research_notes="",
        itinerary="",
        budget_estimate=""
    )
    
    app = create_workflow()
    final_state = app.invoke(initial_state)
    
    return final_state

# ============================================================================
# CLI MODE
# ============================================================================

def print_results(state: AgentState) -> None:
    """Print travel plan to console (CLI mode)"""
    print("\n" + "="*80)
    print("TRAVEL PLAN GENERATED")
    print("="*80)
    
    print("\n📍 DESTINATION")
    print("-" * 80)
    print(state["destination"] if state["destination"] else "Not determined")
    
    print("\n📅 TRAVEL DATES")
    print("-" * 80)
    print(state["travel_dates"] if state["travel_dates"] else "Not specified")
    
    print("\n💰 BUDGET")
    print("-" * 80)
    print(state["budget"] if state["budget"] else "Not specified")
    
    print("\n🎯 PREFERENCES")
    print("-" * 80)
    print(state["preferences"] if state["preferences"] else "Not specified")
    
    print("\n📝 ITINERARY")
    print("-" * 80)
    print(state["itinerary"] if state["itinerary"] else "No itinerary generated")
    
    print("\n💵 BUDGET ESTIMATE")
    print("-" * 80)
    print(state["budget_estimate"] if state["budget_estimate"] else "No estimate generated")
    
    print("\n" + "="*80)

def main() -> None:
    """Main function for CLI mode"""
    print("="*80)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("="*80)
    print("\nThis system uses 4 AI agents to create your perfect travel plan:")
    print("  1. Planner Agent - Extracts travel details")
    print("  2. Research Agent - Finds attractions and tips")
    print("  3. Itinerary Builder - Creates day-by-day plans")
    print("  4. Budget Estimator - Analyzes costs")
    print("\n" + "="*80 + "\n")
    
    user_input = ""
    while not user_input.strip():
        user_input = input("Enter your travel request: ")
    
    print("\n� Processing your request through 4 specialized agents...")
    print("   This may take 15-30 seconds...\n")
    
    final_state = run_travel_planner(user_input)
    print_results(final_state)

# ============================================================================
# STREAMLIT MODE
# ============================================================================

def run_streamlit_app():
    """Run Streamlit web interface"""
    import streamlit as st
    
    st.set_page_config(
        page_title="Multi-Agent Travel Planner",
        page_icon="✈️",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">✈️ Multi-Agent Travel Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by LangChain + LangGraph | 4 Specialized AI Agents</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🤖 Agent System")
        st.markdown("---")
        st.markdown("### Agent 1: Planner")
        st.markdown("� Extracts destination, dates, budget, and preferences")
        st.markdown("### Agent 2: Research")
        st.markdown("🔍 Gathers attractions, highlights, and travel tips")
        st.markdown("### Agent 3: Itinerary Builder")
        st.markdown("📅 Creates detailed day-by-day travel plans")
        st.markdown("### Agent 4: Budget Estimator")
        st.markdown("💰 Provides cost breakdown and budget analysis")
        st.markdown("---")
        st.markdown("**Technology Stack:**")
        st.markdown("- LangGraph for workflow")
        st.markdown("- LangChain for LLM integration")
        st.markdown("- Groq API (llama-3.3-70b)")
        st.markdown("- Streamlit for UI")
    
    st.markdown("### 📝 Enter Your Travel Request")
    st.markdown("Describe your ideal trip in natural language. Include destination, dates, budget, and preferences.")
    
    with st.expander("💡 See Example Requests"):
        st.markdown("""
- "I want to visit Tokyo for 7 days in March with a budget of $3000. I love food and temples."
- "Plan a romantic trip to Paris for 5 days in June. Budget is $2500. We enjoy art museums and cafes."
- "Family vacation to Bali for 10 days in December. Budget $5000. Kids love beaches and animals."
- "Solo backpacking trip to Thailand for 2 weeks. Budget $1500. Interested in culture and nightlife."
        """)
    
    user_input = st.text_area(
        "Your Travel Request:",
        height=100,
        placeholder="Example: I want to travel to Tokyo for 7 days in March with a budget of $3000. I love food and temples."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button("🚀 Generate Travel Plan", use_container_width=True, type="primary")
    
    if generate_button:
        if not user_input.strip():
            st.error("⚠️ Please enter a travel request")
        else:
            with st.spinner("🔄 Processing through 4 AI agents... This may take 15-30 seconds..."):
                try:
                    final_state = run_travel_planner(user_input)
                    
                    st.success("✅ Travel plan generated successfully!")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["📍 Overview", "📝 Itinerary", "💵 Budget", "🔍 Research"])
                    
                    with tab1:
                        st.markdown("### 📍 Trip Overview")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Destination:**")
                            st.info(final_state["destination"] or "Not determined")
                            st.markdown("**Budget:**")
                            st.info(final_state["budget"] or "Not specified")
                        
                        with col2:
                            st.markdown("**Travel Dates:**")
                            st.info(final_state["travel_dates"] or "Not specified")
                            st.markdown("**Preferences:**")
                            st.info(final_state["preferences"] or "Not specified")
                    
                    with tab2:
                        st.markdown("### 📝 Day-by-Day Itinerary")
                        if final_state["itinerary"]:
                            st.markdown(final_state["itinerary"])
                        else:
                            st.warning("No itinerary generated")
                    
                    with tab3:
                        st.markdown("### 💵 Budget Breakdown")
                        if final_state["budget_estimate"]:
                            st.markdown(final_state["budget_estimate"])
                        else:
                            st.warning("No budget estimate generated")
                    
                    with tab4:
                        st.markdown("### 🔍 Destination Research")
                        if final_state["research_notes"]:
                            st.markdown(final_state["research_notes"])
                        else:
                            st.warning("No research notes generated")
                    
                    st.markdown("---")
                    travel_plan_text = f"""TRAVEL PLAN
===========

Destination: {final_state['destination']}
Travel Dates: {final_state['travel_dates']}
Budget: {final_state['budget']}
Preferences: {final_state['preferences']}

ITINERARY
=========
{final_state['itinerary']}

BUDGET ESTIMATE
===============
{final_state['budget_estimate']}

RESEARCH NOTES
==============
{final_state['research_notes']}
"""
                    st.download_button(
                        label="📥 Download Travel Plan",
                        data=travel_plan_text,
                        file_name="travel_plan.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please check your GROQ_API_KEY in .env file or Streamlit secrets")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        import streamlit as st
        if hasattr(st, 'runtime') and st.runtime.exists():
            run_streamlit_app()
        else:
            main()
    except ImportError:
        main()
