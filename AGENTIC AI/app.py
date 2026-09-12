import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# Page Configuration
st.set_page_config(
    page_title="AI Search & Weather Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# --- Custom CSS for Enhanced UI ---
st.markdown("""
    <style>
    /* Chat Container Styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Header Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Define Tools ---
@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city"""
    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not set"

    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    
    try:
        response = requests.get(url)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}: {data}"

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

# --- Helper Functions ---
def init_agent():
    """Initialize agent with current API keys."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None

    try:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=groq_key
        )
        search_tool = TavilySearch(max_results=1)
        tools = [search_tool, get_weather]
        return create_agent(model=llm, tools=tools)
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    # API Key Status Indicators
    st.subheader("API Status")
    groq_status = "🟢 Set" if os.getenv("GROQ_API_KEY") else "🔴 Missing"
    tavily_status = "🟢 Set" if os.getenv("TAVILY_API_KEY") else "🔴 Missing"
    weather_status = "🟢 Set" if os.getenv("WEATHERSTACK_API_KEY") else "🔴 Missing"

    st.write(f"**Groq API:** {groq_status}")
    st.write(f"**Tavily API:** {tavily_status}")
    st.write(f"**WeatherStack API:** {weather_status}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main App Header ---
st.markdown('<div class="main-header">🤖 AI Agent Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Groq, Tavily Search, and WeatherStack</div>', unsafe_allow_html=True)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input & Agent Processing ---
if prompt := st.chat_input("Ask about weather, news, or general knowledge..."):
    # Render user prompt immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Instantiate Agent
    agent = init_agent()

    with st.chat_message("assistant"):
        if not agent:
            error_msg = "Please ensure your `GROQ_API_KEY` is set in your `.env` file."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Agent is searching and thinking..."):
                try:
                    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                    answer = response["messages"][-1].content
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred while executing the request: {str(e)}")