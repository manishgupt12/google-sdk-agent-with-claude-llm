# imports
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# adktools imports
import test_agent.tools
from adktools import discover_adk_tools  

# set the agent model
AGENT_MODEL = "openai/gpt-4o-mini"

# Root agent using auto-discovered tools
root_agent = Agent(
    name="time_agent",
    model=LiteLlm(model=AGENT_MODEL),
    description="Provides current time for specified timezone",
    instruction="You are a helpful time assistant. Your primary goal is to provide the current time for given timezones or cities. "
                "When the user asks for the time in a specific city or time zone "
                "you MUST use the 'get_time' tool to find the information. "
                "Analyze the tool's response: if the status is 'error', inform the user politely about the error message. "
                "If the status is 'success', present the information clearly and concisely to the user. "
                "Only use the tools when appropriate for a time-related request.",
    tools=discover_adk_tools(test_agent.tools),
)