# imports
import asyncio

from google.genai import types

from google.adk import Runner
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService

# adktools imports
import test_agent.tools
from adktools import discover_adk_tools  

# set the agent model
AGENT_MODEL = "anthropic/claude-3-5-sonnet-latest"

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


APP_NAME="google_search_agent"
USER_ID="user1234"
SESSION_ID="1234"

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction (Async)
async def call_agent_async(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    print(f"\n--- Running Query: {query} ---")
    final_response_text = "No final text response captured."
    try:
        # Use run_async
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            print(f"Event ID: {event.id}, Author: {event.author}")

            # --- Check for specific parts FIRST ---
            has_specific_part = False
            if event.content and event.content.parts:
                for part in event.content.parts: # Iterate through all parts
                    if part.executable_code:
                        # Access the actual code string via .code
                        print(f"  Debug: Agent generated code:\n```python\n{part.executable_code.code}\n```")
                        has_specific_part = True
                    elif part.code_execution_result:
                        # Access outcome and output correctly
                        print(f"  Debug: Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}")
                        has_specific_part = True
                    # Also print any text parts found in any event for debugging
                    elif part.text and not part.text.isspace():
                        print(f"  Text: '{part.text.strip()}'")
                        # Do not set has_specific_part=True here, as we want the final response logic below

            # --- Check for final response AFTER specific parts ---
            # Only consider it final if it doesn't have the specific code parts we just handled
            if not has_specific_part and event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_response_text = event.content.parts[0].text.strip()
                    print(f"==> Final Agent Response: {final_response_text}")
                else:
                    print("==> Final Agent Response: [No text content in final event]")
    except Exception as e:
        print(f"ERROR during agent run: {e}")
    print("-" * 30)


if __name__ == "__main__":
    # Main async function to run the examples
    async def main():
        await call_agent_async("time in nyc ")
        await call_agent_async("time at delhi ")

    # Execute the main async function
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # Handle specific error when running asyncio.run in an already running loop (like Jupyter/Colab)
        if "cannot be called from a running event loop" in str(e):
            print("\nRunning in an existing event loop (like Colab/Jupyter).")
            print("Please run `await main()` in a notebook cell instead.")
            # If in an interactive environment like a notebook, you might need to run:
            # await main()
        else:
            raise e # Re-raise other runtime errors