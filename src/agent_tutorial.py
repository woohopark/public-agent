import os
from typing import List, Dict, TypedDict
import requests
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph

# Load environment variables
load_dotenv()

# Define the state type
class AgentState(TypedDict):
    messages: List[Dict[str, str]]

def create_message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}

def call_ollama_api(prompt: str) -> str:
    """Call Ollama API directly"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Define agent node
def agent(state: AgentState) -> AgentState:
    print("Starting agent processing...")
    
    # Get the last message content
    last_message = state["messages"][-1]["content"]
    print(f"Processing message: {last_message}")
    
    try:
        # Generate response using Ollama API
        print("Calling Ollama API...")
        response_text = call_ollama_api(last_message)
        print(f"Received response: {response_text}")
        
        # Add the AI response to messages
        messages = state["messages"]
        messages.append(create_message("assistant", response_text))
        
        return {"messages": messages}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

# Create the graph
def create_graph():
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add the agent node
    workflow.add_node("agent", agent)
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Set the end point
    workflow.set_finish_point("agent")
    
    # Compile the graph
    return workflow.compile()

def main():
    print("Starting the agent...")
    
    # Create the graph
    graph = create_graph()
    
    # Create initial state with the question
    initial_state = {
        "messages": [create_message("user", "프랑스의 수도는 어디인가요?")]
    }
    
    # Run the graph
    print("Running the graph...")
    for output in graph.stream(initial_state):
        if "messages" in output:
            # Print the last message
            last_message = output["messages"][-1]
            print(f"{last_message['role']}: {last_message['content']}")

if __name__ == "__main__":
    main()