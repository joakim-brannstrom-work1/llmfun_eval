"""Evaluator module for calling llmfun agent and parsing responses.

This module handles:
- Calling the llmfun agent with prompts from test cases
- Parsing the agent's response to extract tools called
- Handling the chat history format
- Copying images to llmfun workarea for multimodal evaluation
"""

import subprocess
import json
import re
import time
import shutil
import os, stat
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

# Path to llmfun workarea where images should be copied
LLMFUN_WORKAREA = (Path(".") / "llmfun" / "workarea").resolve()

# Default path where llmfun saves chat history after each run
LLMFUN_HISTORY_PATH = (Path(".") / "llmfun" / "data" / "scratch" / "main_history.json").resolve()

LLMFUN_CMD = (Path(".") / "program" / "llmfun").resolve()
LLMFUN_CONF = (Path(".") / "config" / "default.json").resolve()

# Tool to escalation level mapping
TOOL_ESCALATION_MAP = {
    "trackerLight": 1,
    "audibleWarning": 2,
    "strobeLight": 3,
    "humanOperator": 4,
    "callPatrol": 5,
}


@dataclass
class AgentResponse:
    """Response from the llmfun agent."""
    raw_output: str
    tools_called: List[str]
    escalation_level: int
    reasoning: str
    latency_ms: float

def agent_cleanup(keep_history):
    """Clean the agent workspace and history"""
    if not keep_history and LLMFUN_HISTORY_PATH.exists():
        LLMFUN_HISTORY_PATH.unlink()

    for file in LLMFUN_WORKAREA.glob('*.jpg'):
        current = os.stat(str(file)).st_mode
        os.chmod(str(file), current | stat.S_IWUSR)
        file.unlink()


def call_llmfun_agent(prompt: str, history_file: Optional[Path] = None) -> AgentResponse:
    """Call the llmfun agent with a prompt.
    
    Args:
        prompt: The prompt to send to the agent
        history_file: Optional path to chat history JSON file
        
    Returns:
        AgentResponse object with parsed output
    """
    start_time = time.time()
    
    # Build the command
    cmd = [LLMFUN_CMD, "-c", LLMFUN_CONF, "agent", "-p", prompt]
    
    # If history file is provided, we could load it and pass context
    # For now, we just use the prompt directly
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if result.returncode != 0:
            # Agent returned an error
            return AgentResponse(
                raw_output=result.stderr,
                tools_called=[],
                escalation_level=0,
                reasoning=f"Error: {result.stderr}",
                latency_ms=latency_ms
            )
        
        output = result.stdout
        
    except subprocess.TimeoutExpired:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResponse(
            raw_output="",
            tools_called=[],
            escalation_level=0,
            reasoning="Timeout after 60 seconds",
            latency_ms=latency_ms
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResponse(
            raw_output="",
            tools_called=[],
            escalation_level=0,
            reasoning=f"Exception: {str(e)}",
            latency_ms=latency_ms
        )
    
    # Parse the response to extract tools
    tools_called = parse_tools_from_response(output)
    escalation_level = calculate_escalation_level(tools_called)
    reasoning = extract_reasoning(output)
    
    return AgentResponse(
        raw_output=output,
        tools_called=tools_called,
        escalation_level=escalation_level,
        reasoning=reasoning,
        latency_ms=latency_ms
    )


def call_llmfun_agent_with_history(prompt: str) -> AgentResponse:
    """Call the llmfun agent and extract tool calls from the saved history.
    
    This method runs the agent, then parses the history JSON that is always
    saved to llmfun/data/scratch/main_history.json to extract tool calls.
    This is more reliable than parsing raw output with regex.
    
    The history is saved automatically by the llmfun agent after each run.
    
    Args:
        prompt: The prompt to send to the agent
        
    Returns:
        AgentResponse object with parsed output
    """
    start_time = time.time()
    
    # Build the command
    cmd = [LLMFUN_CMD, "-c", LLMFUN_CONF, "agent", "-p", prompt]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if result.returncode != 0:
            # Agent returned an error
            return AgentResponse(
                raw_output=result.stderr,
                tools_called=[],
                escalation_level=0,
                reasoning=f"Error: {result.stderr}",
                latency_ms=latency_ms
            )
        
        output = result.stdout
        
    except subprocess.TimeoutExpired:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResponse(
            raw_output="",
            tools_called=[],
            escalation_level=0,
            reasoning="Timeout after 60 seconds",
            latency_ms=latency_ms
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResponse(
            raw_output="",
            tools_called=[],
            escalation_level=0,
            reasoning=f"Exception: {str(e)}",
            latency_ms=latency_ms
        )
    
    # Extract tool calls from the history file (the preferred method)
    tools_called = parse_tools_from_history_file(LLMFUN_HISTORY_PATH)
    
    # Fallback to regex parsing if no tools found in history
    if not tools_called:
        tools_called = parse_tools_from_response(output)
    
    escalation_level = calculate_escalation_level(tools_called)
    reasoning = extract_reasoning(output)
    
    return AgentResponse(
        raw_output=output,
        tools_called=tools_called,
        escalation_level=escalation_level,
        reasoning=reasoning,
        latency_ms=latency_ms
    )


def parse_tools_from_history(history_data: Dict) -> List[str]:
    """Parse the chat history JSON to extract tool names.
    
    This is the preferred method as it extracts structured data directly
    from the saved history instead of using fragile regex parsing.
    
    The history JSON should have this structure:
    {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "toolName"}}]},
            ...
        ]
    }
    
    Args:
        history_data: Parsed JSON data from history file
        
    Returns:
        List of tool names found in the history
    """
    tools_found = []
    
    # Known tools to look for
    known_tools = set(TOOL_ESCALATION_MAP.keys())
    
    messages = history_data.get('messages', [])
    
    for message in messages:
        # Look for assistant messages with tool_calls
        if message.get('role') == 'assistant':
            tool_calls = message.get('tool_calls', [])
            for tool_call in tool_calls:
                # Extract function name from the tool call
                if isinstance(tool_call, dict):
                    function = tool_call.get('function', {})
                    if isinstance(function, dict):
                        tool_name = function.get('name')
                        if tool_name and tool_name in known_tools:
                            if tool_name not in tools_found:
                                tools_found.append(tool_name)
    
    return tools_found


def parse_tools_from_history_file(history_file: Path) -> List[str]:
    """Load and parse tool calls from a history JSON file.
    
    Args:
        history_file: Path to the history JSON file
        
    Returns:
        List of tool names found in the history
    """
    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        return parse_tools_from_history(history_data)
    except Exception as e:
        print(f"Warning: Failed to load history file {history_file}: {e}")
        return []


def parse_tools_from_response(response: str) -> List[str]:
    """Parse the agent response to extract tool names.
    
    DEPRECATED: This method uses fragile regex parsing.
    Use parse_tools_from_history() or parse_tools_from_history_file() instead.
    
    This fallback method looks for patterns like:
    - "Calling tool: trackerLight"
    - "tool_call: audibleWarning"
    - Or any known tool names in the response
    
    Args:
        response: Raw output from the agent
        
    Returns:
        List of tool names found in the response
    """
    tools_found = []
    
    # Known tools to look for
    known_tools = list(TOOL_ESCALATION_MAP.keys())
    
    # Pattern 1: Look for explicit tool calls
    tool_patterns = [
        r'(?:calling|call|invoking|execute|using)\s+(?:tool\s+)?(\w+)',
        r'tool[:\s]+(\w+)',
        r'(\w+)\s+\(tool\)',
    ]
    
    for pattern in tool_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            if match in known_tools:
                if match not in tools_found:
                    tools_found.append(match)
    
    # Pattern 2: Direct mention of tool names
    for tool in known_tools:
        if tool.lower() in response.lower():
            if tool not in tools_found:
                tools_found.append(tool)
    
    return tools_found


def calculate_escalation_level(tools_called: List[str]) -> int:
    """Calculate the escalation level based on tools called.
    
    The escalation level is the maximum level among all tools called.
    
    Args:
        tools_called: List of tool names
        
    Returns:
        Escalation level (0 if no tools called)
    """
    if not tools_called:
        return 0
    
    max_level = 0
    for tool in tools_called:
        level = TOOL_ESCALATION_MAP.get(tool, 0)
        if level > max_level:
            max_level = level
    
    return max_level


def extract_reasoning(response: str) -> str:
    """Extract reasoning from the agent response.
    
    Args:
        response: Raw output from the agent
        
    Returns:
        Extracted reasoning text
    """
    # Look for reasoning patterns
    patterns = [
        r'reasoning[:\s]*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'thought[:\s]*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'analysis[:\s]*(.+?)(?:\n\n|\n[A-Z]|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no explicit reasoning found, return first few sentences
    sentences = response.split('.')
    if sentences:
        return sentences[0].strip() + '.'
    
    return response[:200]  # Return first 200 chars as fallback


def load_chat_history(history_file: Path) -> List[Dict[str, str]]:
    """Load chat history from a JSON file.
    
    Args:
        history_file: Path to the history JSON file
        
    Returns:
        List of message dictionaries with 'content' and 'role' keys
    """
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    return data.get('messages', [])


def copy_image_to_workarea(image_path: str, datasets_dir: Path) -> Optional[str]:
    """Copy test image to llmfun workarea so the agent can access it.
    
    Args:
        image_path: Relative path to the image from datasets directory
        datasets_dir: The base datasets directory path
        
    Returns:
        The destination path where image was copied, or None if failed
    """
    if not image_path:
        return None

    try:
        # Ensure workarea exists
        LLMFUN_WORKAREA.mkdir(parents=True, exist_ok=True)
        
        # Source image path (relative to datasets directory)
        src_path = datasets_dir / image_path
        
        if not src_path.exists():
            print(f"Warning: Image not found: {src_path}")
            return None
        
        # Destination in workarea (use same filename)
        dst_path = LLMFUN_WORKAREA / Path(image_path).name

        if dst_path.exists():
            current = os.stat(str(dst_path)).st_mode
            os.chmod(str(dst_path), current | stat.S_IWUSR)
        
        # Copy the image
        shutil.copy2(src_path, dst_path)
        
        return str(dst_path)
        
    except Exception as e:
        print(f"Warning: Failed to copy image to workarea: {e}")
        return None


def evaluate_test_case(test_case, history_file: Optional[Path] = None, datasets_dir: Path = None) -> AgentResponse:
    """Evaluate a single test case.
    
    Args:
        test_case: TestCase object
        history_file: Optional path to chat history (not used, kept for API compatibility)
        datasets_dir: Base directory for datasets (to resolve image paths)
        
    Returns:
        AgentResponse from the evaluation
    """

    agent_cleanup(test_case.keep_history)

    # Build the prompt from test case
    prompt = build_prompt(test_case)

    # Handle image if present
    if test_case.image_path and datasets_dir:
        workarea_path = copy_image_to_workarea(test_case.image_path, datasets_dir)
        if workarea_path:
            prompt += f" Use loadImageApi to analyze at path {str(Path(workarea_path).name)}"
    
    # Call the agent and extract tool calls from the saved history
    return call_llmfun_agent_with_history(prompt)


def build_prompt(test_case) -> str:
    """Build the prompt for a test case.
    
    The prompt includes:
    - Instructions to use loadImageApi for image analysis
    - CNN classification probabilities
    
    Args:
        test_case: TestCase object
        
    Returns:
        Formatted prompt string
    """
    # Start with the test case prompt (should include loadImageApi instruction)
    prompt = test_case.prompt
    
    # Add CNN probabilities if available
    if test_case.cnn_probabilities:
        probs_str = "\nCNN Classification Probabilities:\n"
        for cls, prob in test_case.cnn_probabilities.items():
            probs_str += f"- {cls}: {prob:.2%}\n"
        prompt = prompt + probs_str
    
    # Add final instruction
    prompt = prompt + "\n\nBased on your analysis, determine the appropriate action(s) using your available tools."
    
    return prompt
