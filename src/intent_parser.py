"""
Intent parser for extracting user intent and routing to appropriate tools.

Uses LLM to understand user commands and map them to MCP tool calls.
"""

import json
import re
from typing import Any, Dict, Optional

from loguru import logger

from .lm_client import LMStudioClient
from .mcp_client import MCPClient


class IntentParser:
    """Intent parsing and tool routing."""

    def __init__(self, lm_client: LMStudioClient, mcp_client: MCPClient):
        """
        Initialize intent parser.

        Args:
            lm_client: LMStudio client for LLM inference
            mcp_client: MCP client for tool execution
        """
        self.lm_client = lm_client
        self.mcp_client = mcp_client

    async def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user intent from natural language input.

        Args:
            user_input: User's spoken/text command

        Returns:
            Dict with:
                - intent: Description of user intent
                - tool: Tool name to execute (or None)
                - parameters: Tool parameters
                - confidence: Confidence score (0.0-1.0)
        """
        if not user_input or not user_input.strip():
            return {
                "intent": "",
                "tool": None,
                "parameters": {},
                "confidence": 0.0,
            }

        logger.debug(f"Parsing intent from: '{user_input}'")

        # Get available tools from MCP
        available_tools = self.mcp_client.list_tools()

        # Build tool descriptions
        tool_descriptions = []
        for tool_name in available_tools:
            desc = self.mcp_client.get_tool_description(tool_name)
            tool_descriptions.append(f"- {tool_name}: {desc}")

        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"

        # Create system prompt
        system_prompt = f"""You are an intent parser for a voice AI agent.
Your job is to understand user commands and map them to the appropriate tool.

Available tools:
{tools_text}

Respond ONLY with valid JSON in this exact format:
{{
  "intent": "clear description of what the user wants",
  "tool": "tool_name or null if no tool needed",
  "parameters": {{
    "param_name": "param_value"
  }},
  "confidence": 0.9
}}

Rules:
- Use null for tool if the request is conversational or no tool matches
- Extract all relevant parameters from the user input
- Confidence should be 0.0-1.0 (1.0 = very confident)
- Always respond with valid JSON only"""

        # Query LLM
        response = await self.lm_client.generate(
            prompt=f"User said: '{user_input}'",
            system_prompt=system_prompt,
            temperature=0.2,  # Low temperature for structured output
            max_tokens=512,
        )

        # Parse JSON response
        parsed = self._extract_json(response)

        if parsed:
            # Validate tool exists
            if parsed.get("tool") and parsed["tool"] not in available_tools:
                logger.warning(f"LLM suggested unknown tool: {parsed['tool']}")
                parsed["tool"] = None
                parsed["confidence"] *= 0.5  # Reduce confidence

            logger.info(f"Intent parsed: {parsed['intent']} → {parsed.get('tool', 'conversation')}")

            return parsed
        else:
            # Fallback: treat as conversation
            logger.warning("Failed to parse intent JSON, treating as conversation")
            return {
                "intent": user_input,
                "tool": None,
                "parameters": {},
                "confidence": 0.3,
            }

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response.

        Args:
            text: Response text potentially containing JSON

        Returns:
            Parsed JSON dict or None
        """
        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", text, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.debug(f"JSON text: {json_match.group()}")
                return None

        return None

    async def execute_intent(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the parsed intent.

        Args:
            intent_data: Parsed intent from parse_intent()

        Returns:
            Execution result with 'success', 'result', and 'error' keys
        """
        tool_name = intent_data.get("tool")

        if not tool_name:
            # No tool to execute, just return the intent
            return {
                "success": True,
                "result": intent_data.get("intent", ""),
                "is_conversation": True,
            }

        # Validate parameters
        parameters = intent_data.get("parameters", {})
        is_valid, error = self.mcp_client.validate_parameters(tool_name, parameters)

        if not is_valid:
            logger.error(f"Parameter validation failed: {error}")
            return {
                "success": False,
                "error": f"Invalid parameters: {error}",
                "is_conversation": False,
            }

        # Execute tool
        logger.info(f"Executing tool: {tool_name}")

        result = await self.mcp_client.execute_tool(tool_name, parameters)

        return {
            "success": result.get("success", False),
            "result": result.get("result", result.get("data", "")),
            "error": result.get("error"),
            "is_conversation": False,
            "tool": tool_name,
        }

    async def parse_and_execute(self, user_input: str) -> Dict[str, Any]:
        """
        Parse intent and execute in one call.

        Args:
            user_input: User's command

        Returns:
            Execution result
        """
        intent_data = await self.parse_intent(user_input)
        result = await self.execute_intent(intent_data)

        return {
            **result,
            "intent": intent_data.get("intent"),
            "confidence": intent_data.get("confidence", 0.0),
        }

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        return tool_name in self.mcp_client.list_tools()

    def suggest_tools(self, query: str) -> list[str]:
        """
        Suggest relevant tools based on a query.

        Args:
            query: Search query

        Returns:
            List of relevant tool names
        """
        available_tools = self.mcp_client.list_tools()

        # Simple keyword matching (can be improved with embeddings)
        query_lower = query.lower()
        suggestions = []

        for tool_name in available_tools:
            desc = self.mcp_client.get_tool_description(tool_name)
            if query_lower in tool_name.lower() or query_lower in desc.lower():
                suggestions.append(tool_name)

        return suggestions

    async def clarify_intent(
        self,
        user_input: str,
        previous_context: Optional[Dict] = None,
    ) -> str:
        """
        Generate clarifying question if intent is ambiguous.

        Args:
            user_input: User's command
            previous_context: Previous conversation context

        Returns:
            Clarifying question or empty string if clear
        """
        intent_data = await self.parse_intent(user_input)

        # If confidence is low, ask for clarification
        if intent_data["confidence"] < 0.6:
            # Generate clarifying question
            prompt = f"The user said: '{user_input}'. Their intent seems to be: '{intent_data['intent']}'. Generate a brief clarifying question."

            question = await self.lm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=100,
            )

            return question.strip()

        return ""
