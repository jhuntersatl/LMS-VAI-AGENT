"""
MCP (Model Context Protocol) client for tool execution.

Provides interface to MCP server for executing actions and retrieving information.
"""

import asyncio
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from .config import MCPConfig


class MCPClient:
    """Client for MCP server tool execution."""

    def __init__(self, config: MCPConfig):
        """Initialize MCP client."""
        self.config = config
        self.base_url = config.server_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        self._available_tools: Optional[Dict[str, Dict]] = None

    async def initialize(self) -> None:
        """Initialize client and discover available tools."""
        logger.info("Initializing MCP client...")

        # Discover available tools
        self._available_tools = await self.discover_tools()

        logger.info(f"✓ MCP client initialized ({len(self._available_tools)} tools available)")

    async def discover_tools(self) -> Dict[str, Dict]:
        """
        Discover available tools from MCP server.

        Returns:
            Dict mapping tool names to their schemas
        """
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                url = f"{self.base_url}/v1/tools"
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()

                data = response.json()
                tools = {tool["name"]: tool for tool in data.get("tools", [])}

                logger.debug(f"Discovered {len(tools)} tools: {list(tools.keys())}")

                return tools

        except httpx.HTTPError as e:
            logger.error(f"Failed to discover MCP tools: {e}")
            return {}

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters (optional)

        Returns:
            Tool execution result
        """
        if self._available_tools is None:
            await self.initialize()

        if tool_name not in self._available_tools:
            logger.warning(f"Tool '{tool_name}' not found in available tools")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not available",
            }

        parameters = parameters or {}

        logger.debug(f"Executing tool: {tool_name} with params: {parameters}")

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                url = f"{self.base_url}/v1/tools/{tool_name}/execute"

                payload = {
                    "tool": tool_name,
                    "parameters": parameters,
                }

                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()

                result = response.json()

                logger.info(f"✓ Tool '{tool_name}' executed successfully")

                return result

        except httpx.TimeoutException:
            logger.error(f"Tool '{tool_name}' execution timed out")
            return {
                "success": False,
                "error": "Execution timeout",
            }

        except httpx.HTTPError as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

        except Exception as e:
            logger.error(f"Unexpected error executing tool '{tool_name}': {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def execute_tool_batch(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in batch.

        Args:
            tool_calls: List of dicts with 'tool' and 'parameters' keys

        Returns:
            List of execution results
        """
        tasks = [
            self.execute_tool(call["tool"], call.get("parameters"))
            for call in tool_calls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "success": False,
                    "error": str(result),
                    "tool": tool_calls[i]["tool"],
                })
            else:
                formatted_results.append(result)

        return formatted_results

    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema or None if not found
        """
        if self._available_tools is None:
            return None

        return self._available_tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """
        List available tool names.

        Returns:
            List of tool names
        """
        if self._available_tools is None:
            return []

        return list(self._available_tools.keys())

    def get_tools_by_category(self, category: str) -> List[str]:
        """
        Get tools filtered by category.

        Args:
            category: Tool category (e.g., 'file', 'web', 'system')

        Returns:
            List of matching tool names
        """
        if self._available_tools is None:
            return []

        return [
            name
            for name, schema in self._available_tools.items()
            if schema.get("category") == category
        ]

    def validate_parameters(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against tool schema.

        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate

        Returns:
            (is_valid, error_message)
        """
        schema = self.get_tool_schema(tool_name)

        if schema is None:
            return False, f"Tool '{tool_name}' not found"

        required_params = schema.get("required_parameters", [])
        optional_params = schema.get("optional_parameters", [])
        all_params = required_params + optional_params

        # Check required parameters
        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

        # Check for unknown parameters
        for param in parameters:
            if param not in all_params:
                return False, f"Unknown parameter: {param}"

        return True, None

    async def health_check(self) -> bool:
        """Check if MCP server is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                url = f"{self.base_url}/health"
                response = await client.get(url, headers=self.headers)
                return response.status_code == 200

        except Exception as e:
            logger.warning(f"MCP health check failed: {e}")
            return False

    def get_tool_description(self, tool_name: str) -> str:
        """
        Get human-readable description of a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool description
        """
        schema = self.get_tool_schema(tool_name)

        if schema is None:
            return f"Tool '{tool_name}' not found"

        description = schema.get("description", "No description available")
        params = schema.get("required_parameters", [])

        if params:
            param_str = ", ".join(params)
            return f"{description} (requires: {param_str})"

        return description

    async def close(self) -> None:
        """Clean up resources."""
        self._available_tools = None
        logger.info("MCP client closed")
