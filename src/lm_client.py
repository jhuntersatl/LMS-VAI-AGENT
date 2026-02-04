"""
LMStudio API client for LLM inference.

Handles communication with local LMStudio server for intent parsing and response generation.
"""

import asyncio
from typing import AsyncIterator, Dict, List, Optional

import httpx
from loguru import logger

from .config import LMStudioConfig


class LMStudioClient:
    """Client for LMStudio HTTP API."""

    def __init__(self, config: LMStudioConfig):
        """Initialize LMStudio client."""
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stream: Enable streaming response

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                if stream:
                    return await self._generate_stream(client, payload)
                else:
                    return await self._generate_single(client, payload)

        except httpx.TimeoutException:
            logger.error(f"LMStudio request timed out after {self.config.timeout}s")
            raise
        except httpx.HTTPError as e:
            logger.error(f"LMStudio HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"LMStudio error: {e}")
            raise

    async def _generate_single(
        self, client: httpx.AsyncClient, payload: Dict
    ) -> str:
        """Generate non-streaming response."""
        url = f"{self.base_url}/v1/chat/completions"

        logger.debug(f"Sending request to LMStudio: {url}")
        response = await client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        logger.debug(f"LMStudio response: {content[:100]}...")
        return content.strip()

    async def _generate_stream(
        self, client: httpx.AsyncClient, payload: Dict
    ) -> str:
        """Generate streaming response."""
        url = f"{self.base_url}/v1/chat/completions"

        logger.debug(f"Sending streaming request to LMStudio: {url}")

        full_response = []
        async with client.stream("POST", url, json=payload, headers=self.headers) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]  # Remove "data: " prefix
                    if chunk == "[DONE]":
                        break

                    try:
                        import json
                        data = json.loads(chunk)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response.append(content)
                    except json.JSONDecodeError:
                        continue

        result = "".join(full_response).strip()
        logger.debug(f"LMStudio streaming complete: {len(result)} chars")
        return result

    async def parse_intent(self, user_input: str) -> Dict[str, any]:
        """
        Parse user intent from input text.

        Args:
            user_input: User's spoken/text input

        Returns:
            Dict with 'intent', 'tool', and 'parameters'
        """
        system_prompt = """You are an intent parser for a voice AI agent.
Extract the intent, required tool, and parameters from user input.
Respond ONLY with valid JSON in this format:
{
  "intent": "description of user intent",
  "tool": "tool_name or null",
  "parameters": {
    "key": "value"
  }
}
Available tools: file_search, web_browse, calculator, weather, calendar"""

        prompt = f"Parse this user input: '{user_input}'"

        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for structured output
                max_tokens=256,
            )

            # Extract JSON from response
            import json
            import re

            # Try to find JSON in response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: treat as plain response
                return {
                    "intent": user_input,
                    "tool": None,
                    "parameters": {},
                }

        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            return {
                "intent": user_input,
                "tool": None,
                "parameters": {},
            }

    async def health_check(self) -> bool:
        """Check if LMStudio is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                url = f"{self.base_url}/v1/models"
                response = await client.get(url, headers=self.headers)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"LMStudio health check failed: {e}")
            return False
