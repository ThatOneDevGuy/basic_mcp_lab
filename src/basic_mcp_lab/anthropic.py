"""
Anthropic client implementation for LLM fuzz testing.

This module provides a comprehensive wrapper around the Anthropic SDK
with support for all API parameters and configurations.
"""

import os
import logging
from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass

from anthropic import Anthropic

from .base_client import BaseClient, BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class AnthropicConfig(BaseConfig):
    """Configuration class for Anthropic client."""
    pass


class AnthropicClient(BaseClient):
    """
    Comprehensive Anthropic client for LLM inference and testing.
    
    Supports all Anthropic API parameters and configurations including:
    - Message completions (streaming and non-streaming)
    - Tool/function calling
    - All model parameters (temperature, max_tokens, etc.)
    - System prompts
    - Multiple models including Claude 4 and Claude 3.5 families
    """
    
    
    def __init__(self, config: Optional[AnthropicConfig] = None):
        """
        Initialize Anthropic client.
        
        Args:
            config: Configuration object. If None, uses environment variables.
        """
        super().__init__(config or AnthropicConfig())
        
        # Initialize client with configuration
        client_kwargs = {}
        
        # Use provided API key or fall back to environment variable
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            client_kwargs["api_key"] = api_key
            
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
            
        if self.config.timeout:
            client_kwargs["timeout"] = self.config.timeout
            
        if self.config.max_retries:
            client_kwargs["max_retries"] = self.config.max_retries
            
        self.client = Anthropic(**client_kwargs)
        
    def get_available_models(self) -> List[str]:
        """Get list of available models from Anthropic API."""
        # Fetch models from the API
        models_response = self.client.models.list()
        
        # Extract model IDs from the response
        if hasattr(models_response, 'data'):
            model_ids = [model.id for model in models_response.data if hasattr(model, 'id')]
            return sorted(model_ids)
        else:
            logger.warning("Unexpected models response format")
            return []

    
    def chat_completion(self, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Create a message completion with pass-through parameter support.
        
        This is a pass-through interface that forwards all arguments directly
        to the Anthropic API. When stream=True, automatically collects all chunks
        into an array for easier handling in MCP tools.
        
        Args:
            **kwargs: All parameters passed directly to the Anthropic API
            
        Returns:
            Message completion response (single object) or array of streaming chunks
        """
        try:
            response = self.client.messages.create(**kwargs)
            
            # If streaming, collect all chunks into an array
            if kwargs.get('stream', False):
                chunks = []
                for chunk in response:
                    if hasattr(chunk, 'model_dump'):
                        chunks.append(chunk.model_dump())
                    elif hasattr(chunk, 'to_dict'):
                        chunks.append(chunk.to_dict())
                    else:
                        chunks.append({"chunk": str(chunk)})
                return chunks
            
            return response
        except Exception as e:
            logger.error(f"Message completion failed: {str(e)}")
            raise
    
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the client configuration.
        
        Returns:
            Client configuration information
        """
        # Get SDK version
        try:
            import anthropic
            sdk_version = getattr(anthropic, '__version__', 'unknown')
        except (ImportError, AttributeError):
            sdk_version = 'unknown'
        
        return {
            "provider": self.provider_name,
            "sdk_version": sdk_version,
            "config": {
                "has_api_key": bool(self.client.api_key),
                "base_url": str(self.client.base_url),
                "timeout": str(self.client.timeout),
                "max_retries": self.client.max_retries
            },
        }
    
    @property
    def provider_name(self) -> str:
        """Get the name of the AI provider."""
        return "anthropic"