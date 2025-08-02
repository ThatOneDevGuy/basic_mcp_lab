"""
Base client implementation for LLM fuzz testing.

This module provides a common interface and configuration for AI model clients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any


@dataclass
class BaseConfig:
    """Base configuration class for AI clients."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None


class BaseClient(ABC):
    """
    Abstract base class for AI model clients.
    
    Provides a common interface for different AI providers while allowing
    provider-specific implementations.
    """
    
    def __init__(self, config: Optional[BaseConfig] = None):
        """Initialize the client with configuration."""
        self.config = config or BaseConfig()
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model identifiers
        """
        pass
    
    @abstractmethod
    def chat_completion(self, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Create a chat completion.
        
        Args:
            **kwargs: Provider-specific parameters
            
        Returns:
            Chat completion response (single object) or array of streaming chunks
        """
        pass
    
    @abstractmethod
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the client configuration.
        
        Returns:
            Client configuration information
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the AI provider."""
        pass