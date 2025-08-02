#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for LLM Fuzz Testing

This server provides:
- AI API tools for fuzzing and testing (Cerebras, Anthropic)
- Documentation resources for AI APIs
- Comprehensive parameter support for all AI provider functions
- Multiple transport modes: stdio, SSE, HTTP

Usage:
  python src/main.py                              # Default: stdio mode
  python src/main.py --transport stdio            # Stdio mode (default)
  python src/main.py --transport sse --port 8000  # SSE mode on port 8000
  python src/main.py --transport http --port 8001 # HTTP mode on port 8001

Test with: 
  mcp dev src/main.py
"""

import sys
import argparse
import asyncio
import logging
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from basic_mcp_lab.cerebras import CerebrasClient
from basic_mcp_lab.anthropic import AnthropicClient
from basic_mcp_lab.base_client import BaseClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("ai-fuzz-tests")

# Global clients
cerebras_client = None
anthropic_client = None


def get_cerebras_client() -> CerebrasClient:
    """Get or create Cerebras client instance."""
    global cerebras_client
    if cerebras_client is None:
        cerebras_client = CerebrasClient()
    return cerebras_client


def get_anthropic_client() -> AnthropicClient:
    """Get or create Anthropic client instance."""
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = AnthropicClient()
    return anthropic_client


def get_client_by_provider(provider: str) -> BaseClient:
    """Get client by provider name."""
    if provider.lower() == "cerebras":
        return get_cerebras_client()
    elif provider.lower() == "anthropic":
        return get_anthropic_client()
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Resource templates for dynamic documentation
@mcp.resource("docs://{provider}/sdk/{path}")
def get_sdk_documentation(provider: str, path: str) -> str:
    """
    Get documentation for AI provider SDK modules, classes, functions, etc.
    
    Use '/' or empty path to list available SDK components.
    For modules/classes, shows documentation plus available sub-components.
    
    Args:
        provider: AI provider name ("cerebras" or "anthropic")
        path: Dot-separated path to SDK component 
              Examples:
              - '/' or '' : List all available SDK components
              - 'Cerebras' : Main client class documentation + methods
              - 'AsyncCerebras' : Async client class documentation + methods  
              - 'chat.completions' : Chat completions module documentation
    """
    from urllib.parse import unquote
    
    # URL decode the path
    path = unquote(path)
    
    if provider.lower() == "cerebras":
        return _get_cerebras_sdk_documentation(path)
    elif provider.lower() == "anthropic":
        return _get_anthropic_sdk_documentation(path)
    else:
        return f"# Error\n\nUnsupported provider: {provider}"


def _get_cerebras_sdk_documentation(path: str) -> str:
    """Get documentation for Cerebras SDK components."""
    try:
        import cerebras.cloud.sdk as sdk
        import inspect
        
        # Handle root path - list available resources
        if path == "" or path == "/":
            available = []
            for name in dir(sdk):
                if not name.startswith('_'):
                    available.append(f"docs://cerebras/sdk/{name}")
            
            return f"""# Cerebras SDK Resources

Available SDK components:

{chr(10).join(f"- {resource}" for resource in available)}

## Root SDK Documentation

{sdk.__doc__ or 'No documentation available'}
"""
        
        # Navigate to the requested component
        current = sdk
        parts = path.split('.')
        current_path = "cerebras.cloud.sdk"
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
                current_path += f".{part}"
            else:
                return f"# Error\n\nComponent '{path}' not found in cerebras.cloud.sdk"
        
        # If this is a module/class with attributes, list them
        if hasattr(current, '__dict__') or inspect.ismodule(current) or inspect.isclass(current):
            # List available sub-resources
            available_attrs = []
            for name in dir(current):
                if not name.startswith('_'):
                    full_path = f"{path}.{name}" if path else name
                    available_attrs.append(f"docs://cerebras/sdk/{full_path}")
            
            if available_attrs:
                attr_list = f"""

## Available sub-components:

{chr(10).join(f"- {resource}" for resource in available_attrs[:20])}  # Limit to first 20
{"... (truncated)" if len(available_attrs) > 20 else ""}
"""
            else:
                attr_list = ""
        else:
            attr_list = ""
        
        # Get documentation
        doc = getattr(current, '__doc__', None)
        if doc:
            return f"# {path}\n\n{doc}{attr_list}"
        else:
            # If no __doc__, show type and basic info
            obj_type = type(current).__name__
            
            # For classes, try to get more info
            if inspect.isclass(current):
                methods = [name for name in dir(current) if not name.startswith('_') and callable(getattr(current, name))]
                if methods:
                    method_info = f"\n\nMethods: {', '.join(methods[:10])}"
                    if len(methods) > 10:
                        method_info += " ... (truncated)"
                else:
                    method_info = ""
                return f"# {path}\n\nType: {obj_type} (class){method_info}{attr_list}"
            
            return f"# {path}\n\nType: {obj_type}\n\nNo documentation available for this component.{attr_list}"
            
    except ImportError:
        return f"# Error\n\nCould not import cerebras.cloud.sdk"
    except Exception as e:
        return f"# Error\n\nError accessing '{path}': {str(e)}"


def _get_anthropic_sdk_documentation(path: str) -> str:
    """Get documentation for Anthropic SDK components."""
    try:
        import anthropic
        import inspect
        
        # Handle root path - list available resources
        if path == "" or path == "/":
            available = []
            for name in dir(anthropic):
                if not name.startswith('_'):
                    available.append(f"docs://anthropic/sdk/{name}")
            
            return f"""# Anthropic SDK Resources

Available SDK components:

{chr(10).join(f"- {resource}" for resource in available)}

## Root SDK Documentation

{anthropic.__doc__ or 'No documentation available'}
"""
        
        # Navigate to the requested component
        current = anthropic
        parts = path.split('.')
        current_path = "anthropic"
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
                current_path += f".{part}"
            else:
                return f"# Error\n\nComponent '{path}' not found in anthropic"
        
        # If this is a module/class with attributes, list them
        if hasattr(current, '__dict__') or inspect.ismodule(current) or inspect.isclass(current):
            # List available sub-resources
            available_attrs = []
            for name in dir(current):
                if not name.startswith('_'):
                    full_path = f"{path}.{name}" if path else name
                    available_attrs.append(f"docs://anthropic/sdk/{full_path}")
            
            if available_attrs:
                attr_list = f"""

## Available sub-components:

{chr(10).join(f"- {resource}" for resource in available_attrs[:20])}  # Limit to first 20
{"... (truncated)" if len(available_attrs) > 20 else ""}
"""
            else:
                attr_list = ""
        else:
            attr_list = ""
        
        # Get documentation
        doc = getattr(current, '__doc__', None)
        if doc:
            return f"# {path}\n\n{doc}{attr_list}"
        else:
            # If no __doc__, show type and basic info
            obj_type = type(current).__name__
            
            # For classes, try to get more info
            if inspect.isclass(current):
                methods = [name for name in dir(current) if not name.startswith('_') and callable(getattr(current, name))]
                if methods:
                    method_info = f"\n\nMethods: {', '.join(methods[:10])}"
                    if len(methods) > 10:
                        method_info += " ... (truncated)"
                else:
                    method_info = ""
                return f"# {path}\n\nType: {obj_type} (class){method_info}{attr_list}"
            
            return f"# {path}\n\nType: {obj_type}\n\nNo documentation available for this component.{attr_list}"
            
    except ImportError:
        return f"# Error\n\nCould not import anthropic"
    except Exception as e:
        return f"# Error\n\nError accessing '{path}': {str(e)}"


# Tools for AI API functions
@mcp.tool()
def chat_completion(provider: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a chat completion using specified AI provider.
    
    Args:
        provider: AI provider name ("cerebras" or "anthropic")
        kwargs: Provider-specific parameters (do NOT nest under "kwargs")
    
    Pass parameters directly by name - do NOT pass a "kwargs" parameter.
    When stream=True, automatically returns all chunks as an array.

    Example Arguments:
        provider = "cerebras"
        kwargs = {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "llama3.1-8b",
            "stream": True
        }
        
    Returns:
        Chat completion response (single object or array if streaming)
    """
    try:
        client = get_client_by_provider(provider)
        response = client.chat_completion(**kwargs)
        
        # Handle both streaming (array) and non-streaming (single object) responses
        if isinstance(response, list):
            # Streaming response - already converted to array of dicts
            return {
                "success": True,
                "provider": provider,
                "streaming": True,
                "chunks": response,
                "total_chunks": len(response)
            }
        else:
            # Non-streaming response
            if hasattr(response, 'model_dump'):
                response_data = response.model_dump()
            elif hasattr(response, 'to_dict'):
                response_data = response.to_dict()
            else:
                response_data = {"response": str(response)}
                
            return {
                "success": True,
                "provider": provider,
                "streaming": False,
                **response_data
            }
            
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        return {"error": str(e), "success": False}



@mcp.tool()
def get_models(provider: str) -> Dict[str, Any]:
    """
    Get list of available models for specified AI provider.
    
    Args:
        provider: AI provider name ("cerebras" or "anthropic")
    
    Returns:
        List of available models
    """
    try:
        client = get_client_by_provider(provider)
        models = client.get_available_models()
        
        return {
            "success": True,
            "provider": provider,
            "models": models,
            "total_models": len(models)
        }
        
    except Exception as e:
        logger.error(f"Getting models failed: {str(e)}")
        return {"error": str(e), "success": False}



@mcp.tool()
def get_client_info(provider: str) -> Dict[str, Any]:
    """
    Get information about the specified AI client configuration.
    
    Args:
        provider: AI provider name ("cerebras" or "anthropic")
    
    Returns:
        Client configuration and status information
    """
    try:
        client = get_client_by_provider(provider)
        info = client.get_client_info()
        return {"success": True, **info}
        
    except Exception as e:
        logger.error(f"Getting client info failed: {str(e)}")
        return {"error": str(e), "success": False}




@mcp.tool()
def list_providers() -> Dict[str, Any]:
    """
    List all available AI providers.
    
    Returns:
        List of available providers with their status
    """
    providers = []
    
    # Check Cerebras
    try:
        cerebras_client = get_cerebras_client()
        cerebras_info = cerebras_client.get_client_info()
        providers.append({
            "name": "cerebras",
            "available": True,
            "models_count": len(cerebras_info.get("available_models", [])),
            "has_api_key": cerebras_info.get("config", {}).get("has_api_key", False)
        })
    except Exception as e:
        providers.append({
            "name": "cerebras",
            "available": False,
            "error": str(e)
        })
    
    # Check Anthropic
    try:
        anthropic_client = get_anthropic_client()
        anthropic_info = anthropic_client.get_client_info()
        providers.append({
            "name": "anthropic",
            "available": True,
            "models_count": len(anthropic_info.get("available_models", [])),
            "has_api_key": anthropic_info.get("config", {}).get("has_api_key", False)
        })
    except Exception as e:
        providers.append({
            "name": "anthropic",
            "available": False,
            "error": str(e)
        })
    
    return {
        "success": True,
        "providers": providers,
        "total_providers": len(providers)
    }


def parse_arguments():
    """Parse command line arguments for transport mode selection."""
    parser = argparse.ArgumentParser(
        description="AI LLM Fuzz Testing MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                              # Default: stdio mode
  python src/main.py --transport stdio            # Stdio mode (explicit)
  python src/main.py --transport sse --port 8000  # SSE mode on port 8000
  python src/main.py --transport http --port 8001 # HTTP mode on port 8001
  python src/main.py --host 0.0.0.0 --port 9000  # Custom host and port

Transport Modes:
  stdio - Standard input/output (default, for local MCP clients)
  sse   - Server-Sent Events over HTTP (proper MCP over SSE)
  http  - HTTP with JSON-RPC (proper MCP over HTTP)
        """
    )
    
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport mode for the MCP server (default: stdio)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port number for HTTP/SSE transport modes (default: 8000)"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host address for HTTP/SSE transport modes (default: localhost)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--cors",
        action="store_true",
        help="Enable CORS headers for HTTP/SSE modes"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Server timeout in seconds (for testing, default: no timeout)"
    )
    
    return parser.parse_args()




async def run_stdio_server():
    """Run MCP server in stdio mode using FastMCP's built-in stdio transport."""
    logger.info("Starting MCP server in stdio mode...")
    logger.info("Server ready for MCP client connections via stdin/stdout")
    await mcp.run_stdio_async()


async def run_sse_server(host: str, port: int, enable_cors: bool = False, timeout: Optional[int] = None):
    """Run MCP server in Server-Sent Events mode using FastMCP's built-in SSE transport."""
    try:
        logger.info(f"Starting MCP server in SSE mode on {host}:{port}...")
        logger.info(f"SSE endpoint available at: http://{host}:{port}/sse")
        logger.info(f"Use with: mcp dev src.main.py")
        
        # Configure server settings
        mcp.settings.host = host
        mcp.settings.port = port
        
        if timeout:
            logger.info(f"Server will timeout after {timeout} seconds")
            await asyncio.wait_for(mcp.run_sse_async(), timeout=timeout)
        else:
            await mcp.run_sse_async()
            
    except asyncio.TimeoutError:
        logger.info(f"Server timed out after {timeout} seconds")
    except ImportError as e:
        logger.error(f"SSE mode requires additional dependencies: {e}")
        logger.error("Install with: pip install uvicorn starlette")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start SSE server: {e}")
        sys.exit(1)


async def run_http_server(host: str, port: int, enable_cors: bool = False, timeout: Optional[int] = None):
    """Run MCP server in HTTP mode using FastMCP's built-in HTTP transport."""
    try:
        logger.info(f"Starting MCP server in HTTP mode on {host}:{port}...")
        logger.info(f"HTTP server ready at: http://{host}:{port}")
        logger.info(f"MCP endpoint: http://{host}:{port}/mcp")
        logger.info(f"Use with: mcp dev src.main.py")
        
        # Configure server settings
        mcp.settings.host = host
        mcp.settings.port = port
        
        if timeout:
            logger.info(f"Server will timeout after {timeout} seconds")
            await asyncio.wait_for(mcp.run_streamable_http_async(), timeout=timeout)
        else:
            await mcp.run_streamable_http_async()
            
    except asyncio.TimeoutError:
        logger.info(f"Server timed out after {timeout} seconds")
    except ImportError as e:
        logger.error(f"HTTP mode requires additional dependencies: {e}")
        logger.error("Install with: pip install uvicorn starlette")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start HTTP server: {e}")
        sys.exit(1)




if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        logger.info(f"AI Fuzz Tests MCP Server starting...")
        logger.info(f"Transport: {args.transport}")
        
        if args.transport == "stdio":
            asyncio.run(run_stdio_server())
        elif args.transport in ["sse", "http"]:
            logger.info(f"Host: {args.host}, Port: {args.port}, CORS: {args.cors}")
            
            if args.transport == "sse":
                asyncio.run(run_sse_server(
                    args.host, 
                    args.port, 
                    args.cors, 
                    args.timeout
                ))
            else:
                asyncio.run(run_http_server(
                    args.host, 
                    args.port, 
                    args.cors, 
                    args.timeout
                ))
        else:
            logger.error(f"Unknown transport mode: {args.transport}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
