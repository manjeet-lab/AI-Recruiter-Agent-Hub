"""
MCP Client Module
=================
Establishes connection to the Filesystem MCP Server via stdio transport and exposes
synchronous helpers to save/read resumes and reports.
"""

import os
import sys
import asyncio
import base64
import json
from typing import List, Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Define workspace directory
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Stdio server execution parameters
server_params = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(WORKSPACE_DIR, "mcp", "filesystem_server.py")],
    env=dict(os.environ, PYTHONPATH=WORKSPACE_DIR),
    cwd=WORKSPACE_DIR
)

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Launches the Filesystem MCP Server, initializes a session,
    calls the specified tool, and closes the connection.
    """
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            
            if getattr(result, "isError", False):
                raise RuntimeError(f"MCP Server Error during '{tool_name}': {result.content}")
                
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return ""

def _run_sync(coro):
    """
    Helper to run async coroutines in both running and idle event loops.
    Specially handles Streamlit's running event loop using nest_asyncio.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)

# --- Exposing Synchronous API wrapper functions for Client Use ---

def mcp_save_resume(filename: str, file_bytes: bytes) -> str:
    """Saves raw resume bytes to the uploads/ folder using the MCP server."""
    base64_str = base64.b64encode(file_bytes).decode("utf-8")
    return _run_sync(call_mcp_tool("save_resume", {"filename": filename, "base64_content": base64_str}))

def mcp_read_resume(filename: str) -> str:
    """Reads and parses a resume file through the MCP server (handles PDF parsing)."""
    return _run_sync(call_mcp_tool("read_resume", {"filename": filename}))

def mcp_save_report(filename: str, content: str) -> str:
    """Saves recruiter reports (JSON string) to the reports/ folder using the MCP server."""
    return _run_sync(call_mcp_tool("save_report", {"filename": filename, "content": content}))

def mcp_read_report(filename: str) -> str:
    """Reads a report from the reports/ folder through the MCP server."""
    return _run_sync(call_mcp_tool("read_report", {"filename": filename}))

def mcp_list_files(directory_name: str) -> List[str]:
    """Lists files in either uploads/ or reports/ folder through the MCP server."""
    res_str = _run_sync(call_mcp_tool("list_files", {"directory_name": directory_name}))
    try:
        return json.loads(res_str)
    except Exception:
        return []
