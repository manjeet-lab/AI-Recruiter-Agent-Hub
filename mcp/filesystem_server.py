"""
MCP Filesystem Server
=====================
Exposes secure filesystem tools (read, write, list files) using the Model Context Protocol (MCP).
Exposes only the uploads/ and reports/ directories under the project root.
Prevents directory traversal and unauthorized file access.
"""

import os
import sys
import base64
import json
from mcp.server.fastmcp import FastMCP

# Add the workspace root to Python path so we can import the PDF parser
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.append(WORKSPACE_DIR)

from tools.pdf_parser import extract_text_from_pdf

# Setup allowed/secure directories
UPLOADS_DIR = os.path.abspath(os.path.join(WORKSPACE_DIR, "uploads"))
REPORTS_DIR = os.path.abspath(os.path.join(WORKSPACE_DIR, "reports"))

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize FastMCP Server
mcp = FastMCP("AI Recruiter Filesystem Server")

def secure_resolve(directory_name: str, filename: str) -> str:
    """
    Resolves the target path inside the allowed directory.
    Strips path segments using os.path.basename to prevent directory traversal attacks.
    Ensures the path remains within the bounds of the specific directory.
    """
    if directory_name == "uploads":
        base_dir = UPLOADS_DIR
    elif directory_name == "reports":
        base_dir = REPORTS_DIR
    else:
        raise ValueError(f"Unauthorized directory access: {directory_name}")

    # Strip any directory components from the input filename to prevent traversal
    clean_filename = os.path.basename(filename)
    target_path = os.path.abspath(os.path.join(base_dir, clean_filename))

    # Explicit check to verify path containment
    if not target_path.startswith(base_dir + os.sep) and target_path != base_dir:
        raise ValueError("Security Violation: Directory traversal or unauthorized access detected!")

    return target_path

@mcp.tool()
def save_resume(filename: str, base64_content: str) -> str:
    """
    Saves a resume file (e.g. PDF) to the uploads/ directory securely via base64 encoding.
    
    Args:
        filename (str): The name of the file to save.
        base64_content (str): Base64 encoded string of the file bytes.
    """
    try:
        target_path = secure_resolve("uploads", filename)
        file_bytes = base64.b64decode(base64_content)
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        return f"Successfully saved resume '{filename}' to uploads/ directory."
    except Exception as e:
        return f"Error saving resume: {str(e)}"

@mcp.tool()
def read_resume(filename: str) -> str:
    """
    Reads a resume from the uploads/ directory securely.
    If it is a PDF file, parses the document and returns extracted text content.
    If it is a text file, returns its text content.
    
    Args:
        filename (str): The name of the file to read.
    """
    try:
        target_path = secure_resolve("uploads", filename)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Resume file '{filename}' was not found in uploads/.")

        if filename.lower().endswith(".pdf"):
            with open(target_path, "rb") as f:
                file_bytes = f.read()
            # Extract and normalize text using PDF parser tool
            return extract_text_from_pdf(file_bytes, filename)
        else:
            with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"Error reading resume: {str(e)}"

@mcp.tool()
def save_report(filename: str, content: str) -> str:
    """
    Saves a generated candidate evaluation report (JSON format) to the reports/ directory.
    
    Args:
        filename (str): The name of the report file (will be forced to .json).
        content (str): The string content (e.g., aggregated JSON evaluation) of the report.
    """
    try:
        if not filename.lower().endswith(".json"):
            filename += ".json"
        target_path = secure_resolve("reports", filename)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved recruiter report '{filename}' to reports/ directory."
    except Exception as e:
        return f"Error saving report: {str(e)}"

@mcp.tool()
def read_report(filename: str) -> str:
    """
    Reads an evaluation report from the reports/ directory securely.
    
    Args:
        filename (str): Name of the JSON report to read.
    """
    try:
        target_path = secure_resolve("reports", filename)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Report '{filename}' was not found in reports/.")
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading report: {str(e)}"

@mcp.tool()
def list_files(directory_name: str) -> str:
    """
    Lists all files in the designated directory ('uploads' or 'reports').
    Returns list of filenames as a JSON string.
    
    Args:
        directory_name (str): Either 'uploads' or 'reports'.
    """
    try:
        if directory_name not in ["uploads", "reports"]:
            raise ValueError("Only 'uploads' and 'reports' directories are accessible.")
        
        target_dir = UPLOADS_DIR if directory_name == "uploads" else REPORTS_DIR
        files = os.listdir(target_dir)
        # Filter files to exclude hidden ones
        visible_files = [f for f in files if not f.startswith(".")]
        return json.dumps(visible_files)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    mcp.run()
