#!/usr/bin/env python3
"""
Startup script for the restructured GitLabDev application.
This ensures the correct Python path is set before starting the application.
"""
import sys
import os
from pathlib import Path

# Add the app_restructured directory to Python path
project_root = Path(__file__).parent
app_restructured_path = project_root / "app_restructured"

if str(app_restructured_path) not in sys.path:
    sys.path.insert(0, str(app_restructured_path))

# Now we can import and run the application
if __name__ == "__main__":
    import uvicorn
    
    # Start the application
    uvicorn.run(
        "interfaces.api.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        reload_dirs=[str(app_restructured_path)]
    )