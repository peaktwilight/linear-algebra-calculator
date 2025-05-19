#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Utilities
"""

import sys
from io import StringIO

class StreamOutput:
    """Capture print output to display in Streamlit."""
    def __init__(self):
        self.buffer = StringIO()
        self.old_stdout = sys.stdout
    
    def __enter__(self):
        sys.stdout = self.buffer
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.old_stdout
    
    def get_output(self):
        return self.buffer.getvalue() 