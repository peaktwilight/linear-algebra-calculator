#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the UI changes with panel display using direct output redirection
"""
from ui_rich import LinearAlgebraRichUI
from rich.console import Console
from rich.panel import Panel
import numpy as np

class TestArgs:
    """Dummy arguments class for testing"""
    pass

def main():
    ui = LinearAlgebraRichUI()
    console = Console()
    
    print("Testing direct print redirection:")
    
    # Create a simple test function that prints
    def test_print_func(text):
        print(f"Line 1: {text}")
        print(f"Line 2: Testing")
        print(f"Line 3: Some more text")
        return "Function result"
    
    # Test the capture function
    output, result = ui.run_with_capture(test_print_func, "Hello")
    print(f"Captured output: {repr(output)}")
    print(f"Function result: {repr(result)}")
    
    # Test normalize vector function
    print("\nTesting normalize vector:")
    args = TestArgs()
    args.vector = "3, 4"
    
    output, _ = ui.run_with_capture(ui.framework.normalize_vector, args)
    print("Captured normalize_vector output:")
    console.print(Panel(output, title="Normalize Vector Results", border_style="green"))
    
    # Test polar_to_cartesian function
    print("\nTesting polar_to_cartesian:")
    args = TestArgs()
    args.r = 5.0
    args.phi = 45.0
    args.degrees = True
    
    output, _ = ui.run_with_capture(ui.framework.polar_to_cartesian, args)
    print("Captured polar_to_cartesian output:")
    console.print(Panel(output, title="Polar to Cartesian Results", border_style="green"))
    
    print("\nThe results are now properly captured and displayed in a panel format.")
    print("This should fix the issue with results appearing above the panel.")

if __name__ == "__main__":
    main()