#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for Linear Algebra Framework
"""

import sys
import os
import argparse

def generate_banner():
    """Generate ASCII art banner for Linear Algebra Framework"""
    # Simple placeholder message if someone tries to run the CLI mode
    return "Linear Algebra Framework\n(For fancy banner, run in UI mode with 'pyfiglet' installed)"

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main entry point for the Linear Algebra application"""
    parser = argparse.ArgumentParser(description='Linear Algebra Framework')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--command', help='Directly execute a CLI command')
    parser.add_argument('--no-banner', action='store_true', help='Skip displaying the banner')
    parser.add_argument('--version', action='version', version='Linear Algebra Framework 1.7.1 - github.com/peaktwilight/linear-algebra-calculator')
    
    args, unknown_args = parser.parse_known_args()
    
    # Display the banner unless disabled
    if not args.no_banner:
        clear_screen()
        print(generate_banner())
        
        # Print welcome message
        print("Welcome to the Linear Algebra Framework!")
        print("This tool helps you practice and visualize linear algebra concepts.")
        print()
        
        # A short pause for the user to see the banner
        try:
            from time import sleep
            sleep(0.5)
        except:
            pass
    
    # Choose mode based on arguments
    if args.cli or args.command:
        # CLI mode
        print("Starting CLI mode...")
        from linalg_cli import LinearAlgebraExerciseFramework
        
        framework = LinearAlgebraExerciseFramework()
        
        if args.command:
            # Create synthetic command-line arguments
            sys.argv = [sys.argv[0]] + args.command.split() + unknown_args
            framework.run()
        else:
            # Use the original command-line arguments minus our own arguments
            sys.argv = [sys.argv[0]] + unknown_args
            framework.run()
    else:
        # GUI mode (default)
        try:
            print("Starting interactive UI mode...")
            from linalg_ui import LinearAlgebraRichUI
            
            ui = LinearAlgebraRichUI()
            ui.run()
        except ImportError as e:
            print("Error: Required UI dependencies not found.")
            print("Please install required packages with: pip install rich questionary")
            print(f"Original error: {e}")
            print("\nFalling back to CLI mode...")
            
            from linalg_cli import LinearAlgebraExerciseFramework
            framework = LinearAlgebraExerciseFramework()
            framework.run()

if __name__ == "__main__":
    main()