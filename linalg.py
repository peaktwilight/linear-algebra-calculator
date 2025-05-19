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
    return """
    _     _                          _    _            _               
    | |   (_)_ __   ___  __ _ _ __   / \  | | __ _  ___| |__  _ __ __ _ 
    | |   | | '_ \ / _ \/ _` | '__| / _ \ | |/ _` |/ _ \ '_ \| '__/ _` |
    | |___| | | | |  __/ (_| | |   / ___ \| | (_| |  __/ |_) | | | (_| |
    |_____|_|_| |_|\___|\__,_|_|  /_/   \_\_|\__, |\___|_.__/|_|  \__,_|
                                             |___/                      
     _____                                             _    
    |  ___|_ __ __ _ _ __ ___   _____      _____  _ __| | __
    | |_ | '__/ _` | '_ ` _ \ / _ \ \ /\ / / _ \| '__| |/ /
    |  _|| | | (_| | | | | | |  __/\ V  V / (_) | |  |   < 
    |_|  |_|  \__,_|_| |_| |_|\___| \_/\_/ \___/|_|  |_|\_\\
                                                           
    """

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main entry point for the Linear Algebra application"""
    parser = argparse.ArgumentParser(description='Linear Algebra Framework')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--command', help='Directly execute a CLI command')
    parser.add_argument('--no-banner', action='store_true', help='Skip displaying the banner')
    parser.add_argument('--version', action='version', version='Linear Algebra Framework 1.0')
    
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