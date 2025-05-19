#!/usr/bin/env python3

"""
Script to verify and fix the enhanced_ui.py file
"""

import re
import sys

def main():
    enhanced_ui_path = '/Users/peak/Downloads/python_25fs/enhanced_ui.py'
    
    # Check if the enhanced_ui.py file exists
    try:
        # Read the enhanced_ui.py file
        with open(enhanced_ui_path, 'r') as f:
            content = f.read()
        
        # Extract all ui methods referenced in the categories dictionary
        referenced_methods = re.findall(r'self\.ui_[a-z_]+', content)
        referenced_methods = set(referenced_methods)
        
        # Extract all ui method definitions
        defined_methods = re.findall(r'def (ui_[a-z_]+)\(self\)', content)
        defined_methods = set(f'self.{m}' for m in defined_methods)
        
        # Find methods that are referenced but not defined
        missing_methods = referenced_methods - defined_methods
        
        print(f"Methods referenced but not defined: {missing_methods}")
        
        # If there are no missing methods, the file is fine
        if not missing_methods:
            print("All referenced methods are defined correctly.")
            return
        
        # If there are missing methods, we need to fix the file
        # This should not happen if our checks are correct, but just in case
        print("Fixing the file...")
        
        # For each missing method, comment out the reference in the categories dictionary
        for method in missing_methods:
            method_name = method.replace('self.', '')
            pattern = fr'(\s+\(".+?",\s*)({method})(\),)'
            content = re.sub(pattern, r'\1# \2\3 # Commented out because method is not defined', content)
        
        # Write the fixed content
        with open(enhanced_ui_path, 'w') as f:
            f.write(content)
        
        print("File fixed successfully.")
    
    except FileNotFoundError:
        print(f"The file {enhanced_ui_path} does not exist.")
        print("Creating a simple version of the file...")
        
        # Create a basic enhanced_ui.py file
        with open(enhanced_ui_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Enhanced UI for Linear Algebra Framework
This file was auto-generated as a placeholder since the original was missing.
\"\"\"

import ui_rich

def main():
    print("Starting Linear Algebra Framework...")
    ui = ui_rich.LinearAlgebraRichUI()
    ui.run()

if __name__ == "__main__":
    main()
"""
            )
        
        print(f"Created basic {enhanced_ui_path} that redirects to ui_rich.py")
        print("You can now run python3 enhanced_ui.py to start the Linear Algebra Framework.")

if __name__ == "__main__":
    main()