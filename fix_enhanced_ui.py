#!/usr/bin/env python3

"""
Script to verify and fix the enhanced_ui.py file
"""

import re
import sys

def main():
    # Read the enhanced_ui.py file
    with open('/Users/peak/Downloads/python_25fs/enhanced_ui.py', 'r') as f:
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
    with open('/Users/peak/Downloads/python_25fs/enhanced_ui.py', 'w') as f:
        f.write(content)
    
    print("File fixed successfully.")

if __name__ == "__main__":
    main()