#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix UI script to replace console.capture with run_with_capture
"""
import re

def fix_ui_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Define the pattern to find
    pattern = r'with self\.console\.capture\(\) as capture:\n\s+self\.framework\.([a-zA-Z0-9_]+)\(args\)\n\s+\n\s+# Display results in a nicely formatted panel with green border\n\s+result_text = self\.format_results_for_display\(capture\.get\(\)\)'
    
    # Define the replacement
    replacement = r'# Use our redirection method to capture direct print output\n        output, _ = self.run_with_capture(self.framework.\1, args)\n        \n        # Display results in a nicely formatted panel with green border\n        result_text = self.format_results_for_display(output)'
    
    # Make the replacements
    new_content = re.sub(pattern, replacement, content)
    
    # Write the changes back
    with open(filename, 'w') as file:
        file.write(new_content)
    
    print("Replacements completed")

if __name__ == "__main__":
    fix_ui_file("/Users/peak/Downloads/python_25fs/ui_rich.py")