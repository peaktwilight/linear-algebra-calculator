#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ASCII art banner for Doruk's Linear Algebra Calculator
"""

try:
    import pyfiglet
except ImportError:
    print("Installing pyfiglet...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyfiglet"])
    import pyfiglet

def generate_banner():
    """Generate ASCII art banner for README"""
    # Create figlet objects with different fonts
    fig_standard = pyfiglet.Figlet(font='standard')
    fig_big = pyfiglet.Figlet(font='big')
    
    # Generate ASCII art for "Doruk's" with standard font
    doruk_ascii = fig_standard.renderText("Doruk's")
    
    # Generate ASCII art for "Linear Algebra Calculator" with big font
    calc_ascii = fig_big.renderText("Linear Algebra Calculator")
    
    # Combine the two
    banner = doruk_ascii + calc_ascii
    
    # Print the banner
    print(banner)
    
    # Return the banner for saving to a file
    return banner

def generate_fancy_banner():
    """Generate a one-line ASCII art for README with a more compact style"""
    try:
        fig = pyfiglet.Figlet(font='slant')
        line1 = fig.renderText("Doruk's")
        line2 = fig.renderText("Linear Algebra Calculator")
        
        # Combine the ASCII art
        banner = line1 + line2
        
        # Print the banner
        print(banner)
        
        # Return the banner for saving to a file
        return banner
    except Exception as e:
        print(f"Error generating banner: {e}")
        return None

def save_to_readme(banner):
    """Update the README.md file with the new banner"""
    try:
        with open('README.md', 'r') as f:
            content = f.read()
        
        # Find where to insert the banner
        start_marker = "# Doruk's Linear Algebra Calculator\n\n```"
        end_marker = "```\n\nA comprehensive toolkit"
        
        if start_marker in content and end_marker in content:
            # Extract the part before the start marker
            before_banner = content.split(start_marker)[0] + start_marker + "\n"
            
            # Extract the part after the end marker
            after_banner = end_marker.split("\n", 1)[1]
            
            # Combine the parts with the new banner
            new_content = before_banner + banner + after_banner
            
            # Write the updated content back to the file
            with open('README.md', 'w') as f:
                f.write(new_content)
            
            print("Banner successfully updated in README.md")
        else:
            print("Couldn't find the right place to insert the banner. The README structure may have changed.")
    except Exception as e:
        print(f"Error updating README: {e}")

def update_readme_directly(banner):
    """Update the README.md file with the new banner without user interaction"""
    try:
        with open('README.md', 'r') as f:
            readme_content = f.read()
            
        # Create the updated content
        new_banner_block = f"# Doruk's Linear Algebra Calculator\n\n```\n{banner}```\n\n"
        
        # Find the position after the title
        title_pos = readme_content.find("# Doruk's Linear Algebra Calculator")
        if title_pos == -1:
            print("Could not find the title in README.md")
            return False
            
        # Find the position of the next section (## Features)
        next_section_pos = readme_content.find("## Features", title_pos)
        if next_section_pos == -1:
            print("Could not find the next section in README.md")
            return False
            
        # Create the parts
        start_content = readme_content[:title_pos]
        end_content = readme_content[next_section_pos:]
        
        # Build the new content
        new_content = start_content + new_banner_block + "A comprehensive toolkit for learning and solving linear algebra problems, developed for the LAG Fachmodul at FHNW.\n\n" + end_content
        
        # Write to file
        with open('README.md', 'w') as f:
            f.write(new_content)
            
        print("README.md updated successfully with new banner.")
        return True
    except Exception as e:
        print(f"Error updating README: {e}")
        return False

if __name__ == "__main__":
    # Generate the banner
    banner = generate_fancy_banner()
    
    # Save to README and a separate file
    if banner:
        # Update README directly
        update_readme_directly(banner)
        
        # Save to separate file
        with open('banner.txt', 'w') as f:
            f.write(banner)
        print(f"Banner saved to banner.txt")