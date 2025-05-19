#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rich UI for Linear Algebra Framework
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse
import numpy as np
from linalg_cli import LinearAlgebraExerciseFramework

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm
    import questionary
    from questionary import Style
except ImportError:
    print("Required packages not found. Please install with:")
    print("pip install rich questionary")
    sys.exit(1)

# Define color scheme
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),       # Purple question mark
    ('question', 'bold fg:#03a9f4'),    # Light blue questions
    ('answer', 'fg:#2196f3'),           # Blue answers
    ('pointer', 'fg:#673ab7 bold'),     # Purple pointer
    ('highlighted', 'fg:#673ab7 bold'), # Purple highlighted text
    ('selected', 'fg:#4caf50'),         # Green selected items
    ('separator', 'fg:#546e7a'),        # Separator line
    ('instruction', 'fg:#546e7a'),      # Instructions text
])

class LinearAlgebraRichUI:
    def __init__(self):
        self.framework = LinearAlgebraExerciseFramework()
        self.console = Console()
        self.banner = self.generate_banner()
        
        # Define categories and operations
        self.categories = {
            "Learning Resources": [
                ("Browse Help Topics", self.ui_browse_help_topics),
                ("Show Example Exercises", self.ui_show_example_exercises),
                ("Linear Algebra Guide", self.ui_linear_algebra_guide),
                ("Common Homework Patterns", self.ui_common_homework_patterns),
                ("Recognize Problem Types", self.ui_recognize_problem_types),
            ],
            "Vector Operations": [
                ("Convert polar to Cartesian coordinates", self.ui_polar_to_cartesian),
                ("Normalize a vector", self.ui_normalize_vector),
                ("Create vector with direction and length", self.ui_vector_direction_length),
                ("Calculate vector shadow (projection)", self.ui_vector_shadow),
                ("Check if vectors are orthogonal", self.ui_check_orthogonal),
                ("Calculate angle between vectors", self.ui_vector_angle),
                ("Calculate cross product", self.ui_cross_product),
                ("Calculate triangle area", self.ui_triangle_area),
                ("Calculate point-line distance", self.ui_point_line_distance),
                ("Check if vectors are collinear", self.ui_check_collinear),
            ],
            "Matrix Operations": [
                ("Solve system with Gaussian elimination", self.ui_solve_gauss),
                ("Check if vectors are coplanar", self.ui_check_coplanar),
                ("Check if vector is solution to system", self.ui_check_solution),
                ("Solve vector equation", self.ui_vector_equation),
                ("Extract matrix elements", self.ui_matrix_element),
                ("Find intersections of planes", self.ui_intersection_planes),
                ("Find homogeneous intersections", self.ui_homogeneous_intersection),
                ("Identify pivot and free variables", self.ui_find_pivot_free_vars),
                ("Perform basic matrix operations", self.ui_matrix_operations),
                ("Calculate matrix product", self.ui_matrix_product),
            ],
            "Other Operations": [
                ("Calculate sum of series", self.ui_sum_series),
                ("Check particular solutions", self.ui_check_particular_solution),
                ("Calculate point-plane distance", self.ui_point_plane_distance),
            ]
        }
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def generate_banner(self):
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
    
    def display_header(self, title="Linear Algebra Exercise Framework"):
        self.clear_screen()
        # Only show the banner on the main menu
        if title == "Linear Algebra Exercise Framework":
            self.console.print(self.banner)
        self.console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="blue"))
        self.console.print()
    
    def print_result(self, result, title="Result"):
        self.console.print(Panel(result, title=title, border_style="green"))
        
    def format_results_for_display(self, result_text):
        """Format the results as a nicely formatted string for display in a panel"""
        # Just return the result text as is for now, with proper formatting
        if result_text.strip():
            return result_text
        else:
            # If the captured text is empty, return a message
            return "[italic]No results to display[/italic]"
            
    def run_with_capture(self, func, *args, **kwargs):
        """Run a function and capture its output even if it directly uses print()"""
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Redirect stdout to a string buffer
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = func(*args, **kwargs)
            
        # Get the captured output
        output = buffer.getvalue()
        return output, result
    
    def wait_for_user(self):
        self.console.print()
        self.console.print("[yellow]Press Enter to continue...[/yellow]")
        input()
    
    def show_input_help(self, input_type="vector"):
        """Display help about input formatting"""
        if input_type == "vector":
            self.console.print(Panel("""[bold]Vector Input Format Tips:[/bold]
            
- Separate components with commas or spaces: [green]"1, 2, 3"[/green] or [green]"1 2 3"[/green]
- Decimal numbers are supported: [green]"1.5, -2.3, 0.0"[/green]
- Square brackets are optional: [green]"[1, 2, 3]"[/green] or [green]"1, 2, 3"[/green]
- For 2D vectors: [green]"x, y"[/green] (e.g., [green]"3, 4"[/green])
- For 3D vectors: [green]"x, y, z"[/green] (e.g., [green]"1, 2, 3"[/green])
- For higher dimensions, just add more components
            """, title="Vector Input Help", border_style="blue"))
        
        elif input_type == "matrix":
            self.console.print(Panel("""[bold]Matrix Input Format Tips:[/bold]
            
[bold]Method 1:[/bold] Single line with semicolons separating rows
- Format: [green]"a, b, c; d, e, f; g, h, i"[/green]
- Example 2×3 matrix: [green]"1, 2, 3; 4, 5, 6"[/green]
- Example 3×3 matrix: [green]"1, 2, 3; 4, 5, 6; 7, 8, 9"[/green]

[bold]Method 2:[/bold] Multiple lines (press Enter after each row, empty line to finish)
- Line 1: [green]1 2 3[/green]
- Line 2: [green]4 5 6[/green]
- Line 3: [green]7 8 9[/green]
- Line 4: [empty line to finish]

[bold]Augmented Matrix[/bold] (for linear systems [A|b]):
- Include the right-hand side as the last column
- Example: [green]"1, 2, 3, 10; 4, 5, 6, 20"[/green] represents:
  1x + 2y + 3z = 10
  4x + 5y + 6z = 20
            """, title="Matrix Input Help", border_style="blue"))
        
        elif input_type == "equation":
            self.console.print(Panel("""[bold]Equation Input Format Tips:[/bold]
            
- Use standard algebraic notation with symbols u, v, x, y, z
- Operators: + (add), - (subtract), * (multiply), / (divide), ** (power)
- Examples:
  - Vector equation: [green]"u-v-x-(x+v+x)"[/green]
  - Linear equation: [green]"3*x + 2*y - z"[/green]
- Functions: sin, cos, exp, log, etc.
- Using parentheses for grouping is recommended
            """, title="Equation Input Help", border_style="blue"))
            
        elif input_type == "series":
            self.console.print(Panel("""[bold]Series Formula Format Tips:[/bold]
            
- Use 'i' as the variable representing the index
- Examples:
  - Arithmetic sequence: [green]"5+3*i"[/green] (5, 8, 11, 14, ...)
  - Geometric sequence: [green]"2**i"[/green] (2, 4, 8, 16, ...)
  - Sum formula: [green]"i*(i+1)/2"[/green] (triangle numbers)
  - Complex expression: [green]"1+(i+3)**2"[/green]
- Available operations: +, -, *, /, **, sin, cos, exp, log
            """, title="Series Formula Help", border_style="blue"))
    
    def prompt_vector(self, message="Enter a vector (comma or space separated values)") -> np.ndarray:
        """Prompt user for a vector with rich formatting"""
        # Show help first
        self.show_input_help("vector")
        
        while True:
            try:
                vector_str = questionary.text(
                    message,
                    style=custom_style
                ).ask()
                
                # Handle cancellation
                if vector_str is None:
                    return None
                
                # Handle empty input
                if not vector_str.strip():
                    return np.array([])
                
                return self.framework.parse_vector(vector_str)
            except ValueError as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
                self.console.print("[yellow]Press ? for help or try again.[/yellow]")
                help_response = questionary.confirm("Show input help again?", style=custom_style).ask()
                if help_response:
                    self.show_input_help("vector")
    
    def prompt_matrix(self, message="Enter a matrix") -> np.ndarray:
        """Prompt user for a matrix with rich formatting"""
        # Show help first
        self.show_input_help("matrix")
        
        self.console.print(f"[bold cyan]{message}[/bold cyan]")
        self.console.print("[yellow](Enter each row, or use semicolons for single-line input, empty line to finish)[/yellow]")
        
        lines = []
        while True:
            line = input()
            if not line and lines:  # Empty line to finish if multiline
                break
            elif line == "?":
                self.show_input_help("matrix")
                continue
            elif ";" in line:  # Single line with semicolons
                matrix_str = line
                break
            elif line:  # Add non-empty line
                lines.append(line)
        
        try:
            if lines:
                matrix_str = "\n".join(lines)
            else:
                matrix_str = matrix_str  # Use the single-line input with semicolons
                
            return self.framework.parse_matrix(matrix_str)
        except ValueError as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
            self.console.print("[yellow]Would you like to try again?[/yellow]")
            retry = questionary.confirm("Try again?", style=custom_style).ask()
            if retry:
                return self.prompt_matrix(message)
            else:
                return None
    
    def display_vector(self, vector, label="Vector"):
        """Display a vector with nice formatting"""
        table = Table(title=label)
        table.add_column("Index", style="cyan")
        table.add_column("Value", style="green")
        
        for i, val in enumerate(vector):
            table.add_row(str(i), f"{val:.6f}" if isinstance(val, float) else str(val))
        
        self.console.print(table)
    
    def display_matrix(self, matrix, title="Matrix"):
        """Display a matrix with nice formatting"""
        table = Table(title=title)
        
        # Add column headers (indices)
        headers = [""] + [str(i) for i in range(matrix.shape[1])]
        for header in headers:
            if header == "":
                table.add_column(header, style="bold cyan", justify="right")
            else:
                table.add_column(header, style="cyan")
        
        # Add rows
        for i, row in enumerate(matrix):
            row_values = [str(i)] + [f"{val:.6f}" if isinstance(val, float) and abs(val) > 1e-10 else "0" if isinstance(val, float) and abs(val) <= 1e-10 else str(val) for val in row]
            table.add_row(*row_values)
        
        self.console.print(table)
    
    def run_menu(self):
        while True:
            self.display_header()
            
            # Create list of categories for selection
            categories = list(self.categories.keys())
            categories.append("Exit")
            
            category = questionary.select(
                "Select a category:",
                choices=categories,
                style=custom_style
            ).ask()
            
            if category == "Exit" or category is None:
                sys.exit(0)
            
            # Get operations for the selected category
            operations = self.categories[category]
            operation_names = [op[0] for op in operations]
            operation_names.append("Back to main menu")
            
            operation = questionary.select(
                f"Select an operation from [{category}]:",
                choices=operation_names,
                style=custom_style
            ).ask()
            
            if operation == "Back to main menu" or operation is None:
                continue
            
            # Find and execute the selected operation function
            for op_name, op_func in operations:
                if op_name == operation:
                    op_func()
                    break
    
    # UI Functions for each operation
    def ui_polar_to_cartesian(self):
        self.display_header("Convert Polar to Cartesian Coordinates")
        
        self.console.print(Panel("""[bold]Polar Coordinates Format:[/bold]
        
- [green]Radius (r):[/green] Distance from origin (e.g., [green]5.0[/green])
- [green]Angle (phi):[/green] Angle in radians or degrees (e.g., [green]45[/green] or [green]0.7853[/green])
- Choose whether angle is in degrees or radians
        """, title="Input Guide", border_style="blue"))
        
        r = questionary.text(
            "Enter radius (r):", 
            validate=lambda text: text.replace('.', '', 1).isdigit(), 
            style=custom_style
        ).ask()
        
        phi = questionary.text(
            "Enter angle (phi):", 
            validate=lambda text: text.replace('.', '', 1).replace('-', '', 1).isdigit(), 
            style=custom_style
        ).ask()
        
        is_degrees = questionary.confirm("Is angle in degrees?", style=custom_style).ask()
        
        class Args:
            pass
        
        args = Args()
        args.r = float(r)
        args.phi = float(phi)
        args.degrees = is_degrees
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.polar_to_cartesian, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_normalize_vector(self):
        self.display_header("Normalize a Vector")
        
        self.console.print(Panel("""[bold]Vector Normalization:[/bold]
        
This operation takes a vector and produces a unit vector (length 1) in the same direction.

[bold]Formula:[/bold] unit_vector = vector / |vector|
        
Examples: 
- [green]"3, 4"[/green] normalizes to [green]"0.6, 0.8"[/green]
- [green]"1, 1, 1"[/green] normalizes to [green]"0.577, 0.577, 0.577"[/green]
        """, title="Operation Guide", border_style="blue"))
        
        vector = self.prompt_vector()
        if vector is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.vector = " ".join(map(str, vector))
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.normalize_vector, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_vector_direction_length(self):
        self.display_header("Create Vector with Direction and Length")
        
        self.console.print(Panel("""[bold]Vector Scaling:[/bold]
        
This operation takes a direction vector and scales it to a specific length.

[bold]Formula:[/bold] result = direction * (length / |direction|)
        
Example: 
- Direction [green]"3, 4"[/green] with length [green]10[/green] gives [green]"6, 8"[/green]
        """, title="Operation Guide", border_style="blue"))
        
        direction = self.prompt_vector("Enter direction vector")
        if direction is None:
            return
            
        length = questionary.text(
            "Enter desired length:",
            validate=lambda text: text.replace('.', '', 1).isdigit(),
            style=custom_style
        ).ask()
        
        if length is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.direction = " ".join(map(str, direction))
        args.length = float(length)
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.vector_direction_length, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_vector_shadow(self):
        self.display_header("Calculate Vector Shadow (Projection)")
        
        self.console.print(Panel("""[bold]Vector Projection (Shadow):[/bold]
        
This operation calculates the projection of one vector onto another.

[bold]Formula:[/bold] 
- Scalar projection = (a·b) / |a|
- Vector projection = (a·b) / (a·a) * a

Where:
- a is the vector to project onto
- b is the vector being projected

Example:
- a = [green]"3, -4"[/green], b = [green]"12, -1"[/green]
- Result shows both the scalar value and the vector projection
        """, title="Operation Guide", border_style="blue"))
        
        vector_a = self.prompt_vector("Enter vector to project onto (a)")
        if vector_a is None:
            return
            
        vector_b = self.prompt_vector("Enter vector to be projected (b)")
        if vector_b is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.vector_a = " ".join(map(str, vector_a))
        args.vector_b = " ".join(map(str, vector_b))
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.vector_shadow, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_check_orthogonal(self):
        self.display_header("Check if Vectors are Orthogonal")
        
        self.console.print(Panel("""[bold]Vector Orthogonality:[/bold]
        
This operation checks if vectors are perpendicular to each other.

[bold]Formula:[/bold] Two vectors a and b are orthogonal if a·b = 0

Example:
- [green]"1, 0"[/green] and [green]"0, 1"[/green] are orthogonal (perpendicular)
- [green]"1, 5, 2"[/green] and [green]"263, -35, -44"[/green] are orthogonal

[bold]Procedure:[/bold]
1. Enter one reference vector
2. Enter multiple vectors to check against the reference
3. The dot product will be calculated for each pair
        """, title="Operation Guide", border_style="blue"))
        
        vector = self.prompt_vector("Enter reference vector")
        if vector is None:
            return
        
        self.console.print("[bold cyan]Enter vectors to check (empty input to finish)[/bold cyan]")
        check_vectors = []
        
        while True:
            vec = self.prompt_vector(f"Enter vector #{len(check_vectors) + 1} (or empty to finish)")
            if vec is None or (isinstance(vec, np.ndarray) and vec.size == 0):
                break
            check_vectors.append(" ".join(map(str, vec)))
        
        if not check_vectors:
            self.console.print("[yellow]No vectors to check.[/yellow]")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.vector = " ".join(map(str, vector))
        args.check_vectors = check_vectors
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.check_orthogonal, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_vector_angle(self):
        self.display_header("Calculate Angle Between Vectors")
        
        self.console.print(Panel("""[bold]Angle Between Vectors:[/bold]
        
This operation calculates the angle between two vectors.

[bold]Formula:[/bold] cos(θ) = (a·b) / (|a|·|b|)

Example:
- Vectors [green]"1, 1, -1"[/green] and [green]"1, -1, 1"[/green] 
- Result will show angle in both radians and degrees
        """, title="Operation Guide", border_style="blue"))
        
        vector_a = self.prompt_vector("Enter first vector (a)")
        if vector_a is None:
            return
            
        vector_b = self.prompt_vector("Enter second vector (b)")
        if vector_b is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.vector_a = " ".join(map(str, vector_a))
        args.vector_b = " ".join(map(str, vector_b))
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.vector_angle, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_cross_product(self):
        self.display_header("Calculate Cross Product")
        
        self.console.print(Panel("""[bold]Cross Product:[/bold]
        
This operation calculates the cross product of two vectors, which produces a vector perpendicular to both input vectors.

[bold]Important:[/bold] Cross product is primarily defined for 3D vectors. For 2D vectors, they will be expanded to 3D with z=0.

[bold]Formula:[/bold] a × b = |a|·|b|·sin(θ)·n̂

Where:
- θ is the angle between the vectors
- n̂ is the unit vector perpendicular to both a and b

Example:
- a = [green]"-2, 0, 0"[/green], b = [green]"0, 9, 8"[/green]
- Cross product will be perpendicular to both a and b
        """, title="Operation Guide", border_style="blue"))
        
        vector_a = self.prompt_vector("Enter first vector (a)")
        if vector_a is None:
            return
            
        vector_b = self.prompt_vector("Enter second vector (b)")
        if vector_b is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.vector_a = " ".join(map(str, vector_a))
        args.vector_b = " ".join(map(str, vector_b))
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.cross_product, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_triangle_area(self):
        self.display_header("Calculate Triangle Area")
        
        self.console.print(Panel("""[bold]Triangle Area:[/bold]
        
This operation calculates the area of a triangle defined by three points.

[bold]Formula:[/bold] Area = |AB × AC| / 2

Where:
- AB and AC are vectors from point A to points B and C
- × represents the cross product

[bold]Input:[/bold] Three points in 2D or 3D space
For 2D points, enter: [green]"x, y"[/green]
For 3D points, enter: [green]"x, y, z"[/green]

Example:
- A = [green]"0, 4, 2"[/green], B = [green]"0, 8, 5"[/green], C = [green]"0, 8, -1"[/green]
        """, title="Operation Guide", border_style="blue"))
        
        point_a = self.prompt_vector("Enter first point (A)")
        if point_a is None:
            return
            
        point_b = self.prompt_vector("Enter second point (B)")
        if point_b is None:
            return
            
        point_c = self.prompt_vector("Enter third point (C)")
        if point_c is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.point_a = " ".join(map(str, point_a))
        args.point_b = " ".join(map(str, point_b))
        args.point_c = " ".join(map(str, point_c))
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.triangle_area, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_point_line_distance(self):
        self.display_header("Calculate Point-Line Distance")
        
        self.console.print(Panel("""[bold]Point-Line Distance:[/bold]
        
This operation calculates the shortest distance from a point to a line.

[bold]Formula:[/bold] d = |v × (B-A)| / |v|

Where:
- A is a point on the line
- v is the direction vector of the line
- B is the point from which we want to find the distance
- × represents the cross product

[bold]Input:[/bold]
- Point on line (A): [green]"x, y, z"[/green]
- Direction vector (v): [green]"dx, dy, dz"[/green]
- Point to find distance from (B): [green]"x, y, z"[/green]

Example:
- A = [green]"3, 0, 0"[/green], v = [green]"2, 0, 0"[/green], B = [green]"5, 10, 0"[/green]
        """, title="Operation Guide", border_style="blue"))
        
        point_a = self.prompt_vector("Enter point on line (A)")
        if point_a is None:
            return
            
        direction = self.prompt_vector("Enter line direction vector")
        if direction is None:
            return
            
        point_b = self.prompt_vector("Enter point to find distance from (B)")
        if point_b is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.point_a = " ".join(map(str, point_a))
        args.direction = " ".join(map(str, direction))
        args.point_b = " ".join(map(str, point_b))
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.point_line_distance, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_check_collinear(self):
        self.display_header("Check if Vectors are Collinear")
        
        self.console.print(Panel("""[bold]Vector Collinearity:[/bold]
        
This operation checks if vectors are collinear (parallel or anti-parallel).

[bold]Method:[/bold] Vectors are collinear if one is a scalar multiple of the other.
We check this by applying Gaussian elimination and looking for zero rows.

[bold]Input:[/bold] Enter at least 2 vectors to check

Example:
- [green]"-3, 2"[/green] and [green]"6, -4"[/green] are collinear (one is -2 times the other)
- [green]"-3, 2"[/green] and [green]"2, -3"[/green] are not collinear
        """, title="Operation Guide", border_style="blue"))
        
        self.console.print("[bold cyan]Enter vectors (empty input to finish)[/bold cyan]")
        vectors = []
        
        while True:
            vec = self.prompt_vector(f"Enter vector #{len(vectors) + 1} (or empty to finish)")
            if vec is None or (isinstance(vec, np.ndarray) and vec.size == 0):
                break
            vectors.append(" ".join(map(str, vec)))
        
        if len(vectors) < 2:
            self.console.print("[yellow]Need at least 2 vectors to check collinearity.[/yellow]")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.vectors = vectors
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.check_collinear, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_solve_gauss(self):
        self.display_header("Solve System with Gaussian Elimination")
        
        self.console.print(Panel("""[bold]Solving Linear Systems with Gaussian Elimination:[/bold]
        
This operation solves a system of linear equations using Gaussian elimination.

[bold]Input Format:[/bold] Augmented matrix [A|b]
- Each row represents one equation
- The last column is the constant term

Example representing the system:
  x - 4y - 2z = -25
    - 3y + 6z = -18
  7x - 13y - 4z = -85

Matrix format:
[green]"1, -4, -2, -25; 0, -3, 6, -18; 7, -13, -4, -85"[/green]

[bold]Output:[/bold]
- The row echelon form of the matrix
- Solution (if unique) or parametric form (if infinitely many solutions)
        """, title="Operation Guide", border_style="blue"))
        
        matrix = self.prompt_matrix("Enter augmented matrix [A|b]")
        if matrix is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.solve_gauss, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_check_coplanar(self):
        self.display_header("Check if Vectors are Coplanar")
        
        self.console.print(Panel("""[bold]Vector Coplanarity:[/bold]
        
This operation checks if vectors lie in the same plane (are coplanar).

[bold]Method:[/bold] Vectors are coplanar if they are linearly dependent in 3D space.
We check this by applying Gaussian elimination and looking for zero rows.

[bold]Input:[/bold] Enter at least 3 vectors to check

Example:
- [green]"3, 2, 0"[/green], [green]"0, 4, 3"[/green], and [green]"3, 10, 6"[/green] are coplanar
- [green]"3, 0, -1"[/green], [green]"0, 4, 3"[/green], and [green]"15, -4, -7"[/green] are not coplanar

[bold]Output:[/bold] If vectors are coplanar, a linear combination yielding the zero vector will be shown
        """, title="Operation Guide", border_style="blue"))
        
        self.console.print("[bold cyan]Enter vectors (empty input to finish)[/bold cyan]")
        vectors = []
        
        while True:
            vec = self.prompt_vector(f"Enter vector #{len(vectors) + 1} (or empty to finish)")
            if vec is None or (isinstance(vec, np.ndarray) and vec.size == 0):
                break
            vectors.append(" ".join(map(str, vec)))
        
        if len(vectors) < 3:
            self.console.print("[yellow]Need at least 3 vectors to check coplanarity.[/yellow]")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.vectors = vectors
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.check_coplanar, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_check_solution(self):
        self.display_header("Check if Vector is Solution to System")
        
        self.console.print(Panel("""[bold]Check Solutions to Linear Systems:[/bold]
        
This operation checks if vectors are solutions to a system of linear equations.

[bold]Input:[/bold]
1. Coefficient matrix A (without constants)
2. Optional right-hand side vector b
3. Vectors to check as potential solutions

[bold]Example:[/bold]
For the system:
  5x + 2y = 5
  3x + 2y = 7

Matrix A: [green]"5, 2; 3, 2"[/green]
Vector b: [green]"5, 7"[/green]
Vector to check: [green]"-1, 5"[/green]

[bold]Method:[/bold] For each vector v, we compute A·v and check if it equals b
        """, title="Operation Guide", border_style="blue"))
        
        matrix = self.prompt_matrix("Enter coefficient matrix A")
        if matrix is None:
            return
        
        has_rhs = questionary.confirm("Does the system have a right-hand side b?", style=custom_style).ask()
        
        rhs = None
        if has_rhs:
            rhs = self.prompt_vector("Enter right-hand side vector b")
            if rhs is None:
                return
        
        self.console.print("[bold cyan]Enter vectors to check (empty input to finish)[/bold cyan]")
        vectors = []
        
        while True:
            vec = self.prompt_vector(f"Enter vector #{len(vectors) + 1} (or empty to finish)")
            if vec is None or (isinstance(vec, np.ndarray) and vec.size == 0):
                break
            vectors.append(" ".join(map(str, vec)))
        
        if not vectors:
            self.console.print("[yellow]No vectors to check.[/yellow]")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.rhs = " ".join(map(str, rhs)) if rhs is not None else None
        args.vectors = vectors
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.check_solution, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_vector_equation(self):
        self.display_header("Solve Vector Equation")
        
        # Show help
        self.show_input_help("equation")
        
        self.console.print(Panel("""[bold]Vector Equation Solving:[/bold]
        
This operation uses symbolic mathematics to solve vector equations.

[bold]Example Equations:[/bold]
- [green]"u-v-x-(x+v+x)"[/green] - Solve for x in terms of u and v
- [green]"u-v-x-(4*x-v+x)"[/green] - More complex equation with x

[bold]Available Symbols:[/bold] u, v, x, y, z
[bold]Available Operations:[/bold] +, -, *, /, **, =
        """, title="Operation Guide", border_style="blue"))
        
        equation = questionary.text("Enter equation:", style=custom_style).ask()
        if equation is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.equation = equation
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.vector_equation, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_matrix_element(self):
        self.display_header("Extract Matrix Elements")
        
        self.console.print(Panel("""[bold]Matrix Element Extraction:[/bold]
        
This operation allows you to extract and view specific parts of a matrix.

[bold]Options:[/bold]
- Extract a specific element at position (i,j)
- Extract an entire row i
- Extract an entire column j
- View the transpose of the matrix

[bold]Indexing:[/bold]
- Rows and columns are zero-indexed
- For a 3×3 matrix, valid indices are 0, 1, 2

[bold]Example:[/bold]
For matrix [green]"4, 4, -16, 4; 1, 0, -4, 2; 5, 2, -20, 8"[/green]
- Element (1,3) is the value in the second row, fourth column
        """, title="Operation Guide", border_style="blue"))
        
        matrix = self.prompt_matrix("Enter matrix")
        if matrix is None:
            return
            
        # Display the matrix first
        self.display_matrix(matrix)
        
        operations = []
        
        if questionary.confirm("Extract specific element?", style=custom_style).ask():
            i = questionary.text("Enter row index:", validate=lambda text: text.isdigit(), style=custom_style).ask()
            j = questionary.text("Enter column index:", validate=lambda text: text.isdigit(), style=custom_style).ask()
            operations.append(("element", f"{i},{j}"))
        
        if questionary.confirm("Extract specific row?", style=custom_style).ask():
            row = questionary.text("Enter row index:", validate=lambda text: text.isdigit(), style=custom_style).ask()
            operations.append(("row", row))
        
        if questionary.confirm("Extract specific column?", style=custom_style).ask():
            column = questionary.text("Enter column index:", validate=lambda text: text.isdigit(), style=custom_style).ask()
            operations.append(("column", column))
        
        show_transpose = questionary.confirm("Show matrix transpose?", style=custom_style).ask()
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.element = next((value for op, value in operations if op == "element"), None)
        args.row = int(next((value for op, value in operations if op == "row"), None) or -1) if any(op == "row" for op in [o[0] for o in operations]) else None
        args.column = int(next((value for op, value in operations if op == "column"), None) or -1) if any(op == "column" for op in [o[0] for o in operations]) else None
        args.transpose = show_transpose
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.matrix_element, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_intersection_planes(self):
        self.display_header("Find Intersections of Planes")
        
        self.console.print(Panel("""[bold]Plane Intersection Analysis:[/bold]
        
This operation determines how planes intersect in 3D space.

[bold]Input:[/bold] Augmented matrix representing plane equations
Each row is a plane equation: ax + by + cz + d = 0

[bold]Example:[/bold]
For planes:
  2x + y - 2z = -1
  x + 8y - 4z = 10
  6x - y + 18z = 81

Matrix: [green]"2, 1, -2, -1; 1, 8, -4, 10; 6, -1, 18, 81"[/green]

[bold]Possible results:[/bold]
- Single point (three planes intersect at one point)
- Line (two or more planes intersect along a line)
- Plane (two or more planes are identical)
- No intersection (planes are parallel but don't coincide)
        """, title="Operation Guide", border_style="blue"))
        
        matrix = self.prompt_matrix("Enter augmented matrix representing plane equations")
        if matrix is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.intersection_planes, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_homogeneous_intersection(self):
        self.display_header("Find Homogeneous Intersections")
        
        self.console.print(Panel("""[bold]Homogeneous Plane Intersection:[/bold]
        
This operation determines how planes passing through the origin intersect.

[bold]Input:[/bold] Coefficient matrix representing homogeneous plane equations
Each row is a plane equation: ax + by + cz = 0 (notice no constant term)

[bold]Example:[/bold]
For planes:
  0 + 9y + 8z = 0
  8x + 9y + 6z = 0
  10x + 0 + 6z = 0

Matrix: [green]"0, 9, 8; 8, 9, 6; 10, 0, 6"[/green]

[bold]Possible results:[/bold]
- Origin only (three linearly independent planes)
- Line through origin (intersection along a line)
- Plane through origin (two planes are identical)
        """, title="Operation Guide", border_style="blue"))
        
        matrix = self.prompt_matrix("Enter coefficient matrix representing homogeneous plane equations")
        if matrix is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.homogeneous_intersection, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_find_pivot_free_vars(self):
        self.display_header("Identify Pivot and Free Variables")
        
        self.console.print(Panel("""[bold]Pivot and Free Variables Analysis:[/bold]
        
This operation identifies pivot and free variables in a linear system.

[bold]Definitions:[/bold]
- Pivot variables: Variables corresponding to leading entries in the row echelon form
- Free variables: Variables that can be assigned arbitrary values

[bold]Input:[/bold] Augmented matrix representing the system

[bold]Example:[/bold]
For system:
  3x + 3y - 8z + 6u + 0 = 14
  x + y - 4z + 2u + v = 3
  5x + 5y - 20z + 10u + 3v = 13

Matrix: [green]"3, 3, -8, 6, 0, 14; 1, 1, -4, 2, 1, 3; 5, 5, -20, 10, 3, 13"[/green]

[bold]Output:[/bold]
- Identification of pivot and free variables
- Parametric form of the solution if free variables exist
        """, title="Operation Guide", border_style="blue"))
        
        matrix = self.prompt_matrix("Enter augmented matrix representing the system")
        if matrix is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.find_pivot_free_vars, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_matrix_operations(self):
        self.display_header("Perform Basic Matrix Operations")
        
        self.console.print(Panel("""[bold]Basic Matrix Operations:[/bold]
        
This tool performs common matrix operations:

[bold]1. Addition (A + B):[/bold]
- Matrices must have the same dimensions
- Result has same dimensions as inputs
- Example: [green]"1, 2; 3, 4"[/green] + [green]"5, 6; 7, 8"[/green] = [green]"6, 8; 10, 12"[/green]

[bold]2. Subtraction (A - B):[/bold]
- Matrices must have the same dimensions
- Example: [green]"5, 6; 7, 8"[/green] - [green]"1, 2; 3, 4"[/green] = [green]"4, 4; 4, 4"[/green]

[bold]3. Scalar Multiplication (c * A):[/bold]
- Multiplies every element by a scalar
- Example: 2 * [green]"1, 2; 3, 4"[/green] = [green]"2, 4; 6, 8"[/green]

[bold]4. Transpose (A^T):[/bold]
- Flips rows and columns
- Example: [green]"1, 2; 3, 4"[/green]^T = [green]"1, 3; 2, 4"[/green]
        """, title="Operation Guide", border_style="blue"))
        
        operations = {
            "Add matrices (A + B)": "add",
            "Subtract matrices (A - B)": "subtract",
            "Multiply by scalar (c * A)": "multiply_scalar",
            "Transpose matrix (A^T)": "transpose"
        }
        
        operation_name = questionary.select(
            "Select operation:",
            choices=list(operations.keys()),
            style=custom_style
        ).ask()
        
        if operation_name is None:
            return
            
        operation = operations[operation_name]
        
        matrix_a = self.prompt_matrix("Enter matrix A")
        if matrix_a is None:
            return
        
        matrix_b = None
        if operation in ["add", "subtract"]:
            matrix_b = self.prompt_matrix("Enter matrix B")
            if matrix_b is None:
                return
        
        scalar = None
        if operation == "multiply_scalar":
            scalar_str = questionary.text(
                "Enter scalar value:",
                validate=lambda text: text.replace('.', '', 1).replace('-', '', 1).isdigit(),
                style=custom_style
            ).ask()
            
            if scalar_str is None:
                return
                
            scalar = float(scalar_str)
        
        class Args:
            pass
        
        args = Args()
        args.operation = operation
        args.matrix_a = str(matrix_a.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        if matrix_b is not None:
            args.matrix_b = str(matrix_b.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        else:
            args.matrix_b = None
            
        args.scalar = scalar
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.matrix_operations, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_matrix_product(self):
        self.display_header("Calculate Matrix Product")
        
        self.console.print(Panel("""[bold]Matrix Multiplication:[/bold]
        
This operation calculates the product of two matrices.

[bold]Rules:[/bold]
- For matrices A (m×n) and B (p×q), multiplication A×B is possible only if n = p
- The resulting matrix C = A×B will have dimensions m×q
- Each element c_ij = sum(a_ik * b_kj) for all k

[bold]Example:[/bold]
A = [green]"0, 4, 2; 2, -1, 5; 6, 0, -3"[/green] (3×3 matrix)
B = [green]"4, 1, 9; 5, 4, 2; -3, 5, 2"[/green] (3×3 matrix)
Result will be a 3×3 matrix

[bold]Special cases:[/bold]
- Matrix × Vector: Treat vector as a single-column matrix
- Vector × Matrix: Treat vector as a single-row matrix
        """, title="Operation Guide", border_style="blue"))
        
        matrix_a = self.prompt_matrix("Enter first matrix (A)")
        if matrix_a is None:
            return
            
        matrix_b = self.prompt_matrix("Enter second matrix (B)")
        if matrix_b is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix_a = str(matrix_a.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.matrix_b = str(matrix_b.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.matrix_product, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_sum_series(self):
        self.display_header("Calculate Sum of Series")
        
        # Show help
        self.show_input_help("series")
        
        self.console.print(Panel("""[bold]Mathematical Series Calculation:[/bold]
        
This operation calculates the sum of a mathematical series.

[bold]Parameters:[/bold]
- Start index: The first index value
- End index: The last index value
- Formula: Mathematical expression using 'i' as the variable

[bold]Examples:[/bold]
- Sum from i=0 to 15 of (5+3i): [green]"5+3*i"[/green]
- Sum from i=0 to 9 of (50-5i): [green]"50-5*i"[/green]
- Sum from i=5 to 10 of i(i+1)/2: [green]"i*(i+1)/2"[/green]
- Sum from i=0 to 12 of (1+(i+3)²): [green]"1+(i+3)**2"[/green]
        """, title="Operation Guide", border_style="blue"))
        
        start = questionary.text(
            "Enter start index:",
            validate=lambda text: text.isdigit() or (text[0] == '-' and text[1:].isdigit()),
            style=custom_style
        ).ask()
        
        if start is None:
            return
            
        end = questionary.text(
            "Enter end index:",
            validate=lambda text: text.isdigit() or (text[0] == '-' and text[1:].isdigit()),
            style=custom_style
        ).ask()
        
        if end is None:
            return
        
        formula = questionary.text("Enter formula (using 'i' as the variable):", style=custom_style).ask()
        if formula is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.start = int(start)
        args.end = int(end)
        args.formula = formula
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.sum_series, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_check_particular_solution(self):
        self.display_header("Check Particular Solutions")
        
        self.console.print(Panel("""[bold]Checking Particular Solutions:[/bold]
        
This operation checks if vectors are solutions to a linear system.

[bold]Definitions:[/bold]
- Particular solution: A vector x such that Ax = b
- Homogeneous solution: A vector x such that Ax = 0

[bold]Input:[/bold]
- Augmented matrix [A|b] representing the system
- Vectors to check as potential solutions

[bold]Example:[/bold]
For system:
  4x + 4y - 16z = 4
  x + 0y - 4z = 2
  5x + 2y - 20z = 8

Matrix: [green]"4, 4, -16, 4; 1, 0, -4, 2; 5, 2, -20, 8"[/green]
Vector to check: [green]"2, -1, 0"[/green]

[bold]Output:[/bold]
- For each vector, shows if it's:
  * A particular solution (Ax = b)
  * A homogeneous solution (Ax = 0)
  * Not a solution
        """, title="Operation Guide", border_style="blue"))
        
        matrix = self.prompt_matrix("Enter augmented matrix [A|b]")
        if matrix is None:
            return
        
        self.console.print("[bold cyan]Enter vectors to check (empty input to finish)[/bold cyan]")
        vectors = []
        
        while True:
            vec = self.prompt_vector(f"Enter vector #{len(vectors) + 1} (or empty to finish)")
            if vec is None or (isinstance(vec, np.ndarray) and vec.size == 0):
                break
            vectors.append(" ".join(map(str, vec)))
        
        if not vectors:
            self.console.print("[yellow]No vectors to check.[/yellow]")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.vectors = vectors
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.check_particular_solution, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_point_plane_distance(self):
        self.display_header("Calculate Point-Plane Distance")
        
        self.console.print(Panel("""[bold]Point-Plane Distance:[/bold]
        
This operation calculates the shortest distance from a point to a plane.

[bold]Formula:[/bold] d = |n·(P-Q)| / |n|

Where:
- n is the normal vector to the plane
- Q is a point on the plane
- P is the point from which we want to find the distance
- · represents the dot product

[bold]Input:[/bold]
- Plane normal vector (n): [green]"a, b, c"[/green]
- Point on plane (Q): [green]"x, y, z"[/green]
- Point to find distance from (P): [green]"x, y, z"[/green]

[bold]Example:[/bold]
- Normal: [green]"4, 0, -3"[/green]
- Point on plane: [green]"5, 1, -2"[/green]
- Point to find distance from: [green]"10, 4, -12"[/green]
        """, title="Operation Guide", border_style="blue"))
        
        normal = self.prompt_vector("Enter plane normal vector")
        if normal is None:
            return
            
        plane_point = self.prompt_vector("Enter point on plane")
        if plane_point is None:
            return
            
        point = self.prompt_vector("Enter point to find distance from")
        if point is None:
            return
        
        class Args:
            pass
        
        args = Args()
        args.normal = " ".join(map(str, normal))
        args.plane_point = " ".join(map(str, plane_point))
        args.point = " ".join(map(str, point))
        
        # Use our redirection method to capture direct print output
        output, _ = self.run_with_capture(self.framework.point_plane_distance, args)
        
        # Display results in a nicely formatted panel with green border
        result_text = self.format_results_for_display(output)
        self.console.print(Panel(result_text, title="Results", border_style="green"))
        self.wait_for_user()
    
    def ui_browse_help_topics(self):
        """Display help topics and learning resources"""
        self.display_header("Linear Algebra Help Topics")
        
        topics = {
            "Vector Basics": """
[bold]Vector Representation and Components[/bold]
- A vector is an ordered list of numbers representing magnitude and direction
- In 2D: v = [x, y] represents a point or displacement
- In 3D: v = [x, y, z] represents a point or displacement in space
- Components can be real numbers, including decimals and negative values

[bold]Vector Addition and Subtraction[/bold]
- Add/subtract corresponding components
- Example: [1, 2] + [3, 4] = [4, 6]
- Geometric interpretation: Tip-to-tail method
- Properties: Commutative, Associative, Identity element [0, 0]

[bold]Scalar Multiplication[/bold]
- Multiply each component by a scalar
- Example: 2 * [1, 2] = [2, 4]
- Geometric interpretation: Scaling the vector
- Properties: Distributive, Associative

[bold]Vector Magnitude and Direction[/bold]
- Magnitude (length): |v| = √(x² + y² + z²)
- Direction: Unit vector in same direction
- Example: |[3, 4]| = 5
- Normalization: v/|v| gives unit vector

[bold]Dot Product[/bold]
- Algebraic: a·b = a₁b₁ + a₂b₂ + a₃b₃
- Geometric: a·b = |a||b|cos(θ)
- Properties: Commutative, Distributive
- Applications: Projections, angles, orthogonality

[bold]Cross Product (3D)[/bold]
- Result is perpendicular to both vectors
- |a × b| = |a||b|sin(θ)
- Direction follows right-hand rule
- Applications: Areas, volumes, torque
            """,
            "Matrix Operations": """
[bold]Matrix Representation[/bold]
- Rectangular array of numbers
- Dimensions: m × n (rows × columns)
- Element notation: aᵢⱼ (row i, column j)
- Special matrices: Identity, Zero, Diagonal

[bold]Matrix Addition/Subtraction[/bold]
- Add/subtract corresponding elements
- Matrices must have same dimensions
- Example: [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
- Properties: Commutative, Associative

[bold]Matrix Multiplication[/bold]
- (AB)ᵢⱼ = Σₖ aᵢₖbₖⱼ
- Number of columns of A must equal rows of B
- Result has dimensions: rows of A × columns of B
- Properties: Associative, Distributive
- Not commutative: AB ≠ BA in general

[bold]Transpose[/bold]
- Flip rows and columns: (A^T)ᵢⱼ = Aⱼᵢ
- Properties: (A^T)^T = A, (AB)^T = B^T A^T
- Symmetric matrix: A = A^T
- Skew-symmetric: A = -A^T

[bold]Determinants[/bold]
- 2×2: det([[a,b],[c,d]]) = ad - bc
- 3×3: Use cofactor expansion
- Properties: det(AB) = det(A)det(B)
- Applications: Inverses, areas, volumes

[bold]Matrix Inverse[/bold]
- AA⁻¹ = A⁻¹A = I
- Only square matrices can have inverses
- Not all square matrices have inverses
- 2×2 formula: [[a,b],[c,d]]⁻¹ = (1/det) * [[d,-b],[-c,a]]
            """,
            "Linear Systems": """
[bold]Systems of Linear Equations[/bold]
- General form: Ax = b
- A is coefficient matrix
- x is vector of variables
- b is right-hand side vector

[bold]Gaussian Elimination[/bold]
- Convert to row echelon form
- Use elementary row operations
- Back substitution to solve

[bold]Row Echelon Form[/bold]
- Leading coefficient (pivot) is 1
- Pivots move right as you go down
- Zeros below pivots
- Example:
  [[1,2,3],
   [0,1,4],
   [0,0,1]]

[bold]Reduced Row Echelon Form[/bold]
- Row echelon form plus:
- Zeros above pivots
- Pivots are only non-zero entries in their columns
- Example:
  [[1,0,0],
   [0,1,0],
   [0,0,1]]

[bold]Homogeneous Systems[/bold]
- Form: Ax = 0
- Always has solution x = 0
- May have non-trivial solutions
- Solution space is null space of A

[bold]Particular Solutions[/bold]
- Any solution to Ax = b
- Can be found by:
  * Gaussian elimination
  * Matrix inverse (if A is invertible)
  * Cramer's rule

[bold]Free Variables[/bold]
- Variables not corresponding to pivots
- Can be assigned arbitrary values
- Lead to parametric solutions
- Number = n - rank(A)
            """,
            "Geometric Applications": """
[bold]Lines in 3D[/bold]
- Parametric form: r = r₀ + tv
- r₀ is point on line
- v is direction vector
- t is parameter
- Example: r = [1,2,3] + t[4,5,6]

[bold]Planes in 3D[/bold]
- Point-normal form: n·(r - r₀) = 0
- n is normal vector
- r₀ is point on plane
- General form: ax + by + cz + d = 0
- Example: 2x + 3y + 4z = 5

[bold]Distance Calculations[/bold]
- Point to line: d = |v × (P-P₀)|/|v|
- Point to plane: d = |n·(P-P₀)|/|n|
- Line to line: d = |(P₂-P₁)·(v₁×v₂)|/|v₁×v₂|
- Example: Distance from [1,2,3] to plane 2x+3y+4z=5

[bold]Projections[/bold]
- Vector onto vector: projᵥu = (u·v/|v|²)v
- Vector onto plane: u - projₙu
- Example: Project [1,2,3] onto [4,5,6]

[bold]Areas and Volumes[/bold]
- Triangle area: |AB × AC|/2
- Parallelogram area
- Parallelepiped volume
- Use determinants

[bold]Intersections[/bold]
- Line-plane intersection
- Plane-plane intersection
- Line-line intersection
- Use parametric equations

[bold]Orthogonality[/bold]
- Vectors: a·b = 0
- Lines: direction vectors are perpendicular
- Planes: normal vectors are perpendicular
- Example: [1,0] and [0,1] are orthogonal
            """
        }
        
        topic = questionary.select(
            "Select a topic to learn about:",
            choices=list(topics.keys()),
            style=custom_style
        ).ask()
        
        if topic:
            self.console.print(Panel(topics[topic], title=topic, border_style="blue"))
            self.wait_for_user()
    
    def ui_show_example_exercises(self):
        """Display example exercises with solutions"""
        self.display_header("Example Exercises")
        
        examples = {
            "Vector Operations": """
[bold]Example 1: Vector Addition and Subtraction[/bold]
Given: a = [1, 2, 3], b = [4, 5, 6]
Find: a + b and a - b

Solution:
a + b = [1+4, 2+5, 3+6] = [5, 7, 9]
a - b = [1-4, 2-5, 3-6] = [-3, -3, -3]

[bold]Example 2: Dot Product and Angle[/bold]
Given: a = [1, 2, 3], b = [4, 5, 6]
Find: a·b and the angle between a and b

Solution:
a·b = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
|a| = √(1² + 2² + 3²) = √14
|b| = √(4² + 5² + 6²) = √77
cos(θ) = (a·b)/(|a||b|) = 32/(√14 * √77)
θ ≈ 0.225 radians ≈ 12.9 degrees

[bold]Example 3: Cross Product[/bold]
Given: a = [1, 0, 0], b = [0, 1, 0]
Find: a × b

Solution:
a × b = [0*0 - 0*1, 0*0 - 1*0, 1*1 - 0*0]
      = [0, 0, 1]

[bold]Example 4: Vector Projection[/bold]
Given: a = [3, -4], b = [12, -1]
Find: projᵦa

Solution:
projᵦa = (a·b/|b|²)b
a·b = 3*12 + (-4)*(-1) = 36 + 4 = 40
|b|² = 12² + (-1)² = 144 + 1 = 145
projᵦa = (40/145)[12, -1] ≈ [3.31, -0.28]
            """,
            "Matrix Operations": """
[bold]Example 1: Matrix Multiplication[/bold]
Given: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
Find: AB

Solution:
AB = [[1*5 + 2*7, 1*6 + 2*8],
      [3*5 + 4*7, 3*6 + 4*8]]
   = [[19, 22],
      [43, 50]]

[bold]Example 2: Gaussian Elimination[/bold]
Given: System:
  2x + y = 5
  3x + 2y = 8
Find: Solution

Solution:
Augmented matrix: [[2, 1, 5], [3, 2, 8]]
Step 1: R₂ - (3/2)R₁ → [[2, 1, 5], [0, 1/2, 1/2]]
Step 2: 2R₂ → [[2, 1, 5], [0, 1, 1]]
Step 3: R₁ - R₂ → [[2, 0, 4], [0, 1, 1]]
Step 4: (1/2)R₁ → [[1, 0, 2], [0, 1, 1]]
Therefore: x = 2, y = 1

[bold]Example 3: Determinant and Inverse[/bold]
Given: A = [[1, 2], [3, 4]]
Find: det(A) and A⁻¹

Solution:
det(A) = 1*4 - 2*3 = 4 - 6 = -2
A⁻¹ = (1/-2) * [[4, -2], [-3, 1]]
    = [[-2, 1], [1.5, -0.5]]

[bold]Example 4: Eigenvalues[/bold]
Given: A = [[1, 2], [3, 4]]
Find: Eigenvalues and eigenvectors

Solution:
Characteristic equation: det(A - λI) = 0
det([[1-λ, 2], [3, 4-λ]]) = 0
(1-λ)(4-λ) - 6 = 0
λ² - 5λ - 2 = 0
λ = (5 ± √33)/2
Eigenvalues: λ₁ ≈ 5.37, λ₂ ≈ -0.37
            """,
            "Geometric Problems": """
[bold]Example 1: Point-Line Distance[/bold]
Given: Point P(1, 2, 3), Line through A(0, 0, 0) with direction v[1, 1, 1]
Find: Distance from P to line

Solution:
d = |v × (P-A)|/|v|
P-A = [1, 2, 3]
v × (P-A) = [1, -2, 1]
|v × (P-A)| = √6
|v| = √3
d = √6/√3 = √2 ≈ 1.414

[bold]Example 2: Plane Intersection[/bold]
Given: Planes:
  x + y + z = 1
  2x + 3y + 4z = 5
Find: Intersection

Solution:
Augmented matrix: [[1, 1, 1, 1], [2, 3, 4, 5]]
RREF: [[1, 0, -1, -2], [0, 1, 2, 3]]
Parametric solution:
x = -2 + t
y = 3 - 2t
z = t
Direction vector: [1, -2, 1]
Point on line: [-2, 3, 0]

[bold]Example 3: Triangle Area[/bold]
Given: Points A(0, 0), B(3, 0), C(0, 4)
Find: Area

Solution:
AB = [3, 0]
AC = [0, 4]
Cross product magnitude = |3*4 - 0*0| = 12
Area = 12/2 = 6 square units

[bold]Example 4: Point-Plane Distance[/bold]
Given: Plane 2x + 3y - z = 5, Point P(1, 2, 3)
Find: Distance from P to plane

Solution:
Normal vector n = [2, 3, -1]
Point on plane Q(0, 0, -5)
d = |n·(P-Q)|/|n|
P-Q = [1, 2, 8]
n·(P-Q) = 0
|n| = √14
d = 0/√14 = 0 (point lies on plane)
            """
        }
        
        category = questionary.select(
            "Select a category of examples:",
            choices=list(examples.keys()),
            style=custom_style
        ).ask()
        
        if category:
            self.console.print(Panel(examples[category], title=category, border_style="blue"))
            self.wait_for_user()
    
    def ui_linear_algebra_guide(self):
        """Display a comprehensive guide to linear algebra concepts"""
        self.display_header("Linear Algebra Guide")
        
        guide = """[bold]Linear Algebra: A Comprehensive Guide[/bold]

[bold]1. Vector Spaces[/bold]
[bold]Definition and Properties[/bold]
- A vector space is a set of vectors with operations of addition and scalar multiplication
- Must satisfy 10 axioms (closure, associativity, commutativity, etc.)
- Examples: R², R³, Rⁿ, space of polynomials, space of matrices

[bold]Subspaces[/bold]
- A subset of a vector space that is itself a vector space
- Must be closed under addition and scalar multiplication
- Examples: Lines through origin, planes through origin
- Null space and column space of a matrix are subspaces

[bold]Linear Independence[/bold]
- A set of vectors is linearly independent if no vector can be written as a linear combination of others
- Test: Set up equation c₁v₁ + c₂v₂ + ... + cₙvₙ = 0
- If only solution is c₁ = c₂ = ... = cₙ = 0, vectors are independent
- Maximum number of independent vectors = dimension of space

[bold]Basis and Dimension[/bold]
- Basis: Set of linearly independent vectors that span the space
- Dimension: Number of vectors in any basis
- Standard basis: [1,0,0], [0,1,0], [0,0,1] for R³
- Change of basis: Use transition matrix

[bold]Orthogonal and Orthonormal Bases[/bold]
- Orthogonal: All vectors are perpendicular
- Orthonormal: Orthogonal and all vectors have length 1
- Gram-Schmidt process to create orthonormal basis
- QR decomposition of matrices

[bold]2. Linear Transformations[/bold]
[bold]Definition and Properties[/bold]
- Function T: V → W that preserves vector operations
- T(u + v) = T(u) + T(v)
- T(cu) = cT(u)
- Examples: Rotation, reflection, projection

[bold]Matrix Representation[/bold]
- Every linear transformation can be represented by a matrix
- Columns are images of basis vectors
- Composition of transformations = matrix multiplication
- Change of basis affects matrix representation

[bold]Kernel and Image[/bold]
- Kernel: Set of vectors mapped to zero
- Image: Set of all possible outputs
- Rank-nullity theorem: dim(ker) + dim(im) = dim(domain)
- Applications to solving systems of equations

[bold]Eigenvalues and Eigenvectors[/bold]
- λ is eigenvalue if Av = λv for some v ≠ 0
- v is eigenvector corresponding to λ
- Characteristic polynomial: det(A - λI) = 0
- Applications: Diagonalization, stability analysis

[bold]3. Matrix Operations[/bold]
[bold]Basic Operations[/bold]
- Addition: Element-wise
- Multiplication: Dot product of rows and columns
- Transpose: Flip rows and columns
- Properties: Associative, distributive, not commutative

[bold]Special Matrices[/bold]
- Identity: Iᵢⱼ = 1 if i=j, 0 otherwise
- Diagonal: Non-zero only on main diagonal
- Symmetric: A = A^T
- Orthogonal: A^T A = I

[bold]Determinants[/bold]
- Scalar value associated with square matrix
- Properties: det(AB) = det(A)det(B)
- det(A) = 0 iff A is singular
- Applications: Volume, area, solving systems

[bold]Matrix Inverse[/bold]
- A⁻¹ exists iff det(A) ≠ 0
- AA⁻¹ = A⁻¹A = I
- (AB)⁻¹ = B⁻¹A⁻¹
- Methods: Adjugate, row operations

[bold]4. Systems of Linear Equations[/bold]
[bold]Gaussian Elimination[/bold]
- Convert to row echelon form
- Use elementary row operations
- Back substitution to solve
- Pivoting for numerical stability

[bold]Row Echelon Form[/bold]
- Leading coefficient (pivot) is 1
- Pivots move right as you go down
- Zeros below pivots
- Reduced form: Zeros above pivots

[bold]Solution Spaces[/bold]
- Unique solution: No free variables
- Infinitely many solutions: Free variables
- No solution: Inconsistent system
- Parametric form of solution

[bold]5. Inner Product Spaces[/bold]
[bold]Dot Product[/bold]
- Algebraic: a·b = Σaᵢbᵢ
- Geometric: a·b = |a||b|cos(θ)
- Properties: Commutative, distributive
- Applications: Projections, angles

[bold]Orthogonality[/bold]
- Vectors are orthogonal if a·b = 0
- Orthogonal complement
- Projection onto subspace
- Least squares approximation

[bold]6. Applications[/bold]
[bold]Computer Graphics[/bold]
- Rotation matrices
- Projection matrices
- Transformation composition
- 3D rendering

[bold]Quantum Mechanics[/bold]
- State vectors
- Hermitian operators
- Eigenvalue problems
- Wave functions

[bold]Machine Learning[/bold]
- Principal Component Analysis
- Singular Value Decomposition
- Linear regression
- Feature transformation

[bold]Network Analysis[/bold]
- Adjacency matrices
- Graph theory
- PageRank algorithm
- Social network analysis

[bold]Optimization[/bold]
- Linear programming
- Quadratic programming
- Constraint optimization
- Gradient descent
"""
        
        self.console.print(Panel(guide, title="Linear Algebra Guide", border_style="blue"))
        self.wait_for_user()
    
    def ui_common_homework_patterns(self):
        """Display common patterns in linear algebra homework problems"""
        self.display_header("Common Homework Patterns")
        
        patterns = """[bold]Common Linear Algebra Homework Patterns[/bold]

[bold]1. Vector Operations[/bold]
[bold]Finding Unit Vectors[/bold]
- Given vector v, find v/|v|
- Example: v = [3, 4] → [3/5, 4/5]
- Common in physics and computer graphics
- Used in normalization and direction calculations

[bold]Computing Projections[/bold]
- Project vector a onto vector b
- Formula: projᵦa = (a·b/|b|²)b
- Applications: Force decomposition, shadow calculations
- Related to least squares approximation

[bold]Checking Orthogonality[/bold]
- Test if vectors are perpendicular
- Check if dot product is zero
- Example: [1, 0] and [0, 1] are orthogonal
- Applications: Basis construction, coordinate systems

[bold]Calculating Angles[/bold]
- Use dot product formula: cos(θ) = (a·b)/(|a||b|)
- Convert between radians and degrees
- Example: angle between [1, 1] and [1, 0] is π/4
- Applications: Geometry, physics, computer graphics

[bold]Cross Products[/bold]
- Only defined in 3D
- Result is perpendicular to both vectors
- Magnitude gives area of parallelogram
- Applications: Torque, angular momentum

[bold]2. Matrix Operations[/bold]
[bold]Matrix Multiplication[/bold]
- Check dimensions match
- Use row-column method
- Properties: Associative, distributive
- Not commutative in general

[bold]Finding Determinants[/bold]
- 2×2: ad - bc
- 3×3: Use cofactor expansion
- Properties: det(AB) = det(A)det(B)
- Applications: Inverses, areas, volumes

[bold]Computing Inverses[/bold]
- Check if matrix is invertible
- Use adjugate method or row operations
- Verify AA⁻¹ = I
- Applications: Solving systems, transformations

[bold]Gaussian Elimination[/bold]
- Convert to row echelon form
- Use back substitution
- Identify free variables
- Write parametric solution

[bold]3. Linear Systems[/bold]
[bold]Unique Solutions[/bold]
- System has exactly one solution
- No free variables
- Matrix is invertible
- Example: 2x + y = 5, 3x + 2y = 8

[bold]Parametric Solutions[/bold]
- System has infinitely many solutions
- Free variables present
- Write solution in terms of parameters
- Example: x = 2 + t, y = 1 - 2t

[bold]Consistency Checking[/bold]
- System has no solution
- Inconsistent equations
- Row of zeros with non-zero constant
- Example: x + y = 1, x + y = 2

[bold]4. Eigenvalue Problems[/bold]
[bold]Finding Eigenvalues[/bold]
- Solve characteristic equation
- det(A - λI) = 0
- Factor polynomial
- Find roots

[bold]Finding Eigenvectors[/bold]
- For each eigenvalue λ
- Solve (A - λI)v = 0
- Find basis for null space
- Normalize if needed

[bold]Diagonalization[/bold]
- Find eigenvalues and eigenvectors
- Construct P and D matrices
- Verify A = PDP⁻¹
- Applications: Powers of matrices

[bold]5. Geometric Applications[/bold]
[bold]Distance Calculations[/bold]
- Point to line
- Point to plane
- Line to line
- Use vector formulas

[bold]Area and Volume[/bold]
- Triangle area using cross product
- Parallelogram area
- Parallelepiped volume
- Use determinants

[bold]Intersections[/bold]
- Line-plane intersection
- Plane-plane intersection
- Line-line intersection
- Use parametric equations

[bold]6. Proof Problems[/bold]
[bold]Vector Space Properties[/bold]
- Verify closure under addition
- Verify closure under scalar multiplication
- Check all 10 axioms
- Use properties of real numbers

[bold]Linear Independence[/bold]
- Set up equation c₁v₁ + ... + cₙvₙ = 0
- Show only solution is cᵢ = 0
- Use properties of vectors
- May need to use contradiction

[bold]Subspace Verification[/bold]
- Check non-empty
- Verify closure under addition
- Verify closure under scalar multiplication
- May need to find basis

[bold]Matrix Properties[/bold]
- Prove properties of operations
- Use definition of operations
- Apply properties of real numbers
- May need to use induction
"""
        
        self.console.print(Panel(patterns, title="Common Homework Patterns", border_style="blue"))
        self.wait_for_user()
    
    def ui_recognize_problem_types(self):
        """Help users recognize different types of linear algebra problems"""
        self.display_header("Problem Type Recognition")
        
        self.console.print(Panel("""[bold]How to Recognize Linear Algebra Problems[/bold]

This guide will help you identify the type of problem you're dealing with and choose the appropriate solution method.

[bold]1. Vector Problems[/bold]
[bold]Points, Directions, Magnitudes[/bold]
- Problem mentions coordinates or points
- Involves distances or lengths
- Uses terms like "direction", "magnitude", "unit vector"
- If you need to find angles, distances, or projections
- If the problem mentions perpendicularity or parallel lines
- If you need to compute areas or volumes

[bold]2. Matrix Problems[/bold]
[bold]Grid of Numbers[/bold]
- Problem shows a grid or array of numbers
- Involves rows and columns
- Uses terms like "matrix", "array", "table"
- If you need to perform operations on multiple equations
- If the problem involves transformations
- If you need to find determinants or inverses

[bold]3. System of Equations[/bold]
[bold]Multiple Variables[/bold]
- Problem has several variables
- Involves multiple equations
- Uses terms like "solve", "find", "determine"
- If you need to find values that satisfy all equations
- If the problem involves consistency or uniqueness
- If you need to find parametric solutions

[bold]4. Eigenvalue Problems[/bold]
- If the problem involves repeated transformations
- If you need to find special vectors that don't change direction
- If the problem mentions stability or oscillations
- If you need to diagonalize a matrix

[bold]5. Geometric Problems[/bold]
- If the problem involves points, lines, or planes
- If you need to find intersections or distances
- If the problem mentions areas or volumes
- If you need to find projections or shadows
""", title="Problem Recognition Guide", border_style="blue"))
        
        # Offer an interactive quiz
        if questionary.confirm("Would you like to take a problem recognition quiz?", style=custom_style).ask():
            self.problem_recognition_quiz()
        
        self.wait_for_user()
    
    def problem_recognition_quiz(self):
        """Interactive quiz to help users recognize problem types"""
        questions = [
            {
                "question": "Given vectors a = [1, 2, 3] and b = [4, 5, 6], find a·b",
                "type": "Vector",
                "explanation": "This is a vector problem because it involves the dot product of two vectors. The problem gives vectors in component form and asks for a vector operation (dot product)."
            },
            {
                "question": "Solve the system: 2x + y = 5, 3x + 2y = 8",
                "type": "System",
                "explanation": "This is a system of equations problem because it involves multiple equations with multiple variables. The problem asks to find values that satisfy all equations simultaneously."
            },
            {
                "question": "Find the eigenvalues of matrix A = [[1, 2], [3, 4]]",
                "type": "Eigenvalue",
                "explanation": "This is an eigenvalue problem because it involves finding special values that satisfy the characteristic equation. The problem gives a matrix and asks for an eigenvalue operation."
            },
            {
                "question": "Calculate the area of the triangle with vertices (0,0), (3,0), and (0,4)",
                "type": "Geometric",
                "explanation": "This is a geometric problem because it involves calculating the area of a shape defined by points. The problem mentions geometric objects (triangle) and measurements (area)."
            },
            {
                "question": "Find the inverse of matrix A = [[1, 2], [3, 4]]",
                "type": "Matrix",
                "explanation": "This is a matrix problem because it involves operations on a matrix to find its inverse. The problem gives a matrix and asks for a matrix operation (inverse)."
            },
            {
                "question": "Determine if the vectors [1, 2, 3] and [4, 5, 6] are orthogonal",
                "type": "Vector",
                "explanation": "This is a vector problem because it involves checking a relationship between vectors (orthogonality). The problem gives vectors and asks about their geometric relationship."
            },
            {
                "question": "Find the projection of vector [1, 2, 3] onto [4, 5, 6]",
                "type": "Vector",
                "explanation": "This is a vector problem because it involves vector projection. The problem gives vectors and asks for a vector operation (projection)."
            },
            {
                "question": "Solve the matrix equation AX = B for X, where A and B are given matrices",
                "type": "Matrix",
                "explanation": "This is a matrix problem because it involves solving a matrix equation. The problem mentions matrices and matrix operations."
            },
            {
                "question": "Find the distance from point (1, 2, 3) to the plane 2x+3y-z=5",
                "type": "Geometric",
                "explanation": "This is a geometric problem because it involves calculating a distance between a point and a geometric object (plane)."
            },
            {
                "question": "Determine if the system has a unique solution, infinitely many solutions, or no solution",
                "type": "System",
                "explanation": "This is a system of equations problem because it involves analyzing the solution set of a system of equations. The problem asks about the nature of solutions."
            }
        ]
        
        score = 0
        for i, q in enumerate(questions, 1):
            self.console.print(f"\n[bold]Question {i}:[/bold]")
            self.console.print(q["question"])
            
            answer = questionary.select(
                "What type of problem is this?",
                choices=["Vector", "Matrix", "System", "Eigenvalue", "Geometric"],
                style=custom_style
            ).ask()
            
            if answer == q["type"]:
                score += 1
                self.console.print("[green]Correct![/green]")
            else:
                self.console.print("[red]Incorrect![/red]")
            
            self.console.print(f"Explanation: {q['explanation']}")
            self.wait_for_user()
        
        self.console.print(f"\n[bold]Final Score: {score}/{len(questions)}[/bold]")
        if score == len(questions):
            self.console.print("[green]Perfect! You're great at recognizing problem types![/green]")
        elif score >= len(questions) * 0.7:
            self.console.print("[yellow]Good job! You're getting better at recognizing problem types.[/yellow]")
        else:
            self.console.print("[red]Keep practicing! Review the problem recognition guide.[/red]")
    
    def run(self):
        parser = argparse.ArgumentParser(description='Linear Algebra Exercise Framework with Rich UI')
        parser.add_argument('--version', action='version', version='Linear Algebra Framework 1.0')
        parser.add_argument('--no-color', action='store_true', help='Disable colored output')
        
        args = parser.parse_args()
        
        if args.no_color:
            self.console = Console(color_system=None)
        
        self.run_menu()


def main():
    """Main entry point with welcome message"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Linear Algebra Exercise Framework with Rich UI')
    parser.add_argument('--version', action='version', version='Linear Algebra Framework 1.0')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--no-banner', action='store_true', help='Skip displaying the banner')
    
    args = parser.parse_args()
    
    # Create UI instance
    ui = LinearAlgebraRichUI()
    
    if args.no_color:
        ui.console = Console(color_system=None)
    
    # Show welcome message
    if not args.no_banner:
        ui.clear_screen()
        ui.console.print(ui.banner)
        
        # Print welcome message
        ui.console.print("[bold green]Welcome to the Linear Algebra Exercise Framework![/bold green]")
        ui.console.print("This tool helps you practice and visualize linear algebra concepts.")
        ui.console.print("Starting up the interactive interface...")
        ui.console.print()
        
        # A short pause for the user to see the banner
        try:
            from time import sleep
            sleep(1.0)
        except:
            pass
    
    # Run the UI
    ui.run()


if __name__ == "__main__":
    main()