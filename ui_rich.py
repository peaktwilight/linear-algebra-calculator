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
        
        # Define categories and operations
        self.categories = {
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
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self, title="Linear Algebra Exercise Framework"):
        self.clear_screen()
        self.console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="blue"))
        self.console.print()
    
    def print_result(self, result, title="Result"):
        self.console.print(Panel(result, title=title, border_style="green"))
    
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
        
        with self.console.capture() as capture:
            self.framework.polar_to_cartesian(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.normalize_vector(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.vector_direction_length(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.vector_shadow(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.check_orthogonal(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.vector_angle(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.cross_product(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.triangle_area(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.point_line_distance(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.check_collinear(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.solve_gauss(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.check_coplanar(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.check_solution(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.vector_equation(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.matrix_element(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.intersection_planes(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.homogeneous_intersection(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.find_pivot_free_vars(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.matrix_operations(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.matrix_product(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.sum_series(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.check_particular_solution(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
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
        
        with self.console.capture() as capture:
            self.framework.point_plane_distance(args)
        
        self.console.print(Panel(capture.get(), title="Results", border_style="green"))
        self.wait_for_user()
    
    def run(self):
        parser = argparse.ArgumentParser(description='Linear Algebra Exercise Framework with Rich UI')
        parser.add_argument('--version', action='version', version='Linear Algebra Framework 1.0')
        parser.add_argument('--no-color', action='store_true', help='Disable colored output')
        
        args = parser.parse_args()
        
        if args.no_color:
            self.console = Console(color_system=None)
        
        self.run_menu()


def main():
    ui = LinearAlgebraRichUI()
    ui.run()


if __name__ == "__main__":
    main()