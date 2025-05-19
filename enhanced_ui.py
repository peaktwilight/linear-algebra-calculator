#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced UI for Linear Algebra Framework with Help System
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse
import random
import numpy as np
from linalg_cli import LinearAlgebraExerciseFramework
from help_guides import (
    display_topic_help, 
    list_help_topics, 
    OPERATION_TO_HELP, 
    generate_exercise,
    COMMON_QUESTIONS
)

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

class EnhancedLinearAlgebraUI:
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
            ],
            "Learning Resources": [
                ("Browse help topics", self.ui_browse_help_topics),
                ("Show example exercises", self.ui_show_example_exercises),
                ("Linear algebra quick guide", self.ui_linear_algebra_guide),
                ("Common homework patterns", self.ui_common_homework_patterns),
                ("How to recognize problem types", self.ui_recognize_problem_types),
            ]
        }
        
        # Map operation names to help topics
        self.operation_to_help = OPERATION_TO_HELP
    
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
    
    def show_help_for_operation(self, operation_name):
        """Show contextual help for the current operation"""
        if operation_name in self.operation_to_help:
            help_topic = self.operation_to_help[operation_name]
            return display_topic_help(self.console, help_topic)
        else:
            self.console.print(f"[yellow]No specific help available for this operation.[/yellow]")
            return False
    
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
        
        # Add help button to the prompt
        message = f"{message} (type ? for help)"
        
        while True:
            try:
                vector_str = questionary.text(
                    message,
                    style=custom_style
                ).ask()
                
                # Handle cancellation
                if vector_str is None:
                    return None
                
                # Check for help request
                if vector_str == "?":
                    self.show_input_help("vector")
                    continue
                
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
        self.console.print("[yellow]Type ? and press Enter for help[/yellow]")
        
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
                    # Show contextual help option before operation
                    for key, value in self.operation_to_help.items():
                        if key in op_func.__name__:
                            show_help = questionary.confirm(
                                f"Would you like to see an explanation about {op_name.lower()} before starting?", 
                                style=custom_style
                            ).ask()
                            if show_help:
                                display_topic_help(self.console, value)
                                self.wait_for_user()
                    
                    # Execute the operation
                    op_func()
                    break
    
    # Learning resource functions
    def ui_browse_help_topics(self):
        """Browse all available help topics"""
        self.display_header("Help Topics")
        
        # List all help topics
        list_help_topics(self.console)
        
        # Allow user to select a topic
        topics = list(COMMON_QUESTIONS.keys())
        topics.append("Back to menu")
        
        while True:
            topic = questionary.select(
                "Select a topic to learn more:",
                choices=topics,
                style=custom_style
            ).ask()
            
            if topic == "Back to menu" or topic is None:
                return
            
            # Display the selected topic
            self.display_header(f"Help: {COMMON_QUESTIONS[topic]['title']}")
            display_topic_help(self.console, topic)
            
            # Ask if user wants to see another topic
            continue_browsing = questionary.confirm("View another topic?", style=custom_style).ask()
            if not continue_browsing:
                return
    
    def ui_show_example_exercises(self):
        """Show example exercises with solutions"""
        self.display_header("Example Exercises")
        
        exercise_types = list(generate_exercise("polar_to_cartesian").keys())
        exercise_types.extend([
            "normalize_vector", 
            "check_orthogonal", 
            "vector_angle", 
            "solve_linear_system",
            "matrix_product"
        ])
        exercise_types.append("Back to menu")
        
        while True:
            exercise_type = questionary.select(
                "Select exercise type:",
                choices=[t for t in exercise_types if t != "Back to menu"],
                style=custom_style
            ).ask()
            
            if exercise_type == "Back to menu" or exercise_type is None:
                return
            
            # Generate and display an example
            exercise_data = generate_exercise(exercise_type)
            
            self.console.print(Panel(
                exercise_data["exercise"],
                title="Example Exercise",
                border_style="cyan"
            ))
            
            show_solution = questionary.confirm("Show solution?", style=custom_style).ask()
            if show_solution:
                self.console.print(Panel(
                    Markdown(exercise_data["solution"]),
                    title="Step-by-Step Solution",
                    border_style="green"
                ))
            
            # Ask if user wants to see another example
            continue_browsing = questionary.confirm("View another example?", style=custom_style).ask()
            if not continue_browsing:
                return
    
    def ui_linear_algebra_guide(self):
        """Display a comprehensive guide to linear algebra concepts"""
        self.display_header("Linear Algebra Quick Guide")
        
        # Create a table of contents
        toc = Table(title="Linear Algebra Concepts", box=None)
        toc.add_column("Topic", style="cyan")
        toc.add_column("Description", style="green")
        
        # Core topics in recommended learning order
        core_topics = [
            ("vector_basics", "Understanding vectors and their representation"),
            ("vector_operations", "Basic vector operations (addition, scalar multiplication, etc.)"),
            ("vector_applications", "Applications of vectors (projections, angles, etc.)"),
            ("matrices", "Introduction to matrices and their properties"),
            ("matrix_operations", "Matrix operations (addition, multiplication, etc.)"),
            ("linear_systems", "Systems of linear equations"),
            ("gauss_elimination", "Solving systems with Gaussian elimination"),
            ("parametric_solutions", "Parametric solutions and free variables"),
            ("linear_independence", "Linear independence and dependence"),
            ("planes_intersection", "Intersection of planes in 3D space"),
            ("geometric_interpretation", "Geometric interpretation of linear algebra"),
            ("practical_tips", "Practical tips for solving problems")
        ]
        
        for topic_id, description in core_topics:
            toc.add_row(COMMON_QUESTIONS[topic_id]["title"], description)
        
        self.console.print(toc)
        
        # Allow user to select a topic
        topic_choices = [topic_id for topic_id, _ in core_topics]
        topic_choices.append("Back to menu")
        
        while True:
            topic_index = questionary.select(
                "Select a topic to explore:",
                choices=[COMMON_QUESTIONS[t]["title"] for t in topic_choices if t != "Back to menu"],
                style=custom_style
            ).ask()
            
            if topic_index == "Back to menu" or topic_index is None:
                return
            
            # Find the topic ID
            selected_topic = None
            for topic_id, _ in core_topics:
                if COMMON_QUESTIONS[topic_id]["title"] == topic_index:
                    selected_topic = topic_id
                    break
            
            if selected_topic:
                self.display_header(f"Guide: {COMMON_QUESTIONS[selected_topic]['title']}")
                display_topic_help(self.console, selected_topic)
                
                # Ask if user wants to see another topic
                continue_browsing = questionary.confirm("Explore another topic?", style=custom_style).ask()
                if not continue_browsing:
                    return
    
    def ui_common_homework_patterns(self):
        """Display common homework problem patterns and solution strategies"""
        self.display_header("Common Homework Problem Patterns")
        
        # Create a table of common problem types
        problem_types = [
            ("Converting between coordinate systems", "polar_to_cartesian"),
            ("Finding unit vectors", "vector_normalization"),
            ("Calculating angles between vectors", "vector_applications"),
            ("Determining if vectors are orthogonal", "vector_applications"),
            ("Computing cross products", "vector_operations"),
            ("Solving systems of linear equations", "gauss_elimination"),
            ("Checking if vectors are linearly independent", "linear_independence"),
            ("Finding plane intersections", "planes_intersection"),
            ("Determining parametric solutions", "parametric_solutions"),
            ("Matrix multiplication problems", "matrix_operations")
        ]
        
        problems_table = Table()
        problems_table.add_column("Problem Type", style="cyan")
        problems_table.add_column("Approach Strategy", style="green")
        
        for problem_type, topic in problem_types:
            # Get a short strategy from the help topic
            strategy = COMMON_QUESTIONS[topic]["explanation"].split("\n")[0:3]
            strategy = "\n".join(strategy)
            
            problems_table.add_row(problem_type, strategy)
        
        self.console.print(problems_table)
        
        # Allow user to select a problem type for detailed help
        problem_choices = [problem_type for problem_type, _ in problem_types]
        problem_choices.append("Back to menu")
        
        while True:
            selected_problem = questionary.select(
                "Select a problem type for detailed help:",
                choices=problem_choices[:-1],  # Exclude "Back to menu"
                style=custom_style
            ).ask()
            
            if selected_problem == "Back to menu" or selected_problem is None:
                return
            
            # Find the associated topic
            topic = None
            for p, t in problem_types:
                if p == selected_problem:
                    topic = t
                    break
            
            if topic:
                self.display_header(f"Help: {selected_problem}")
                display_topic_help(self.console, topic)
                
                # Ask if user wants to see another problem type
                continue_browsing = questionary.confirm("View another problem type?", style=custom_style).ask()
                if not continue_browsing:
                    return
    
    def ui_recognize_problem_types(self):
        """Help users recognize different types of linear algebra problems"""
        self.display_header("Recognizing Problem Types")
        
        # Create a panel with problem type recognition tips
        recognition_tips = """
[bold cyan]How to Recognize Problem Types from Question Wording:[/bold cyan]

[bold]"Convert from polar to Cartesian"[/bold]
- Contains terms like: polar coordinates, r and theta, angle and distance
- Solution: Use [green]polar_to_cartesian[/green] with formulas x = r·cos(θ), y = r·sin(θ)

[bold]"Find a unit vector"[/bold]
- Contains terms like: unit vector, normalize, same direction
- Solution: Use [green]normalize_vector[/green] with v̂ = v/|v|

[bold]"Calculate the angle between"[/bold]
- Contains terms like: angle, vectors, find theta
- Solution: Use [green]vector_angle[/green] with formula θ = arccos((a·b)/(|a|·|b|))

[bold]"Are these vectors orthogonal/perpendicular?"[/bold]
- Contains terms like: orthogonal, perpendicular, right angle
- Solution: Use [green]check_orthogonal[/green] and check if a·b = 0

[bold]"Find the projection/shadow"[/bold]
- Contains terms like: projection, shadow, component in direction
- Solution: Use [green]vector_shadow[/green] with formulas for scalar/vector projection

[bold]"Solve the system of linear equations"[/bold]
- Contains multiple equations with variables
- Solution: Use [green]solve_gauss[/green] with Gaussian elimination

[bold]"Are these vectors linearly independent?"[/bold]
- Contains terms like: linearly independent/dependent, span, basis
- Solution: Use [green]check_collinear[/green] or [green]check_coplanar[/green]

[bold]"Find where these planes intersect"[/bold]
- Contains multiple plane equations
- Solution: Use [green]intersection_planes[/green] or [green]homogeneous_intersection[/green]

[bold]"Find the distance from a point to a plane/line"[/bold]
- Contains terms like: distance, point, plane/line
- Solution: Use [green]point_plane_distance[/green] or [green]point_line_distance[/green]

[bold]"Perform matrix operations"[/bold]
- Contains terms like: add matrices, multiply matrices, transpose
- Solution: Use [green]matrix_operations[/green] or [green]matrix_product[/green]
"""
        
        self.console.print(Panel(
            recognition_tips,
            title="Problem Recognition Guide",
            border_style="blue"
        ))
        
        # Offer a quiz to practice problem recognition
        take_quiz = questionary.confirm("Would you like to take a quick quiz to practice recognizing problem types?", style=custom_style).ask()
        
        if take_quiz:
            self.problem_recognition_quiz()
        
        self.wait_for_user()
    
    def problem_recognition_quiz(self):
        """Interactive quiz for recognizing problem types"""
        self.display_header("Problem Type Recognition Quiz")
        
        quiz_questions = [
            {
                "question": "Convert the polar coordinates (r=4, θ=30°) to Cartesian coordinates.",
                "type": "polar_to_cartesian",
                "function": "polar_to_cartesian"
            },
            {
                "question": "Find a unit vector in the direction of v = (3, 4, 0).",
                "type": "Normalization/Unit Vector",
                "function": "normalize_vector"
            },
            {
                "question": "Calculate the angle between vectors a = (1, 0, 1) and b = (0, 1, 1).",
                "type": "Vector Angle",
                "function": "vector_angle"
            },
            {
                "question": "Determine if the vectors u = (2, -1, 3) and v = (1, 2, 0) are orthogonal.",
                "type": "Orthogonality Check",
                "function": "check_orthogonal"
            },
            {
                "question": "Find the projection of vector b = (4, 2) onto a = (1, 0).",
                "type": "Vector Projection/Shadow",
                "function": "vector_shadow"
            },
            {
                "question": "Solve the system: 2x + 3y = 8, 4x - y = 2",
                "type": "Linear System",
                "function": "solve_gauss"
            },
            {
                "question": "Are the vectors (1, 2, 3), (2, 4, 6), and (0, 1, -1) linearly independent?",
                "type": "Linear Independence",
                "function": "check_coplanar"
            },
            {
                "question": "Calculate the product of matrices A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]].",
                "type": "Matrix Multiplication",
                "function": "matrix_product"
            },
            {
                "question": "Find the distance from point (3, 4, 5) to the plane 2x + 3y - z = 6.",
                "type": "Point-Plane Distance",
                "function": "point_plane_distance"
            },
            {
                "question": "Find the distance from point (1, 2, 3) to the line passing through points (0, 0, 0) and (1, 1, 1).",
                "type": "Point-Line Distance",
                "function": "point_line_distance"
            }
        ]
        
        # Shuffle the questions
        random.shuffle(quiz_questions)
        
        # Take the first 5 questions
        quiz_questions = quiz_questions[:5]
        
        score = 0
        for i, question in enumerate(quiz_questions):
            self.console.print(f"\n[bold cyan]Question {i+1}:[/bold cyan] {question['question']}")
            
            # Create a list of all function names for choices
            all_functions = [q["function"] for q in quiz_questions]
            # Make sure the correct answer is in the choices
            if question["function"] not in all_functions:
                all_functions.append(question["function"])
            # Remove duplicates and limit to 4 choices
            all_functions = list(set(all_functions))
            if len(all_functions) > 4:
                wrong_choices = [f for f in all_functions if f != question["function"]]
                random.shuffle(wrong_choices)
                choices = [question["function"]] + wrong_choices[:3]
                random.shuffle(choices)
            else:
                choices = all_functions
                random.shuffle(choices)
            
            # Convert function names to more readable format
            readable_choices = [f.replace('_', ' ').title() for f in choices]
            
            # Remember the mapping from readable to actual function names
            choice_map = dict(zip(readable_choices, choices))
            
            # Ask the question
            answer = questionary.select(
                "Which function would you use to solve this problem?",
                choices=readable_choices,
                style=custom_style
            ).ask()
            
            # Check if answer is correct
            if choice_map[answer] == question["function"]:
                self.console.print("[green]Correct![/green]")
                score += 1
            else:
                self.console.print(f"[red]Incorrect.[/red] The correct answer is [green]{question['function'].replace('_', ' ').title()}[/green]")
                
                # Show explanation
                help_topic = self.operation_to_help.get(question["function"], None)
                if help_topic:
                    show_explanation = questionary.confirm("Would you like to see an explanation?", style=custom_style).ask()
                    if show_explanation:
                        display_topic_help(self.console, help_topic)
        
        # Show final score
        self.console.print(f"\n[bold]Quiz complete! Your score: {score}/{len(quiz_questions)}[/bold]")
        
        # Offer to take another quiz
        if score < len(quiz_questions):
            retake = questionary.confirm("Would you like to take another quiz?", style=custom_style).ask()
            if retake:
                self.problem_recognition_quiz()
    
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
        
        # Offer to show explanation/derivation
        explain = questionary.confirm("Would you like to see an explanation of how polar coordinates work?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("polar_to_cartesian")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of vector normalization?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("normalize_vector")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of vector basics?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("vector_direction_length")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of vector projections?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("vector_shadow")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of vector orthogonality?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("check_orthogonal")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of calculating angles between vectors?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("vector_angle")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of cross products?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("cross_product")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of calculating triangle areas?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("triangle_area")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of point-line distance calculation?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("point_line_distance")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of vector collinearity?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("check_collinear")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of Gaussian elimination?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("solve_gauss")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of vector coplanarity?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("check_coplanar")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of solution checking?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("check_solution")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of vector equations?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("vector_equation")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of matrix elements?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("matrix_element")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of plane intersections?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("intersection_planes")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of homogeneous intersections?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("homogeneous_intersection")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of pivot and free variables?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("find_pivot_free_vars")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of matrix operations?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("matrix_operations")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of matrix multiplication?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("matrix_product")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of series summation?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("sum_series")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of particular solutions?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("check_particular_solution")
        
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
        
        # Offer to show explanation
        explain = questionary.confirm("Would you like to see an explanation of point-plane distance calculation?", style=custom_style).ask()
        if explain:
            self.show_help_for_operation("point_plane_distance")
        
        self.wait_for_user()
    
    def run(self):
        parser = argparse.ArgumentParser(description='Enhanced Linear Algebra Exercise Framework')
        parser.add_argument('--version', action='version', version='Linear Algebra Framework 1.0')
        parser.add_argument('--no-color', action='store_true', help='Disable colored output')
        
        args = parser.parse_args()
        
        if args.no_color:
            self.console = Console(color_system=None)
        
        self.run_menu()


def main():
    ui = EnhancedLinearAlgebraUI()
    ui.run()


if __name__ == "__main__":
    main()