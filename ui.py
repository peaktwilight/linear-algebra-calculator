#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive UI for Linear Algebra Framework
"""

import sys
import os
import numpy as np
from linalg_cli import LinearAlgebraExerciseFramework

class LinearAlgebraUI:
    def __init__(self):
        self.framework = LinearAlgebraExerciseFramework()
        self.categories = {
            "Vector Operations": [
                ("1", "Convert polar to Cartesian coordinates", self.ui_polar_to_cartesian),
                ("2", "Normalize a vector", self.ui_normalize_vector),
                ("3", "Create vector with direction and length", self.ui_vector_direction_length),
                ("4", "Calculate vector shadow (projection)", self.ui_vector_shadow),
                ("5", "Check if vectors are orthogonal", self.ui_check_orthogonal),
                ("6", "Calculate angle between vectors", self.ui_vector_angle),
                ("7", "Calculate cross product", self.ui_cross_product),
                ("8", "Calculate triangle area", self.ui_triangle_area),
                ("9", "Calculate point-line distance", self.ui_point_line_distance),
                ("10", "Check if vectors are collinear", self.ui_check_collinear),
            ],
            "Matrix Operations": [
                ("11", "Solve system with Gaussian elimination", self.ui_solve_gauss),
                ("12", "Check if vectors are coplanar", self.ui_check_coplanar),
                ("13", "Check if vector is solution to system", self.ui_check_solution),
                ("14", "Solve vector equation", self.ui_vector_equation),
                ("15", "Extract matrix elements", self.ui_matrix_element),
                ("16", "Find intersections of planes", self.ui_intersection_planes),
                ("17", "Find homogeneous intersections", self.ui_homogeneous_intersection),
                ("18", "Identify pivot and free variables", self.ui_find_pivot_free_vars),
                ("19", "Perform basic matrix operations", self.ui_matrix_operations),
                ("20", "Calculate matrix product", self.ui_matrix_product),
            ],
            "Other Operations": [
                ("21", "Calculate sum of series", self.ui_sum_series),
                ("22", "Check particular solutions", self.ui_check_particular_solution),
                ("23", "Calculate point-plane distance", self.ui_point_plane_distance),
            ]
        }
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title="Linear Algebra Exercise Framework"):
        self.clear_screen()
        print("=" * 50)
        print(f"{title:^50}")
        print("=" * 50)
        print()

    def parse_vector_input(self, prompt="Enter vector (comma or space separated)"):
        while True:
            try:
                vector_str = input(prompt + ": ")
                return self.framework.parse_vector(vector_str)
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
    
    def parse_matrix_input(self, prompt="Enter matrix (rows separated by ; or new lines)"):
        print(prompt + ":")
        print("Example: '1, 2, 3; 4, 5, 6' or enter multiple lines and end with an empty line")
        
        lines = []
        while True:
            line = input()
            if not line and lines:  # Empty line to finish input if multiline
                break
            elif ";" in line:  # Single line with semicolons
                matrix_str = line
                break
            else:
                lines.append(line)
        
        try:
            if lines:
                matrix_str = "\n".join(lines)
            return self.framework.parse_matrix(matrix_str)
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
            return self.parse_matrix_input(prompt)

    def wait_for_user(self):
        input("\nPress Enter to continue...")
    
    def display_main_menu(self):
        self.print_header()
        print("Main Menu")
        print("-" * 50)
        
        # Print categories and options
        option_counter = 1
        for category, options in self.categories.items():
            print(f"\n{category}:")
            for option_id, option_name, _ in options:
                print(f"  {option_id}. {option_name}")
        
        print("\n0. Exit")
        print("-" * 50)
        
        choice = input("Enter your choice: ")
        
        if choice == "0":
            sys.exit(0)
        
        # Find the chosen function
        for category, options in self.categories.items():
            for option_id, _, function in options:
                if choice == option_id:
                    function()
                    return
        
        print("Invalid choice. Please try again.")
        self.wait_for_user()
    
    # UI Functions for each operation
    def ui_polar_to_cartesian(self):
        self.print_header("Convert Polar to Cartesian Coordinates")
        
        r = float(input("Enter radius (r): "))
        phi = float(input("Enter angle (phi): "))
        is_degrees = input("Is angle in degrees? (y/n): ").lower() == 'y'
        
        class Args:
            pass
        
        args = Args()
        args.r = r
        args.phi = phi
        args.degrees = is_degrees
        
        self.framework.polar_to_cartesian(args)
        self.wait_for_user()
    
    def ui_normalize_vector(self):
        self.print_header("Normalize a Vector")
        
        vector = self.parse_vector_input()
        
        class Args:
            pass
        
        args = Args()
        args.vector = " ".join(map(str, vector))
        
        self.framework.normalize_vector(args)
        self.wait_for_user()
    
    def ui_vector_direction_length(self):
        self.print_header("Create Vector with Direction and Length")
        
        direction = self.parse_vector_input("Enter direction vector")
        length = float(input("Enter desired length: "))
        
        class Args:
            pass
        
        args = Args()
        args.direction = " ".join(map(str, direction))
        args.length = length
        
        self.framework.vector_direction_length(args)
        self.wait_for_user()
    
    def ui_vector_shadow(self):
        self.print_header("Calculate Vector Shadow (Projection)")
        
        vector_a = self.parse_vector_input("Enter vector to project onto (a)")
        vector_b = self.parse_vector_input("Enter vector to be projected (b)")
        
        class Args:
            pass
        
        args = Args()
        args.vector_a = " ".join(map(str, vector_a))
        args.vector_b = " ".join(map(str, vector_b))
        
        self.framework.vector_shadow(args)
        self.wait_for_user()
    
    def ui_check_orthogonal(self):
        self.print_header("Check if Vectors are Orthogonal")
        
        vector = self.parse_vector_input("Enter reference vector")
        
        print("Enter vectors to check (empty line to finish):")
        check_vectors = []
        while True:
            try:
                vec_str = input("Enter vector (or empty to finish): ")
                if not vec_str:
                    break
                check_vectors.append(" ".join(map(str, self.framework.parse_vector(vec_str))))
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
        
        if not check_vectors:
            print("No vectors to check.")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.vector = " ".join(map(str, vector))
        args.check_vectors = check_vectors
        
        self.framework.check_orthogonal(args)
        self.wait_for_user()
    
    def ui_vector_angle(self):
        self.print_header("Calculate Angle Between Vectors")
        
        vector_a = self.parse_vector_input("Enter first vector (a)")
        vector_b = self.parse_vector_input("Enter second vector (b)")
        
        class Args:
            pass
        
        args = Args()
        args.vector_a = " ".join(map(str, vector_a))
        args.vector_b = " ".join(map(str, vector_b))
        
        self.framework.vector_angle(args)
        self.wait_for_user()
    
    def ui_cross_product(self):
        self.print_header("Calculate Cross Product")
        
        vector_a = self.parse_vector_input("Enter first vector (a)")
        vector_b = self.parse_vector_input("Enter second vector (b)")
        
        class Args:
            pass
        
        args = Args()
        args.vector_a = " ".join(map(str, vector_a))
        args.vector_b = " ".join(map(str, vector_b))
        
        self.framework.cross_product(args)
        self.wait_for_user()
    
    def ui_triangle_area(self):
        self.print_header("Calculate Triangle Area")
        
        point_a = self.parse_vector_input("Enter first point (A)")
        point_b = self.parse_vector_input("Enter second point (B)")
        point_c = self.parse_vector_input("Enter third point (C)")
        
        class Args:
            pass
        
        args = Args()
        args.point_a = " ".join(map(str, point_a))
        args.point_b = " ".join(map(str, point_b))
        args.point_c = " ".join(map(str, point_c))
        
        self.framework.triangle_area(args)
        self.wait_for_user()
    
    def ui_point_line_distance(self):
        self.print_header("Calculate Point-Line Distance")
        
        point_a = self.parse_vector_input("Enter point on line (A)")
        direction = self.parse_vector_input("Enter line direction vector")
        point_b = self.parse_vector_input("Enter point to find distance from (B)")
        
        class Args:
            pass
        
        args = Args()
        args.point_a = " ".join(map(str, point_a))
        args.direction = " ".join(map(str, direction))
        args.point_b = " ".join(map(str, point_b))
        
        self.framework.point_line_distance(args)
        self.wait_for_user()
    
    def ui_check_collinear(self):
        self.print_header("Check if Vectors are Collinear")
        
        print("Enter vectors (empty line to finish):")
        vectors = []
        while True:
            try:
                vec_str = input("Enter vector (or empty to finish): ")
                if not vec_str:
                    break
                vectors.append(" ".join(map(str, self.framework.parse_vector(vec_str))))
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
        
        if len(vectors) < 2:
            print("Need at least 2 vectors to check collinearity.")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.vectors = vectors
        
        self.framework.check_collinear(args)
        self.wait_for_user()
    
    def ui_solve_gauss(self):
        self.print_header("Solve System with Gaussian Elimination")
        
        matrix = self.parse_matrix_input("Enter augmented matrix [A|b]")
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        self.framework.solve_gauss(args)
        self.wait_for_user()
    
    def ui_check_coplanar(self):
        self.print_header("Check if Vectors are Coplanar")
        
        print("Enter vectors (empty line to finish):")
        vectors = []
        while True:
            try:
                vec_str = input("Enter vector (or empty to finish): ")
                if not vec_str:
                    break
                vectors.append(" ".join(map(str, self.framework.parse_vector(vec_str))))
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
        
        if len(vectors) < 3:
            print("Need at least 3 vectors to check coplanarity.")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.vectors = vectors
        
        self.framework.check_coplanar(args)
        self.wait_for_user()
    
    def ui_check_solution(self):
        self.print_header("Check if Vector is Solution to System")
        
        matrix = self.parse_matrix_input("Enter coefficient matrix A")
        
        has_rhs = input("Does the system have a right-hand side b? (y/n): ").lower() == 'y'
        rhs = None
        if has_rhs:
            rhs = self.parse_vector_input("Enter right-hand side vector b")
        
        print("Enter vectors to check (empty line to finish):")
        vectors = []
        while True:
            try:
                vec_str = input("Enter vector (or empty to finish): ")
                if not vec_str:
                    break
                vectors.append(" ".join(map(str, self.framework.parse_vector(vec_str))))
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
        
        if not vectors:
            print("No vectors to check.")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.rhs = " ".join(map(str, rhs)) if rhs is not None else None
        args.vectors = vectors
        
        self.framework.check_solution(args)
        self.wait_for_user()
    
    def ui_vector_equation(self):
        self.print_header("Solve Vector Equation")
        
        print("Enter vector equation using symbolic notation.")
        print("Use u, v, x, y, z as symbols.")
        print("Examples: 'u-v-x-(x+v+x)' or 'u-v-x-(4*x-v+x)'")
        
        equation = input("Enter equation: ")
        
        class Args:
            pass
        
        args = Args()
        args.equation = equation
        
        self.framework.vector_equation(args)
        self.wait_for_user()
    
    def ui_matrix_element(self):
        self.print_header("Extract Matrix Elements")
        
        matrix = self.parse_matrix_input("Enter matrix")
        
        show_element = input("Extract specific element? (y/n): ").lower() == 'y'
        element = None
        if show_element:
            while True:
                try:
                    i = int(input("Enter row index: "))
                    j = int(input("Enter column index: "))
                    element = f"{i},{j}"
                    break
                except ValueError:
                    print("Invalid indices. Please enter integers.")
        
        show_row = input("Extract specific row? (y/n): ").lower() == 'y'
        row = None
        if show_row:
            while True:
                try:
                    row = int(input("Enter row index: "))
                    break
                except ValueError:
                    print("Invalid index. Please enter an integer.")
        
        show_column = input("Extract specific column? (y/n): ").lower() == 'y'
        column = None
        if show_column:
            while True:
                try:
                    column = int(input("Enter column index: "))
                    break
                except ValueError:
                    print("Invalid index. Please enter an integer.")
        
        show_transpose = input("Show matrix transpose? (y/n): ").lower() == 'y'
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.element = element
        args.row = row
        args.column = column
        args.transpose = show_transpose
        
        self.framework.matrix_element(args)
        self.wait_for_user()
    
    def ui_intersection_planes(self):
        self.print_header("Find Intersections of Planes")
        
        matrix = self.parse_matrix_input("Enter augmented matrix representing plane equations")
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        self.framework.intersection_planes(args)
        self.wait_for_user()
    
    def ui_homogeneous_intersection(self):
        self.print_header("Find Homogeneous Intersections")
        
        matrix = self.parse_matrix_input("Enter coefficient matrix representing homogeneous plane equations")
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        self.framework.homogeneous_intersection(args)
        self.wait_for_user()
    
    def ui_find_pivot_free_vars(self):
        self.print_header("Identify Pivot and Free Variables")
        
        matrix = self.parse_matrix_input("Enter augmented matrix representing the system")
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        self.framework.find_pivot_free_vars(args)
        self.wait_for_user()
    
    def ui_matrix_operations(self):
        self.print_header("Perform Basic Matrix Operations")
        
        print("Available operations:")
        print("1. Add matrices (A + B)")
        print("2. Subtract matrices (A - B)")
        print("3. Multiply by scalar (c * A)")
        print("4. Transpose matrix (A^T)")
        
        op_choice = input("Enter operation number: ")
        op_map = {"1": "add", "2": "subtract", "3": "multiply_scalar", "4": "transpose"}
        
        if op_choice not in op_map:
            print("Invalid operation.")
            self.wait_for_user()
            return
        
        operation = op_map[op_choice]
        
        matrix_a = self.parse_matrix_input("Enter matrix A")
        
        matrix_b = None
        if operation in ["add", "subtract"]:
            matrix_b = self.parse_matrix_input("Enter matrix B")
        
        scalar = None
        if operation == "multiply_scalar":
            scalar = float(input("Enter scalar value: "))
        
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
        
        self.framework.matrix_operations(args)
        self.wait_for_user()
    
    def ui_matrix_product(self):
        self.print_header("Calculate Matrix Product")
        
        matrix_a = self.parse_matrix_input("Enter first matrix (A)")
        matrix_b = self.parse_matrix_input("Enter second matrix (B)")
        
        class Args:
            pass
        
        args = Args()
        args.matrix_a = str(matrix_a.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.matrix_b = str(matrix_b.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        
        self.framework.matrix_product(args)
        self.wait_for_user()
    
    def ui_sum_series(self):
        self.print_header("Calculate Sum of Series")
        
        print("Calculate sum of a mathematical series.")
        
        start = int(input("Enter start index: "))
        end = int(input("Enter end index: "))
        
        print("Enter formula using 'i' as the variable.")
        print("Examples: '5+3*i', '50-5*i', 'i*(i+1)/2'")
        formula = input("Enter formula: ")
        
        class Args:
            pass
        
        args = Args()
        args.start = start
        args.end = end
        args.formula = formula
        
        self.framework.sum_series(args)
        self.wait_for_user()
    
    def ui_check_particular_solution(self):
        self.print_header("Check Particular Solutions")
        
        matrix = self.parse_matrix_input("Enter augmented matrix [A|b]")
        
        print("Enter vectors to check (empty line to finish):")
        vectors = []
        while True:
            try:
                vec_str = input("Enter vector (or empty to finish): ")
                if not vec_str:
                    break
                vectors.append(" ".join(map(str, self.framework.parse_vector(vec_str))))
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
        
        if not vectors:
            print("No vectors to check.")
            self.wait_for_user()
            return
        
        class Args:
            pass
        
        args = Args()
        args.matrix = str(matrix.tolist()).replace('], [', '; ').replace('[[', '').replace(']]', '')
        args.vectors = vectors
        
        self.framework.check_particular_solution(args)
        self.wait_for_user()
    
    def ui_point_plane_distance(self):
        self.print_header("Calculate Point-Plane Distance")
        
        normal = self.parse_vector_input("Enter plane normal vector")
        plane_point = self.parse_vector_input("Enter point on plane")
        point = self.parse_vector_input("Enter point to find distance from")
        
        class Args:
            pass
        
        args = Args()
        args.normal = " ".join(map(str, normal))
        args.plane_point = " ".join(map(str, plane_point))
        args.point = " ".join(map(str, point))
        
        self.framework.point_plane_distance(args)
        self.wait_for_user()
    
    def run(self):
        while True:
            self.display_main_menu()


def main():
    ui = LinearAlgebraUI()
    ui.run()


if __name__ == "__main__":
    main()