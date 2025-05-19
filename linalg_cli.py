#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Algebra Exercise Framework
"""

import numpy as np
import sympy as sym
import scipy.linalg as sp
import argparse
import textwrap
from given_reference.core import eliminate, mrref, mnull, FirstNonZero, SortRows

class LinearAlgebraExerciseFramework:
    def __init__(self):
        self.exercise_map = {
            # Vector operations
            "polar_to_cartesian": self.polar_to_cartesian,
            "normalize_vector": self.normalize_vector,
            "vector_direction_length": self.vector_direction_length,
            "vector_shadow": self.vector_shadow,
            "check_orthogonal": self.check_orthogonal,
            "vector_angle": self.vector_angle,
            "cross_product": self.cross_product,
            "triangle_area": self.triangle_area,
            "point_line_distance": self.point_line_distance,
            "check_collinear": self.check_collinear,
            
            # Matrix operations
            "solve_gauss": self.solve_gauss,
            "check_coplanar": self.check_coplanar,
            "check_solution": self.check_solution,
            "vector_equation": self.vector_equation,
            "matrix_element": self.matrix_element,
            "intersection_planes": self.intersection_planes,
            "homogeneous_intersection": self.homogeneous_intersection,
            "find_pivot_free_vars": self.find_pivot_free_vars,
            "matrix_operations": self.matrix_operations,
            "matrix_product": self.matrix_product,
            "sum_series": self.sum_series,
            "check_particular_solution": self.check_particular_solution,
            "point_plane_distance": self.point_plane_distance,
        }
        
    def parse_vector(self, vector_str):
        """Parse a vector from string input"""
        try:
            # Handle different input formats
            vector_str = vector_str.strip()
            if vector_str.startswith('[') and vector_str.endswith(']'):
                vector_str = vector_str[1:-1]
            
            # Split by comma or space
            if ',' in vector_str:
                components = [float(x.strip()) for x in vector_str.split(',')]
            else:
                components = [float(x.strip()) for x in vector_str.split()]
                
            return np.array(components)
        except:
            raise ValueError("Invalid vector format. Use '[x1, x2, ...]' or 'x1 x2 ...'")
    
    def parse_matrix(self, matrix_str):
        """Parse a matrix from string input"""
        try:
            # Remove outer brackets if present
            matrix_str = matrix_str.strip()
            if matrix_str.startswith('[') and matrix_str.endswith(']'):
                matrix_str = matrix_str[1:-1]
            
            # Split rows
            if ';' in matrix_str:
                # Format: "[a,b,c; d,e,f]" or "a,b,c; d,e,f"
                rows = matrix_str.split(';')
                matrix = []
                for row in rows:
                    row = row.strip()
                    if row.startswith('[') and row.endswith(']'):
                        row = row[1:-1]
                    if ',' in row:
                        matrix.append([float(x.strip()) for x in row.split(',')])
                    else:
                        matrix.append([float(x.strip()) for x in row.split()])
            else:
                # Format with newlines or as a flat array to reshape
                lines = matrix_str.strip().split('\n')
                matrix = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('[') and line.endswith(']'):
                        line = line[1:-1]
                    if ',' in line:
                        matrix.append([float(x.strip()) for x in line.split(',')])
                    else:
                        matrix.append([float(x.strip()) for x in line.split()])
            
            return np.array(matrix)
        except:
            raise ValueError("Invalid matrix format. Use '[a,b,c; d,e,f]' or separate rows with newlines")
    
    # Vector operations
    def polar_to_cartesian(self, args):
        """Convert from polar to Cartesian coordinates"""
        if args.degrees:
            phi = float(args.phi) * np.pi / 180
        else:
            phi = float(args.phi)
        
        r = float(args.r)
        v = r * np.array([np.cos(phi), np.sin(phi)])
        
        print(f"Cartesian coordinates: [{v[0]:.4f}, {v[1]:.4f}]")
        return v
    
    def normalize_vector(self, args):
        """Normalize a vector to unit length"""
        vector = self.parse_vector(args.vector)
        norm = np.linalg.norm(vector)
        normalized = vector / norm
        
        print(f"Original vector: {vector}")
        print(f"Length: {norm:.4f}")
        print(f"Normalized vector: {normalized}")
        print(f"Verification - length of normalized vector: {np.linalg.norm(normalized):.4f}")
        return normalized
    
    def vector_direction_length(self, args):
        """Create a vector with specified direction and length"""
        direction = self.parse_vector(args.direction)
        length = float(args.length)
        
        # Normalize the direction
        norm = np.linalg.norm(direction)
        unit_direction = direction / norm
        
        # Create vector with desired length
        result = length * unit_direction
        
        print(f"Unit direction: {unit_direction}")
        print(f"Result vector with length {length}: {result}")
        print(f"Verification - length: {np.linalg.norm(result):.4f}")
        return result
    
    def vector_shadow(self, args):
        """Calculate the shadow (projection) of one vector onto another"""
        a = self.parse_vector(args.vector_a)
        b = self.parse_vector(args.vector_b)
        
        # Calculate the scalar projection
        scalar_proj = np.dot(a, b) / np.linalg.norm(a)
        
        # Calculate the vector projection
        vector_proj = np.dot(a, b) / np.dot(a, a) * a
        
        # Determine if angle is acute or obtuse
        angle_type = "acute (0° < φ < 90°)" if scalar_proj > 0 else "obtuse (90° < φ < 180°)"
        
        print(f"Vector a: {a}")
        print(f"Vector b: {b}")
        print(f"Scalar projection (length of shadow): {scalar_proj:.4f}")
        print(f"Vector projection (shadow vector): {vector_proj}")
        print(f"Angle between vectors is {angle_type}")
        
        return {"scalar_proj": scalar_proj, "vector_proj": vector_proj}
    
    def check_orthogonal(self, args):
        """Check if vectors are orthogonal"""
        v = self.parse_vector(args.vector)
        other_vectors = []
        
        # Parse the list of vectors to check
        for vector_str in args.check_vectors:
            other_vectors.append(self.parse_vector(vector_str))
        
        results = []
        print(f"Checking orthogonality with vector: {v}")
        for i, other in enumerate(other_vectors):
            dot_product = np.dot(v, other)
            is_orthogonal = abs(dot_product) < 1e-10
            results.append(is_orthogonal)
            
            print(f"Vector {i+1}: {other}")
            print(f"  Dot product: {dot_product:.4f}")
            print(f"  Orthogonal: {is_orthogonal}")
            
        return results
    
    def vector_angle(self, args):
        """Calculate the angle between two vectors"""
        a = self.parse_vector(args.vector_a)
        b = self.parse_vector(args.vector_b)
        
        # Calculate cosine of the angle
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Ensure value is in valid range for arccos
        cos_angle = min(1.0, max(-1.0, cos_angle))
        
        # Calculate angle in radians
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees if requested
        angle_deg = angle_rad * 180 / np.pi
        
        print(f"Vector a: {a}")
        print(f"Vector b: {b}")
        print(f"Angle in radians: {angle_rad:.4f}")
        print(f"Angle in degrees: {angle_deg:.4f}")
        
        return {"radians": angle_rad, "degrees": angle_deg}
    
    def cross_product(self, args):
        """Calculate the cross product of two vectors"""
        a = self.parse_vector(args.vector_a)
        b = self.parse_vector(args.vector_b)
        
        # Ensure vectors are 3D (pad with zeros if needed)
        if len(a) < 3:
            a = np.pad(a, (0, 3 - len(a)), 'constant')
        if len(b) < 3:
            b = np.pad(b, (0, 3 - len(b)), 'constant')
        
        # Calculate cross product
        cross = np.cross(a, b)
        
        print(f"Vector a: {a}")
        print(f"Vector b: {b}")
        print(f"Cross product (a × b): {cross}")
        
        # Verify orthogonality
        print(f"Verification - dot product with a: {np.dot(cross, a):.10f}")
        print(f"Verification - dot product with b: {np.dot(cross, b):.10f}")
        
        return cross
    
    def triangle_area(self, args):
        """Calculate the area of a triangle defined by three points"""
        A = self.parse_vector(args.point_a)
        B = self.parse_vector(args.point_b)
        C = self.parse_vector(args.point_c)
        
        # Vectors representing two sides of the triangle
        AB = B - A
        AC = C - A
        
        # Area = 1/2 * |AB × AC|
        cross = np.cross(AB, AC)
        area = np.linalg.norm(cross) / 2
        
        print(f"Point A: {A}")
        print(f"Point B: {B}")
        print(f"Point C: {C}")
        print(f"Vector AB: {AB}")
        print(f"Vector AC: {AC}")
        print(f"Cross product (AB × AC): {cross}")
        print(f"Triangle area: {area:.4f}")
        
        return area
    
    def point_line_distance(self, args):
        """Calculate the distance from a point to a line"""
        A = self.parse_vector(args.point_a)  # Point on line
        v = self.parse_vector(args.direction)  # Direction vector
        B = self.parse_vector(args.point_b)  # Point to find distance from
        
        # Ensure vectors are 3D (pad with zeros if needed)
        if len(A) < 3:
            A = np.pad(A, (0, 3 - len(A)), 'constant')
        if len(v) < 3:
            v = np.pad(v, (0, 3 - len(v)), 'constant')
        if len(B) < 3:
            B = np.pad(B, (0, 3 - len(B)), 'constant')
        
        # Distance = |cross(v, B-A)| / |v|
        distance = np.linalg.norm(np.cross(v, B - A)) / np.linalg.norm(v)
        
        print(f"Point on line (A): {A}")
        print(f"Line direction vector (v): {v}")
        print(f"Point to find distance from (B): {B}")
        print(f"Distance: {distance:.4f}")
        
        return distance
    
    def check_collinear(self, args):
        """Check if vectors are collinear"""
        vectors = []
        for vector_str in args.vectors:
            vectors.append(self.parse_vector(vector_str))
        
        # Create a matrix with the vectors as rows
        matrix = np.array(vectors)
        print(f"Matrix of vectors:\n{matrix}")
        
        # Apply Gaussian elimination
        reduced = mrref(matrix)
        print(f"Row echelon form:\n{reduced}")
        
        # Check for zero rows to determine linear dependence
        rank = np.sum([np.linalg.norm(row) > 1e-10 for row in reduced])
        is_collinear = rank < len(vectors) and rank <= 1
        
        if is_collinear:
            print("The vectors are collinear (linearly dependent)")
        else:
            print("The vectors are not collinear (linearly independent)")
        
        return is_collinear
    
    # Matrix operations
    def solve_gauss(self, args):
        """Solve a system of linear equations using Gaussian elimination"""
        # Parse the augmented matrix [A|b]
        augmented = self.parse_matrix(args.matrix)
        
        # Apply row reduction
        reduced = mrref(augmented)
        
        print("Original augmented matrix [A|b]:")
        print(augmented)
        print("\nRow echelon form:")
        print(reduced)
        
        # Check if the system has a solution
        n_rows, n_cols = reduced.shape
        
        # Check for inconsistency (0 = non-zero)
        inconsistent = False
        for i in range(n_rows):
            if np.all(abs(reduced[i, :-1]) < 1e-10) and abs(reduced[i, -1]) > 1e-10:
                inconsistent = True
                print("\nThe system is inconsistent (no solution)")
                return None
        
        # Check if the system has a unique solution
        if n_cols - 1 == n_rows:
            # Square system with unique solution
            solution = reduced[:, -1]
            print("\nUnique solution:")
            for i, val in enumerate(solution):
                print(f"x{i+1} = {val:.4f}")
            return solution
        else:
            # System with free variables
            print("\nThe system has infinitely many solutions (free variables exist)")
            
            # Identify pivot and free variables
            pivots = []
            for i in range(n_rows):
                non_zero_cols = np.where(abs(reduced[i, :-1]) > 1e-10)[0]
                if len(non_zero_cols) > 0:
                    pivots.append(non_zero_cols[0])
                    
            free_vars = [j for j in range(n_cols-1) if j not in pivots]
            
            print(f"Pivot variables: x{[p+1 for p in pivots]}")
            print(f"Free variables: x{[f+1 for f in free_vars]}")
            
            # Create a null space basis for homogeneous part
            if free_vars:
                null_space = mnull(augmented[:, :-1])
                print("\nParametric form of the solution:")
                
                # Create a particular solution if the system is non-homogeneous
                particular = np.zeros(n_cols - 1)
                for i, p in enumerate(pivots):
                    particular[p] = reduced[i, -1]
                
                print(f"x = {particular} + t * {null_space}")
                
                return {"particular": particular, "null_space": null_space}
            
            return reduced
    
    def check_coplanar(self, args):
        """Check if vectors are coplanar (linearly dependent in 3D)"""
        vectors = []
        for vector_str in args.vectors:
            vectors.append(self.parse_vector(vector_str))
        
        # Create a matrix with the vectors as rows
        matrix = np.array(vectors)
        print(f"Matrix of vectors:\n{matrix}")
        
        # Apply Gaussian elimination
        reduced = mrref(matrix)
        print(f"Row echelon form:\n{reduced}")
        
        # Check for zero rows to determine linear dependence
        # In 3D, vectors are coplanar if rank <= 2
        rank = np.sum([np.linalg.norm(row) > 1e-10 for row in reduced])
        is_coplanar = rank < len(vectors) and rank <= 2
        
        if is_coplanar:
            print("The vectors are coplanar (linearly dependent)")
            
            # Try to find a linear combination that gives the zero vector
            augmented = np.zeros((len(vectors), len(vectors) + 1))
            augmented[:, :-1] = matrix
            null_space = mnull(augmented[:, :-1])
            
            if null_space.size > 0:
                print("\nLinear combination coefficients that yield the zero vector:")
                print(null_space)
        else:
            print("The vectors are not coplanar (linearly independent)")
        
        return is_coplanar
    
    def check_solution(self, args):
        """Check if a vector is a solution to a system of linear equations"""
        # Parse the coefficient matrix A
        A = self.parse_matrix(args.matrix)
        
        # Parse the right-hand side vector b
        b = self.parse_vector(args.rhs) if args.rhs else None
        
        # Parse the vectors to check
        vectors = []
        for vector_str in args.vectors:
            vectors.append(self.parse_vector(vector_str))
        
        # Check each vector
        results = []
        for i, v in enumerate(vectors):
            print(f"Checking vector {i+1}: {v}")
            
            # Compute A·v
            Av = A @ v
            
            print(f"A·v = {Av}")
            
            # Check if A·v = b (within tolerance)
            if b is not None:
                diff = Av - b
                is_solution = np.all(np.abs(diff) < 1e-10)
                results.append(is_solution)
                
                print(f"b = {b}")
                print(f"A·v - b = {diff}")
                print(f"Is solution: {is_solution}")
            else:
                # Just show the result of A·v
                print(f"No right-hand side provided, showing A·v only")
                results.append(None)
            
            print("")
        
        return results
    
    def vector_equation(self, args):
        """Solve vector equations using symbolic mathematics"""
        # Define symbols
        u, v, x, y, z = sym.symbols('u v x y z')
        
        # Parse equation terms
        equation = args.equation
        
        # Common substitutions for vector notation
        equation = equation.replace("||", "norm")
        
        # Try to solve
        try:
            # Parse and solve the equation
            solution = sym.solve(equation, x)
            
            print(f"Equation: {equation}")
            print(f"Solution for x: {solution}")
            
            return solution
        except Exception as e:
            print(f"Error solving equation: {e}")
            print("Please ensure the equation is in valid SymPy format")
            return None
    
    def matrix_element(self, args):
        """Extract and display elements from a matrix"""
        matrix = self.parse_matrix(args.matrix)
        
        print("Matrix:")
        print(matrix)
        print(f"Shape: {matrix.shape}")
        
        if args.element:
            # Parse the element index (i,j)
            try:
                i, j = map(int, args.element.split(','))
                value = matrix[i, j]
                print(f"Element ({i}, {j}): {value}")
            except Exception as e:
                print(f"Error accessing element: {e}")
                print("Use format 'i,j' for indices")
        
        if args.row is not None:
            # Get specific row
            try:
                row = int(args.row)
                print(f"Row {row}:")
                print(matrix[row])
            except Exception as e:
                print(f"Error accessing row: {e}")
        
        if args.column is not None:
            # Get specific column
            try:
                col = int(args.column)
                print(f"Column {col}:")
                print(matrix[:, col])
            except Exception as e:
                print(f"Error accessing column: {e}")
                
        # Additional operations if requested
        if args.transpose:
            print("Transpose:")
            print(matrix.T)
            
        return matrix
    
    def intersection_planes(self, args):
        """Determine the intersection of planes (point, line, empty, etc.)"""
        # Parse the augmented matrix representing the plane equations
        augmented = self.parse_matrix(args.matrix)
        
        # Apply row reduction
        reduced = mrref(augmented)
        
        print("Augmented matrix representing plane equations:")
        print(augmented)
        print("\nRow echelon form:")
        print(reduced)
        
        # Analyze the result
        n_rows, n_cols = reduced.shape
        n_vars = n_cols - 1  # Number of variables (x, y, z)
        
        # Check for inconsistency (0 = non-zero)
        for i in range(n_rows):
            if np.all(abs(reduced[i, :-1]) < 1e-10) and abs(reduced[i, -1]) > 1e-10:
                print("\nThe planes have no intersection (inconsistent system)")
                return "no intersection"
        
        # Count the number of independent equations
        rank = np.sum([np.linalg.norm(row[:-1]) > 1e-10 for row in reduced])
        
        if rank == n_vars:
            # Unique intersection point
            solution = np.zeros(n_vars)
            for i in range(rank):
                # Find the first non-zero coefficient
                j = np.where(abs(reduced[i, :-1]) > 1e-10)[0][0]
                solution[j] = reduced[i, -1]
            
            print("\nThe planes intersect at a single point:")
            print(solution)
            return {"type": "point", "point": solution}
            
        elif rank == n_vars - 1:
            # Line intersection
            # Find the null space of the coefficient matrix
            null_space = mnull(augmented[:, :-1])
            
            # Create a particular solution
            particular = np.zeros(n_vars)
            pivots = []
            for i in range(rank):
                non_zero_cols = np.where(abs(reduced[i, :-1]) > 1e-10)[0]
                pivots.append(non_zero_cols[0])
                particular[pivots[-1]] = reduced[i, -1]
            
            print("\nThe planes intersect along a line:")
            print(f"Point on line: {particular}")
            print(f"Direction vector: {null_space.flatten()}")
            
            return {"type": "line", "point": particular, "direction": null_space.flatten()}
            
        elif rank == n_vars - 2 and n_vars == 3:
            # Coincident planes (plane intersection)
            print("\nTwo of the planes are identical, resulting in a plane intersection")
            return {"type": "plane"}
            
        else:
            print("\nThe planes have a more complex intersection structure")
            return {"type": "complex", "rank": rank}
    
    def homogeneous_intersection(self, args):
        """Determine the intersection of planes through the origin"""
        # Parse the coefficient matrix (no right-hand side)
        coefficients = self.parse_matrix(args.matrix)
        
        # Apply row reduction
        reduced = mrref(coefficients)
        
        print("Coefficient matrix representing homogeneous plane equations:")
        print(coefficients)
        print("\nRow echelon form:")
        print(reduced)
        
        # Analyze the result
        n_rows, n_cols = reduced.shape
        
        # Count the number of independent equations
        rank = np.sum([np.linalg.norm(row) > 1e-10 for row in reduced])
        
        if rank == n_cols:
            # Only the trivial solution exists
            print("\nThe planes only intersect at the origin (trivial solution)")
            return {"type": "point", "point": np.zeros(n_cols)}
            
        elif rank == n_cols - 1:
            # Line through the origin
            null_space = mnull(coefficients)
            direction = null_space.flatten()
            
            print("\nThe planes intersect along a line through the origin:")
            print(f"Direction vector: {direction}")
            
            return {"type": "line", "direction": direction}
            
        elif rank == n_cols - 2 and n_cols == 3:
            # Plane through the origin
            print("\nThe planes intersect in a plane through the origin")
            return {"type": "plane"}
            
        else:
            print("\nThe planes have a more complex intersection structure")
            return {"type": "complex", "rank": rank}
    
    def find_pivot_free_vars(self, args):
        """Identify pivot and free variables in a linear system"""
        # Parse the augmented matrix
        augmented = self.parse_matrix(args.matrix)
        
        # Apply row reduction
        reduced = mrref(augmented)
        
        print("Augmented matrix:")
        print(augmented)
        print("\nRow echelon form:")
        print(reduced)
        
        # Analyze the result
        n_rows, n_cols = reduced.shape
        n_vars = n_cols - 1  # Number of variables
        
        # Check for inconsistency
        for i in range(n_rows):
            if np.all(abs(reduced[i, :-1]) < 1e-10) and abs(reduced[i, -1]) > 1e-10:
                print("\nThe system is inconsistent (no solution)")
                return None
        
        # Identify pivot variables
        pivots = []
        for i in range(min(n_rows, n_vars)):
            non_zero_cols = np.where(abs(reduced[i, :-1]) > 1e-10)[0]
            if len(non_zero_cols) > 0:
                pivots.append(non_zero_cols[0])
        
        # Identify free variables
        free_vars = [j for j in range(n_vars) if j not in pivots]
        
        print("\nPivot variables:", [f"x{p+1}" for p in pivots])
        print("Free variables:", [f"x{f+1}" for f in free_vars])
        
        if free_vars:
            # Find the null space basis
            null_space = mnull(augmented[:, :-1])
            
            # Find a particular solution
            particular = np.zeros(n_vars)
            for i, p in enumerate(pivots):
                if i < len(reduced) and np.linalg.norm(reduced[i, :-1]) > 1e-10:
                    particular[p] = reduced[i, -1]
            
            print("\nSolution form:")
            print(f"x = {particular} + t * {null_space}")
            
            return {
                "pivot_vars": pivots,
                "free_vars": free_vars,
                "particular": particular,
                "null_space": null_space
            }
        else:
            # Unique solution
            solution = np.zeros(n_vars)
            for i, p in enumerate(pivots):
                solution[p] = reduced[i, -1]
            
            print("\nUnique solution:")
            print(solution)
            
            return {
                "pivot_vars": pivots,
                "free_vars": [],
                "solution": solution
            }
    
    def matrix_operations(self, args):
        """Perform basic matrix operations"""
        # Parse matrices
        if args.matrix_a:
            A = self.parse_matrix(args.matrix_a)
            print("Matrix A:")
            print(A)
        
        if args.matrix_b:
            B = self.parse_matrix(args.matrix_b)
            print("Matrix B:")
            print(B)
        
        # Perform the requested operation
        if args.operation == "add":
            if A.shape != B.shape:
                print("Error: Matrices must have the same dimensions for addition")
                return None
            
            result = A + B
            print("\nA + B:")
            print(result)
            return result
            
        elif args.operation == "subtract":
            if A.shape != B.shape:
                print("Error: Matrices must have the same dimensions for subtraction")
                return None
            
            result = A - B
            print("\nA - B:")
            print(result)
            return result
            
        elif args.operation == "multiply_scalar":
            scalar = float(args.scalar)
            result = scalar * A
            print(f"\n{scalar} * A:")
            print(result)
            return result
            
        elif args.operation == "transpose":
            result = A.T
            print("\nA^T (transpose):")
            print(result)
            return result
            
        else:
            print("Error: Unknown operation")
            return None
    
    def matrix_product(self, args):
        """Calculate the product of two matrices"""
        # Parse matrices
        A = self.parse_matrix(args.matrix_a)
        B = self.parse_matrix(args.matrix_b)
        
        print("Matrix A:")
        print(A)
        print(f"Shape: {A.shape}")
        
        print("\nMatrix B:")
        print(B)
        print(f"Shape: {B.shape}")
        
        # Check if multiplication is possible
        if A.shape[1] != B.shape[0]:
            print(f"\nError: Cannot multiply matrices of shapes {A.shape} and {B.shape}")
            print("Number of columns in A must equal number of rows in B")
            return None
        
        # Calculate the product
        product = A @ B
        
        print("\nMatrix product A × B:")
        print(product)
        print(f"Shape: {product.shape}")
        
        return product
    
    def sum_series(self, args):
        """Calculate mathematical series"""
        start = int(args.start)
        end = int(args.end)
        formula = args.formula
        
        # Replace i with a SymPy symbol for evaluation
        i = sym.Symbol('i')
        
        try:
            # Parse the formula
            expr = sym.sympify(formula)
            
            # Calculate the sum
            total = 0
            terms = []
            
            for idx in range(start, end + 1):
                term_value = float(expr.subs(i, idx))
                total += term_value
                terms.append(term_value)
            
            print(f"Sum from i={start} to i={end} of {formula}")
            print(f"Terms: {terms}")
            print(f"Sum: {total}")
            
            # Try to compute the sum symbolically if possible
            symbolic_sum = sym.Sum(expr, (i, start, end))
            try:
                symbolic_result = symbolic_sum.doit()
                print(f"Symbolic result: {symbolic_result}")
            except:
                pass
            
            return total
            
        except Exception as e:
            print(f"Error calculating sum: {e}")
            print("Please ensure the formula is in valid SymPy format")
            return None
    
    def check_particular_solution(self, args):
        """Check if vectors are particular solutions to a linear system"""
        # Parse the augmented matrix [A|b]
        augmented = self.parse_matrix(args.matrix)
        
        # Split into coefficient matrix and right-hand side
        A = augmented[:, :-1]
        b = augmented[:, -1]
        
        print("Coefficient matrix A:")
        print(A)
        print("\nRight-hand side b:")
        print(b)
        
        # Parse the vectors to check
        vectors = []
        for vector_str in args.vectors:
            vectors.append(self.parse_vector(vector_str))
        
        # Check each vector
        results = []
        for i, v in enumerate(vectors):
            print(f"\nChecking vector {i+1}: {v}")
            
            # Compute A·v
            Av = A @ v
            
            print(f"A·v = {Av}")
            
            # Check if A·v = b (within tolerance)
            diff = Av - b
            residual_norm = np.linalg.norm(diff)
            is_solution = residual_norm < 1e-10
            
            print(f"A·v - b = {diff}")
            print(f"Residual norm: {residual_norm:.10f}")
            
            if is_solution:
                print("This vector is a particular solution")
                
                # Check if it's also a homogeneous solution
                is_homogeneous = np.linalg.norm(v) < 1e-10
                
                if is_homogeneous:
                    print("This is the trivial solution (zero vector)")
                else:
                    print("This is a non-trivial particular solution")
            else:
                # Check if it's a homogeneous solution
                Av_zero = A @ v
                is_homogeneous = np.linalg.norm(Av_zero) < 1e-10
                
                if is_homogeneous:
                    print("This vector is a homogeneous solution (A·v = 0)")
                else:
                    print("This vector is not a solution")
            
            results.append({"is_particular": is_solution, "is_homogeneous": is_homogeneous if 'is_homogeneous' in locals() else False})
        
        return results
    
    def point_plane_distance(self, args):
        """Calculate the distance from a point to a plane"""
        # Parse the plane normal vector
        normal = self.parse_vector(args.normal)
        
        # Parse the point on the plane
        plane_point = self.parse_vector(args.plane_point)
        
        # Parse the point to find distance from
        point = self.parse_vector(args.point)
        
        # Calculate the distance
        normal_unit = normal / np.linalg.norm(normal)
        distance = abs(np.dot(normal_unit, point - plane_point))
        
        print(f"Plane normal vector: {normal}")
        print(f"Unit normal vector: {normal_unit}")
        print(f"Point on plane: {plane_point}")
        print(f"Point to find distance from: {point}")
        print(f"Distance: {distance:.4f}")
        
        return distance
    
    def run(self):
        parser = argparse.ArgumentParser(
            description="Linear Algebra Exercise Framework",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent('''
            Examples:
              # Convert polar coordinates to Cartesian
              python linalg_cli.py polar_to_cartesian --r 5 --phi 45 --degrees
              
              # Normalize a vector
              python linalg_cli.py normalize_vector --vector "3, 4"
              
              # Create a vector with specific direction and length
              python linalg_cli.py vector_direction_length --direction "3, 4" --length 10
              
              # Calculate the projection of one vector onto another
              python linalg_cli.py vector_shadow --vector_a "3, -4" --vector_b "12, -1"
              
              # Check if vectors are orthogonal
              python linalg_cli.py check_orthogonal --vector "1, 5, 2" --check_vectors "263, -35, -44" "71, 5, -48"
              
              # Calculate the angle between two vectors
              python linalg_cli.py vector_angle --vector_a "1, 1, -1" --vector_b "1, -1, 1"
              
              # Calculate the cross product of two vectors
              python linalg_cli.py cross_product --vector_a "-2, 0, 0" --vector_b "0, 9, 8"
              
              # Calculate the area of a triangle
              python linalg_cli.py triangle_area --point_a "0, 4, 2" --point_b "0, 8, 5" --point_c "0, 8, -1"
              
              # Calculate the distance from a point to a line
              python linalg_cli.py point_line_distance --point_a "3, 0, 0" --direction "2, 0, 0" --point_b "5, 10, 0"
              
              # Check if vectors are collinear
              python linalg_cli.py check_collinear --vectors "-3, 2" "6, -4"
              
              # Solve a system of linear equations
              python linalg_cli.py solve_gauss --matrix "1, -4, -2, -25; 0, -3, 6, -18; 7, -13, -4, -85"
              
              # Check if vectors are coplanar
              python linalg_cli.py check_coplanar --vectors "3, 2, 0" "0, 4, 3" "3, 10, 6"
              
              # Check if a vector is a solution to a system of equations
              python linalg_cli.py check_solution --matrix "5, 2; 3, 2" --rhs "5, 7" --vectors "-1, 5" "5, -1"
              
              # Solve vector equations
              python linalg_cli.py vector_equation --equation "u-v-x-(x+v+x)"
              
              # Extract elements from a matrix
              python linalg_cli.py matrix_element --matrix "4, 4, -16, 4; 1, 0, -4, 2; 5, 2, -20, 8" --element "1,3"
              
              # Determine the intersection of planes
              python linalg_cli.py intersection_planes --matrix "2, 1, -2, -1; 1, 8, -4, 10; 6, -1, 18, 81"
              
              # Determine the intersection of planes through the origin
              python linalg_cli.py homogeneous_intersection --matrix "0, 9, 8; 8, 9, 6; 10, 0, 6"
              
              # Identify pivot and free variables
              python linalg_cli.py find_pivot_free_vars --matrix "3, 3, -8, 6, 0, 14; 1, 1, -4, 2, 1, 3; 5, 5, -20, 10, 3, 13"
              
              # Perform matrix operations
              python linalg_cli.py matrix_operations --operation add --matrix_a "0, 1; 2, 3; 4, 2" --matrix_b "1, 2; 2, 0; 0, -3"
              
              # Calculate matrix product
              python linalg_cli.py matrix_product --matrix_a "0, 4, 2; 2, -1, 5; 6, 0, -3" --matrix_b "4, 1, 9; 5, 4, 2; -3, 5, 2"
              
              # Calculate the sum of a series
              python linalg_cli.py sum_series --start 0 --end 15 --formula "5+3*i"
              
              # Check if vectors are particular solutions
              python linalg_cli.py check_particular_solution --matrix "4, 4, -16, 4; 1, 0, -4, 2; 5, 2, -20, 8" --vectors "2, -1, 0" "0, 2, 1"
              
              # Calculate the distance from a point to a plane
              python linalg_cli.py point_plane_distance --normal "4, 0, -3" --plane_point "5, 1, -2" --point "10, 4, -12"
            '''))
        
        subparsers = parser.add_subparsers(dest='command', help='Exercise type')
        
        # Vector operations
        polar_parser = subparsers.add_parser('polar_to_cartesian', help='Convert from polar to Cartesian coordinates')
        polar_parser.add_argument('--r', required=True, help='Radius/magnitude')
        polar_parser.add_argument('--phi', required=True, help='Angle (phi)')
        polar_parser.add_argument('--degrees', action='store_true', help='Interpret phi as degrees (default is radians)')
        
        normalize_parser = subparsers.add_parser('normalize_vector', help='Normalize a vector to unit length')
        normalize_parser.add_argument('--vector', required=True, help='Vector to normalize')
        
        direction_parser = subparsers.add_parser('vector_direction_length', help='Create a vector with specified direction and length')
        direction_parser.add_argument('--direction', required=True, help='Direction vector')
        direction_parser.add_argument('--length', required=True, help='Desired length')
        
        shadow_parser = subparsers.add_parser('vector_shadow', help='Calculate the shadow (projection) of one vector onto another')
        shadow_parser.add_argument('--vector_a', required=True, help='Vector to project onto (base)')
        shadow_parser.add_argument('--vector_b', required=True, help='Vector to be projected (shadow)')
        
        orthogonal_parser = subparsers.add_parser('check_orthogonal', help='Check if vectors are orthogonal')
        orthogonal_parser.add_argument('--vector', required=True, help='Reference vector')
        orthogonal_parser.add_argument('--check_vectors', required=True, nargs='+', help='Vectors to check for orthogonality')
        
        angle_parser = subparsers.add_parser('vector_angle', help='Calculate the angle between two vectors')
        angle_parser.add_argument('--vector_a', required=True, help='First vector')
        angle_parser.add_argument('--vector_b', required=True, help='Second vector')
        
        cross_parser = subparsers.add_parser('cross_product', help='Calculate the cross product of two vectors')
        cross_parser.add_argument('--vector_a', required=True, help='First vector')
        cross_parser.add_argument('--vector_b', required=True, help='Second vector')
        
        triangle_parser = subparsers.add_parser('triangle_area', help='Calculate the area of a triangle defined by three points')
        triangle_parser.add_argument('--point_a', required=True, help='First point')
        triangle_parser.add_argument('--point_b', required=True, help='Second point')
        triangle_parser.add_argument('--point_c', required=True, help='Third point')
        
        line_dist_parser = subparsers.add_parser('point_line_distance', help='Calculate the distance from a point to a line')
        line_dist_parser.add_argument('--point_a', required=True, help='Point on the line')
        line_dist_parser.add_argument('--direction', required=True, help='Direction vector of the line')
        line_dist_parser.add_argument('--point_b', required=True, help='Point to find distance from')
        
        collinear_parser = subparsers.add_parser('check_collinear', help='Check if vectors are collinear')
        collinear_parser.add_argument('--vectors', required=True, nargs='+', help='Vectors to check')
        
        # Matrix operations
        gauss_parser = subparsers.add_parser('solve_gauss', help='Solve a system of linear equations using Gaussian elimination')
        gauss_parser.add_argument('--matrix', required=True, help='Augmented matrix [A|b]')
        
        coplanar_parser = subparsers.add_parser('check_coplanar', help='Check if vectors are coplanar (linearly dependent in 3D)')
        coplanar_parser.add_argument('--vectors', required=True, nargs='+', help='Vectors to check')
        
        solution_parser = subparsers.add_parser('check_solution', help='Check if a vector is a solution to a system of linear equations')
        solution_parser.add_argument('--matrix', required=True, help='Coefficient matrix A')
        solution_parser.add_argument('--rhs', help='Right-hand side vector b')
        solution_parser.add_argument('--vectors', required=True, nargs='+', help='Vectors to check as solutions')
        
        equation_parser = subparsers.add_parser('vector_equation', help='Solve vector equations using symbolic mathematics')
        equation_parser.add_argument('--equation', required=True, help='Vector equation to solve (using SymPy syntax)')
        
        matrix_element_parser = subparsers.add_parser('matrix_element', help='Extract and display elements from a matrix')
        matrix_element_parser.add_argument('--matrix', required=True, help='Matrix to analyze')
        matrix_element_parser.add_argument('--element', help='Element to extract (i,j)')
        matrix_element_parser.add_argument('--row', type=int, help='Row to extract')
        matrix_element_parser.add_argument('--column', type=int, help='Column to extract')
        matrix_element_parser.add_argument('--transpose', action='store_true', help='Display the transpose of the matrix')
        
        intersection_parser = subparsers.add_parser('intersection_planes', help='Determine the intersection of planes')
        intersection_parser.add_argument('--matrix', required=True, help='Augmented matrix representing plane equations')
        
        homogeneous_parser = subparsers.add_parser('homogeneous_intersection', help='Determine the intersection of planes through the origin')
        homogeneous_parser.add_argument('--matrix', required=True, help='Coefficient matrix representing homogeneous plane equations')
        
        pivot_parser = subparsers.add_parser('find_pivot_free_vars', help='Identify pivot and free variables in a linear system')
        pivot_parser.add_argument('--matrix', required=True, help='Augmented matrix representing the system')
        
        matrix_op_parser = subparsers.add_parser('matrix_operations', help='Perform basic matrix operations')
        matrix_op_parser.add_argument('--operation', required=True, choices=['add', 'subtract', 'multiply_scalar', 'transpose'], help='Operation to perform')
        matrix_op_parser.add_argument('--matrix_a', required=True, help='First matrix')
        matrix_op_parser.add_argument('--matrix_b', help='Second matrix (for binary operations)')
        matrix_op_parser.add_argument('--scalar', help='Scalar value (for scalar multiplication)')
        
        matrix_product_parser = subparsers.add_parser('matrix_product', help='Calculate the product of two matrices')
        matrix_product_parser.add_argument('--matrix_a', required=True, help='First matrix')
        matrix_product_parser.add_argument('--matrix_b', required=True, help='Second matrix')
        
        sum_parser = subparsers.add_parser('sum_series', help='Calculate mathematical series')
        sum_parser.add_argument('--start', required=True, help='Start index')
        sum_parser.add_argument('--end', required=True, help='End index')
        sum_parser.add_argument('--formula', required=True, help='Formula to evaluate (use i as the variable)')
        
        particular_parser = subparsers.add_parser('check_particular_solution', help='Check if vectors are particular solutions to a linear system')
        particular_parser.add_argument('--matrix', required=True, help='Augmented matrix [A|b]')
        particular_parser.add_argument('--vectors', required=True, nargs='+', help='Vectors to check as solutions')
        
        plane_dist_parser = subparsers.add_parser('point_plane_distance', help='Calculate the distance from a point to a plane')
        plane_dist_parser.add_argument('--normal', required=True, help='Normal vector of the plane')
        plane_dist_parser.add_argument('--plane_point', required=True, help='Point on the plane')
        plane_dist_parser.add_argument('--point', required=True, help='Point to find distance from')
        
        # Parse args
        args = parser.parse_args()
        
        if args.command is None:
            parser.print_help()
            return
        
        # Execute the requested exercise
        if args.command in self.exercise_map:
            return self.exercise_map[args.command](args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()


def main():
    framework = LinearAlgebraExerciseFramework()
    framework.run()


if __name__ == "__main__":
    main()