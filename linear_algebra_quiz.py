#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Algebra Quiz Generator

This module provides functions to generate quiz questions for linear algebra concepts.
"""

import numpy as np
import random
import sympy as sym
from given_reference.core import mrref, mnull
import json

class LinearAlgebraQuiz:
    """
    Quiz generator for linear algebra concepts.
    Creates quizzes based on the example problems from the reference materials.
    """
    
    def __init__(self):
        """Initialize the quiz generator with the question types."""
        # Define quiz types with their generation functions
        self.quiz_types = {
            "polar_to_cartesian": self.generate_polar_to_cartesian,
            "vector_normalization": self.generate_vector_normalization,
            "vector_projection": self.generate_vector_projection,
            "orthogonal_vectors": self.generate_orthogonal_vectors,
            "vector_angle": self.generate_vector_angle,
            "cross_product": self.generate_cross_product,
            "triangle_area": self.generate_triangle_area,
            "point_line_distance": self.generate_point_line_distance,
            "vector_collinearity": self.generate_vector_collinearity,
            "gaussian_elimination": self.generate_gaussian_elimination,
            "coplanar_vectors": self.generate_coplanar_vectors,
            "matrix_product": self.generate_matrix_product
        }
        
        # Define difficulty levels
        self.difficulty_levels = ["easy", "medium", "hard"]
    
    def get_quiz_types(self):
        """Return the available quiz types."""
        return list(self.quiz_types.keys())
    
    def generate_quiz(self, quiz_type, difficulty="medium", count=1):
        """
        Generate quiz questions of the specified type.
        
        Args:
            quiz_type (str): Type of quiz to generate
            difficulty (str): Difficulty level ("easy", "medium", "hard")
            count (int): Number of questions to generate
            
        Returns:
            list: List of quiz question dictionaries
        """
        if quiz_type not in self.quiz_types:
            raise ValueError(f"Unknown quiz type: {quiz_type}")
        
        if difficulty not in self.difficulty_levels:
            raise ValueError(f"Unknown difficulty level: {difficulty}")
        
        questions = []
        for _ in range(count):
            question = self.quiz_types[quiz_type](difficulty)
            questions.append(question)
            
        return questions
    
    def generate_random_quiz(self, count=5, include_types=None, difficulty="medium"):
        """
        Generate a quiz with random question types.
        
        Args:
            count (int): Number of questions
            include_types (list): List of question types to include (None for all)
            difficulty (str): Difficulty level
            
        Returns:
            list: List of quiz question dictionaries
        """
        available_types = include_types or list(self.quiz_types.keys())
        questions = []
        
        # Choose random types
        selected_types = random.choices(available_types, k=count)
        
        # Generate questions
        for quiz_type in selected_types:
            question = self.quiz_types[quiz_type](difficulty)
            questions.append(question)
            
        return questions
    
    def generate_polar_to_cartesian(self, difficulty="medium"):
        """Generate a polar to Cartesian conversion question."""
        if difficulty == "easy":
            # Nice angles like 30°, 45°, 60°, 90°
            radius = random.choice([1, 2, 5, 10])
            angle_deg = random.choice([0, 30, 45, 60, 90, 180, 270, 360])
            use_degrees = True
        elif difficulty == "medium":
            # More arbitrary angles but still clean numbers
            radius = random.uniform(1, 10)
            radius = round(radius, 1)
            angle_deg = random.randint(0, 360)
            use_degrees = random.choice([True, False])
        else:  # hard
            # Arbitrary angles and radii, negative angles
            radius = random.uniform(1, 20)
            radius = round(radius, 2)
            angle_deg = random.randint(-720, 720)
            use_degrees = random.choice([True, False])
        
        # Convert to radians for calculation
        angle_rad = angle_deg * np.pi / 180
        
        # Calculate the Cartesian coordinates
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        
        # Round to 4 decimal places
        x = round(float(x), 4)
        y = round(float(y), 4)
        
        # Create the question
        angle_display = angle_deg if use_degrees else round(angle_rad, 4)
        angle_unit = "degrees" if use_degrees else "radians"
        
        question = {
            "type": "polar_to_cartesian",
            "title": "Polar to Cartesian Conversion",
            "question": f"Convert the polar coordinates (r={radius}, θ={angle_display} {angle_unit}) to Cartesian coordinates.",
            "parameters": {
                "radius": radius,
                "angle": angle_display,
                "angle_unit": angle_unit
            },
            "answer": {
                "x": x,
                "y": y
            },
            "solution_steps": [
                f"For polar coordinates (r={radius}, θ={angle_display} {angle_unit}):",
                f"1. {'No conversion needed' if not use_degrees else f'First convert to radians: θ = {angle_display} × (π/180) = {round(angle_rad, 4)} radians'}",
                f"2. Calculate x = r⋅cos(θ) = {radius} × cos({round(angle_rad, 4)}) = {x}",
                f"3. Calculate y = r⋅sin(θ) = {radius} × sin({round(angle_rad, 4)}) = {y}",
                f"4. The Cartesian coordinates are ({x}, {y})"
            ]
        }
        
        return question
    
    def generate_vector_normalization(self, difficulty="medium"):
        """Generate a vector normalization question."""
        if difficulty == "easy":
            # Simple 2D vectors with integer components
            vector = [random.randint(-10, 10), random.randint(-10, 10)]
            while vector[0] == 0 and vector[1] == 0:  # Avoid zero vector
                vector = [random.randint(-10, 10), random.randint(-10, 10)]
        elif difficulty == "medium":
            # 3D vectors with integer components
            vector = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
            while all(v == 0 for v in vector):  # Avoid zero vector
                vector = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
        else:  # hard
            # Higher-dimensional vectors with decimal components
            dim = random.randint(3, 5)
            vector = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
            while all(abs(v) < 0.1 for v in vector):  # Avoid near-zero vector
                vector = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
        
        # Calculate the normalized vector
        magnitude = np.sqrt(sum(v**2 for v in vector))
        normalized = [round(v / magnitude, 6) for v in vector]
        
        # Create the question
        question = {
            "type": "vector_normalization",
            "title": "Vector Normalization",
            "question": f"Normalize the vector v = {vector} to a unit vector.",
            "parameters": {
                "vector": vector
            },
            "answer": {
                "normalized_vector": normalized,
                "magnitude": round(magnitude, 6)
            },
            "solution_steps": [
                f"To normalize a vector, we divide each component by the vector's magnitude.",
                f"1. Find the magnitude: |v| = √({' + '.join(f'({v})²' for v in vector)}) = {round(magnitude, 6)}",
                f"2. Divide each component by the magnitude:",
                f"   v̂ = v / |v| = {vector} / {round(magnitude, 6)} = {normalized}",
                f"3. Verify: The magnitude of the normalized vector should be 1"
            ]
        }
        
        return question
    
    def generate_vector_projection(self, difficulty="medium"):
        """Generate a vector projection (shadow) question."""
        if difficulty == "easy":
            # Simple 2D vectors with integer components
            a = [random.randint(-10, 10), random.randint(-10, 10)]
            b = [random.randint(-10, 10), random.randint(-10, 10)]
            while a[0] == 0 and a[1] == 0:  # Avoid zero vector for a
                a = [random.randint(-10, 10), random.randint(-10, 10)]
        elif difficulty == "medium":
            # 3D vectors with integer components
            a = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
            b = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
            while all(v == 0 for v in a):  # Avoid zero vector for a
                a = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
        else:  # hard
            # 3D vectors with decimal components
            a = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            b = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            while all(abs(v) < 0.1 for v in a):  # Avoid near-zero vector for a
                a = [round(random.uniform(-10, 10), 2) for _ in range(3)]
        
        # Calculate the projection
        a_np = np.array(a)
        b_np = np.array(b)
        
        # Scalar projection (length of the shadow)
        scalar_proj = np.dot(a_np, b_np) / np.linalg.norm(a_np)
        
        # Vector projection (the shadow vector itself)
        vector_proj = (np.dot(a_np, b_np) / np.dot(a_np, a_np)) * a_np
        vector_proj = [round(v, 6) for v in vector_proj]
        
        # Create the question
        question = {
            "type": "vector_projection",
            "title": "Vector Projection (Shadow)",
            "question": f"Calculate the projection (shadow) of vector b = {b} onto vector a = {a}.",
            "parameters": {
                "vector_a": a,
                "vector_b": b
            },
            "answer": {
                "scalar_projection": round(scalar_proj, 6),
                "vector_projection": vector_proj
            },
            "solution_steps": [
                f"To find the projection of b onto a:",
                f"1. Calculate the dot product a·b = {' + '.join(f'({a[i]}×{b[i]})' for i in range(len(a)))} = {round(np.dot(a_np, b_np), 6)}",
                f"2. Calculate |a| = √({' + '.join(f'({v})²' for v in a)}) = {round(np.linalg.norm(a_np), 6)}",
                f"3. The scalar projection (length of shadow) = (a·b)/|a| = {round(np.dot(a_np, b_np), 6)}/{round(np.linalg.norm(a_np), 6)} = {round(scalar_proj, 6)}",
                f"4. The vector projection = ((a·b)/(a·a))a = ({round(np.dot(a_np, b_np), 6)}/{round(np.dot(a_np, a_np), 6)})×{a} = {vector_proj}"
            ]
        }
        
        return question
    
    def generate_orthogonal_vectors(self, difficulty="medium"):
        """Generate a question to determine if vectors are orthogonal."""
        if difficulty == "easy":
            # Generate actually orthogonal vectors (2D)
            a = [random.randint(-10, 10), random.randint(-10, 10)]
            b = [-a[1], a[0]]  # Perpendicular to a
            are_orthogonal = True
        elif difficulty == "medium":
            # Mix of orthogonal and non-orthogonal vectors (3D)
            a = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
            if random.choice([True, False]):
                # Generate orthogonal vector
                if a[0] != 0 and a[1] != 0:
                    b = [a[1], -a[0], 0]  # One possible orthogonal vector
                else:
                    b = [1, 0, 0] if a[0] == 0 else [0, 1, 0]
                are_orthogonal = True
            else:
                # Generate non-orthogonal vector
                b = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
                while np.dot(a, b) == 0:  # Ensure it's not orthogonal
                    b = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
                are_orthogonal = False
        else:  # hard
            # Higher dimensional vectors
            dim = random.randint(3, 5)
            a = [random.randint(-10, 10) for _ in range(dim)]
            if random.choice([True, False]):
                # Create orthogonal vector by solving a system
                b = [0] * dim
                nonzero_indices = [i for i, val in enumerate(a) if val != 0]
                if nonzero_indices:
                    for i in range(dim):
                        if i != nonzero_indices[0]:
                            b[i] = random.randint(-10, 10)
                    # Set the value at the nonzero index to make dot product zero
                    b[nonzero_indices[0]] = -sum(a[i] * b[i] for i in range(dim) if i != nonzero_indices[0]) / a[nonzero_indices[0]]
                    b[nonzero_indices[0]] = round(b[nonzero_indices[0]])
                else:
                    # If a is zero vector, any non-zero vector is fine
                    b = [random.randint(1, 10) for _ in range(dim)]
                are_orthogonal = True
            else:
                # Generate non-orthogonal vector
                b = [random.randint(-10, 10) for _ in range(dim)]
                while np.dot(a, b) == 0:  # Ensure it's not orthogonal
                    b = [random.randint(-10, 10) for _ in range(dim)]
                are_orthogonal = False
        
        # Calculate the dot product
        dot_product = sum(a[i] * b[i] for i in range(len(a)))
        
        # Create the question
        question = {
            "type": "orthogonal_vectors",
            "title": "Orthogonal Vectors",
            "question": f"Determine if the vectors a = {a} and b = {b} are orthogonal (perpendicular) to each other.",
            "parameters": {
                "vector_a": a,
                "vector_b": b
            },
            "answer": {
                "are_orthogonal": are_orthogonal,
                "dot_product": dot_product
            },
            "solution_steps": [
                f"To check if vectors are orthogonal, we calculate their dot product.",
                f"1. Calculate a·b = {' + '.join(f'({a[i]}×{b[i]})' for i in range(len(a)))} = {dot_product}",
                f"2. Vectors are orthogonal if and only if their dot product equals zero.",
                f"3. Since a·b = {dot_product}, the vectors are {'orthogonal' if are_orthogonal else 'not orthogonal'}."
            ]
        }
        
        return question
    
    def generate_vector_angle(self, difficulty="medium"):
        """Generate a question to calculate the angle between two vectors."""
        if difficulty == "easy":
            # Simple 2D vectors with common angles
            # Generate vectors that make a common angle like 0°, 45°, 90°, 180°
            angle_deg = random.choice([0, 45, 90, 180])
            angle_rad = angle_deg * np.pi / 180
            
            # Create a simple first vector
            a = [random.randint(1, 5), 0]
            
            # Create second vector based on angle
            if angle_deg == 0:
                b = [a[0], 0]  # Same direction
            elif angle_deg == 180:
                b = [-a[0], 0]  # Opposite direction
            elif angle_deg == 90:
                b = [0, a[0]]  # Perpendicular
            else:  # 45 degrees
                b = [a[0], a[0]]  # 45 degrees
        
        elif difficulty == "medium":
            # 3D vectors with more angles
            # Generate random 3D vectors and compute their angle
            a = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
            b = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
            
            # Make sure they're not zero vectors
            while all(v == 0 for v in a) or all(v == 0 for v in b):
                a = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
                b = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]
            
            # Calculate the angle
            a_np = np.array(a)
            b_np = np.array(b)
            cos_angle = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
            # Clip to handle small floating-point errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = angle_rad * 180 / np.pi
            
        else:  # hard
            # Higher dimensional vectors with arbitrary angles
            dim = random.randint(3, 5)
            a = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
            b = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
            
            # Make sure they're not near-zero vectors
            while all(abs(v) < 0.1 for v in a) or all(abs(v) < 0.1 for v in b):
                a = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
                b = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
            
            # Calculate the angle
            a_np = np.array(a)
            b_np = np.array(b)
            cos_angle = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
            # Clip to handle small floating-point errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = angle_rad * 180 / np.pi
        
        # Calculate the values for the solution
        a_np = np.array(a)
        b_np = np.array(b)
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        cos_angle = dot_product / (norm_a * norm_b)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # To handle small floating-point errors
        angle_rad = np.arccos(cos_angle)
        angle_deg = angle_rad * 180 / np.pi
        
        # Create the question
        question = {
            "type": "vector_angle",
            "title": "Angle Between Vectors",
            "question": f"Calculate the angle between the vectors a = {a} and b = {b} in both radians and degrees.",
            "parameters": {
                "vector_a": a,
                "vector_b": b
            },
            "answer": {
                "angle_radians": round(float(angle_rad), 6),
                "angle_degrees": round(float(angle_deg), 6)
            },
            "solution_steps": [
                f"To find the angle between two vectors, we use the formula: cos(θ) = (a·b)/(|a|×|b|)",
                f"1. Calculate the dot product: a·b = {' + '.join(f'({a[i]}×{b[i]})' for i in range(len(a)))} = {round(dot_product, 6)}",
                f"2. Calculate |a| = √({' + '.join(f'({v})²' for v in a)}) = {round(norm_a, 6)}",
                f"3. Calculate |b| = √({' + '.join(f'({v})²' for v in b)}) = {round(norm_b, 6)}",
                f"4. cos(θ) = {round(dot_product, 6)}/({round(norm_a, 6)}×{round(norm_b, 6)}) = {round(cos_angle, 6)}",
                f"5. θ = arccos({round(cos_angle, 6)}) = {round(angle_rad, 6)} radians",
                f"6. Convert to degrees: θ = {round(angle_rad, 6)} × (180°/π) = {round(angle_deg, 6)}°"
            ]
        }
        
        return question
    
    def generate_cross_product(self, difficulty="medium"):
        """Generate a cross product question."""
        if difficulty == "easy":
            # Simple 3D vectors with integer components, one with zeros
            a = [random.randint(-5, 5), 0, 0]
            b = [0, random.randint(-5, 5), random.randint(-5, 5)]
            # Make sure at least one component is non-zero in each vector
            while all(v == 0 for v in a):
                a = [random.randint(-5, 5), 0, 0]
            while all(v == 0 for v in b):
                b = [0, random.randint(-5, 5), random.randint(-5, 5)]
        
        elif difficulty == "medium":
            # 3D vectors with integer components
            a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            # Make sure at least one component is non-zero in each vector
            while all(v == 0 for v in a):
                a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            while all(v == 0 for v in b):
                b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
        
        else:  # hard
            # 3D vectors with decimal components
            a = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            b = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            # Make sure they're not near-zero vectors
            while all(abs(v) < 0.1 for v in a):
                a = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            while all(abs(v) < 0.1 for v in b):
                b = [round(random.uniform(-10, 10), 2) for _ in range(3)]
        
        # Calculate the cross product
        cross = np.cross(a, b)
        cross = [round(v, 6) for v in cross]
        
        # Create the question
        question = {
            "type": "cross_product",
            "title": "Vector Cross Product",
            "question": f"Calculate the cross product a × b of the vectors a = {a} and b = {b}.",
            "parameters": {
                "vector_a": a,
                "vector_b": b
            },
            "answer": {
                "cross_product": cross
            },
            "solution_steps": [
                f"To calculate the cross product a × b, we use the formula:",
                f"a × b = [a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁]",
                f"1. Calculate the first component: a₂b₃ - a₃b₂ = ({a[1]}×{b[2]}) - ({a[2]}×{b[1]}) = {round(a[1]*b[2] - a[2]*b[1], 6)}",
                f"2. Calculate the second component: a₃b₁ - a₁b₃ = ({a[2]}×{b[0]}) - ({a[0]}×{b[2]}) = {round(a[2]*b[0] - a[0]*b[2], 6)}",
                f"3. Calculate the third component: a₁b₂ - a₂b₁ = ({a[0]}×{b[1]}) - ({a[1]}×{b[0]}) = {round(a[0]*b[1] - a[1]*b[0], 6)}",
                f"4. Therefore, a × b = {cross}"
            ]
        }
        
        return question
    
    def generate_triangle_area(self, difficulty="medium"):
        """Generate a triangle area calculation question."""
        if difficulty == "easy":
            # Simple 2D or 3D triangles with integer coordinates
            # Generate a triangle in a coordinate plane (z=0 for 3D)
            A = [random.randint(-5, 5), random.randint(-5, 5), 0]
            B = [random.randint(-5, 5), random.randint(-5, 5), 0]
            C = [random.randint(-5, 5), random.randint(-5, 5), 0]
            # Ensure the triangle is not degenerate
            while np.linalg.norm(np.cross(np.array(B) - np.array(A), np.array(C) - np.array(A))) < 0.1:
                A = [random.randint(-5, 5), random.randint(-5, 5), 0]
                B = [random.randint(-5, 5), random.randint(-5, 5), 0]
                C = [random.randint(-5, 5), random.randint(-5, 5), 0]
            
            # For 2D points, drop the z-coordinate
            is_3d = False
            A = A[:2]
            B = B[:2]
            C = C[:2]
        
        elif difficulty == "medium":
            # 3D triangles with integer coordinates
            A = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            B = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            C = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            # Ensure the triangle is not degenerate
            while np.linalg.norm(np.cross(np.array(B) - np.array(A), np.array(C) - np.array(A))) < 0.1:
                A = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                B = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                C = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            
            is_3d = True
        
        else:  # hard
            # 3D triangles with decimal coordinates
            A = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            B = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            C = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            # Ensure the triangle is not degenerate
            while np.linalg.norm(np.cross(np.array(B) - np.array(A), np.array(C) - np.array(A))) < 0.1:
                A = [round(random.uniform(-10, 10), 2) for _ in range(3)]
                B = [round(random.uniform(-10, 10), 2) for _ in range(3)]
                C = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            
            is_3d = True
        
        # Calculate the area
        if is_3d:
            AB = np.array(B) - np.array(A)
            AC = np.array(C) - np.array(A)
            area = np.linalg.norm(np.cross(AB, AC)) / 2
        else:
            # For 2D triangles, we can use the cross product or the Shoelace formula
            # Here we're using the fact that the magnitude of the cross product
            # of two 2D vectors (treated as 3D with z=0) equals twice the area
            A_3d = A + [0]
            B_3d = B + [0]
            C_3d = C + [0]
            AB = np.array(B_3d) - np.array(A_3d)
            AC = np.array(C_3d) - np.array(A_3d)
            area = np.linalg.norm(np.cross(AB, AC)) / 2
        
        area = round(float(area), 6)
        
        # Create the question
        dimension = "3D" if is_3d else "2D"
        question = {
            "type": "triangle_area",
            "title": f"Triangle Area Calculation ({dimension})",
            "question": f"Calculate the area of the triangle with vertices A = {A}, B = {B}, and C = {C}.",
            "parameters": {
                "point_A": A,
                "point_B": B,
                "point_C": C,
                "is_3d": is_3d
            },
            "answer": {
                "area": area
            },
            "solution_steps": [
                f"To calculate the area of a triangle using vectors:",
                f"1. Create vectors from point A to points B and C:",
                f"   AB = B - A = {B} - {A} = {(np.array(B) - np.array(A)).tolist()}",
                f"   AC = C - A = {C} - {A} = {(np.array(C) - np.array(A)).tolist()}",
                f"2. Calculate the cross product of these vectors:",
                f"   AB × AC = {np.cross(np.array(B) - np.array(A), np.array(C) - np.array(A)).tolist() if is_3d else np.cross(np.array(B + [0]) - np.array(A + [0]), np.array(C + [0]) - np.array(A + [0])).tolist()}",
                f"3. Calculate the magnitude of the cross product and divide by 2:",
                f"   Area = |AB × AC| / 2 = {np.linalg.norm(np.cross(np.array(B) - np.array(A), np.array(C) - np.array(A))) if is_3d else np.linalg.norm(np.cross(np.array(B + [0]) - np.array(A + [0]), np.array(C + [0]) - np.array(A + [0])))} / 2 = {area}"
            ]
        }
        
        return question
    
    def generate_point_line_distance(self, difficulty="medium"):
        """Generate a point-line distance calculation question."""
        if difficulty == "easy":
            # 2D line along an axis
            A = [random.randint(-5, 5), 0, 0]  # Point on the line (x-axis)
            v = [1, 0, 0]  # Direction vector (along x-axis)
            B = [random.randint(-5, 5), random.randint(1, 5), 0]  # Point (off the line)
            
            # For 2D points, drop the z-coordinate
            is_3d = False
            A = A[:2]
            v = v[:2]
            B = B[:2]
        
        elif difficulty == "medium":
            # 2D or 3D line with integer components
            is_3d = random.choice([True, False])
            dim = 3 if is_3d else 2
            
            A = [random.randint(-5, 5) for _ in range(dim)]  # Point on the line
            v = [random.randint(-5, 5) for _ in range(dim)]  # Direction vector
            # Make sure the direction vector is not zero
            while all(val == 0 for val in v):
                v = [random.randint(-5, 5) for _ in range(dim)]
            
            # Generate point B
            B = [random.randint(-5, 5) for _ in range(dim)]
            # Make sure B is not on the line
            while np.linalg.norm(np.cross(np.array(v), np.array(B) - np.array(A))) < 0.1:
                B = [random.randint(-5, 5) for _ in range(dim)]
        
        else:  # hard
            # 3D line with decimal components
            is_3d = True
            
            A = [round(random.uniform(-10, 10), 2) for _ in range(3)]  # Point on the line
            v = [round(random.uniform(-10, 10), 2) for _ in range(3)]  # Direction vector
            # Make sure the direction vector is not near zero
            while all(abs(val) < 0.1 for val in v):
                v = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            
            # Generate point B
            B = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            # Make sure B is not on the line
            while np.linalg.norm(np.cross(np.array(v), np.array(B) - np.array(A))) < 0.1:
                B = [round(random.uniform(-10, 10), 2) for _ in range(3)]
        
        # Calculate the distance
        if is_3d:
            # 3D distance
            distance = np.linalg.norm(np.cross(np.array(v), np.array(B) - np.array(A))) / np.linalg.norm(np.array(v))
        else:
            # 2D distance (treat as 3D with z=0)
            A_3d = A + [0]
            v_3d = v + [0]
            B_3d = B + [0]
            distance = np.linalg.norm(np.cross(np.array(v_3d), np.array(B_3d) - np.array(A_3d))) / np.linalg.norm(np.array(v_3d))
        
        distance = round(float(distance), 6)
        
        # Create the question
        dimension = "3D" if is_3d else "2D"
        question = {
            "type": "point_line_distance",
            "title": f"Point to Line Distance ({dimension})",
            "question": f"Calculate the shortest distance from point B = {B} to the line passing through point A = {A} in the direction of vector v = {v}.",
            "parameters": {
                "point_A": A,
                "direction_v": v,
                "point_B": B,
                "is_3d": is_3d
            },
            "answer": {
                "distance": distance
            },
            "solution_steps": [
                f"To calculate the distance from a point to a line in {dimension}:",
                f"1. The line is defined by point A = {A} and direction vector v = {v}",
                f"2. The point is B = {B}",
                f"3. The formula for the distance is: d = |v × (B - A)| / |v|",
                f"4. Calculate B - A = {B} - {A} = {(np.array(B) - np.array(A)).tolist()}",
                f"5. Calculate the cross product v × (B - A) = {np.cross(np.array(v + [0] if not is_3d else v), np.array(B + [0] if not is_3d else B) - np.array(A + [0] if not is_3d else A)).tolist()}",
                f"6. Calculate |v × (B - A)| = {np.linalg.norm(np.cross(np.array(v + [0] if not is_3d else v), np.array(B + [0] if not is_3d else B) - np.array(A + [0] if not is_3d else A)))}",
                f"7. Calculate |v| = {np.linalg.norm(np.array(v))}",
                f"8. Distance = |v × (B - A)| / |v| = {np.linalg.norm(np.cross(np.array(v + [0] if not is_3d else v), np.array(B + [0] if not is_3d else B) - np.array(A + [0] if not is_3d else A)))} / {np.linalg.norm(np.array(v))} = {distance}"
            ]
        }
        
        return question
    
    def generate_vector_collinearity(self, difficulty="medium"):
        """Generate a vector collinearity test question."""
        if difficulty == "easy":
            # Simple 2D vectors, one is actually a multiple of the other
            a = [random.randint(-5, 5), random.randint(-5, 5)]
            # Make sure a is not a zero vector
            while all(v == 0 for v in a):
                a = [random.randint(-5, 5), random.randint(-5, 5)]
            
            # Choose a scalar multiple
            scalar = random.randint(-3, 3)
            while scalar == 0:  # Avoid zero
                scalar = random.randint(-3, 3)
            
            b = [scalar * x for x in a]
            
            # The vectors are collinear by construction
            are_collinear = True
        
        elif difficulty == "medium":
            # 3D vectors, might be collinear or not
            a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            # Make sure a is not a zero vector
            while all(v == 0 for v in a):
                a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            
            if random.choice([True, False]):
                # Generate a collinear vector
                scalar = random.randint(-3, 3)
                while scalar == 0:  # Avoid zero
                    scalar = random.randint(-3, 3)
                
                b = [scalar * x for x in a]
                are_collinear = True
            else:
                # Generate a non-collinear vector
                b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                # Check if they are accidentally collinear
                rank = np.linalg.matrix_rank(np.array([a, b]))
                while rank < 2:  # If rank < 2, they are collinear
                    b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                    rank = np.linalg.matrix_rank(np.array([a, b]))
                
                are_collinear = False
        
        else:  # hard
            # Higher dimensional vectors with decimal components
            dim = random.randint(3, 5)
            a = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
            # Make sure a is not a near-zero vector
            while all(abs(v) < 0.1 for v in a):
                a = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
            
            if random.choice([True, False]):
                # Generate a collinear vector
                scalar = round(random.uniform(-3, 3), 2)
                while abs(scalar) < 0.1:  # Avoid near-zero
                    scalar = round(random.uniform(-3, 3), 2)
                
                b = [scalar * x for x in a]
                are_collinear = True
            else:
                # Generate a non-collinear vector
                b = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
                # Check if they are accidentally collinear
                rank = np.linalg.matrix_rank(np.array([a, b]))
                while rank < 2:  # If rank < 2, they are collinear
                    b = [round(random.uniform(-10, 10), 2) for _ in range(dim)]
                    rank = np.linalg.matrix_rank(np.array([a, b]))
                
                are_collinear = False
        
        # Calculate the matrix rank for the solution
        matrix = np.array([a, b])
        rank = np.linalg.matrix_rank(matrix)
        
        # Create the question
        question = {
            "type": "vector_collinearity",
            "title": "Vector Collinearity Test",
            "question": f"Determine if the vectors a = {a} and b = {b} are collinear (one is a scalar multiple of the other).",
            "parameters": {
                "vector_a": a,
                "vector_b": b
            },
            "answer": {
                "are_collinear": are_collinear
            },
            "solution_steps": [
                f"To determine if vectors are collinear, we check if one is a scalar multiple of the other.",
                f"1. We can do this by checking the rank of the matrix formed by these vectors.",
                f"2. The matrix is: [a, b] = [{a}, {b}]",
                f"3. The rank of this matrix is {rank}.",
                f"4. If the rank is 1, the vectors are collinear (linearly dependent).",
                f"5. If the rank is 2, the vectors are not collinear (linearly independent).",
                f"6. Since the rank is {rank}, the vectors are {'collinear' if are_collinear else 'not collinear'}."
            ]
        }
        
        return question
    
    def generate_gaussian_elimination(self, difficulty="medium"):
        """Generate a Gaussian elimination question (solving a system of linear equations)."""
        if difficulty == "easy":
            # Simple 2×2 system with integer solutions
            x_sol = random.randint(-5, 5)
            y_sol = random.randint(-5, 5)
            
            # Create a system with a unique solution
            a11 = random.randint(-5, 5)
            a12 = random.randint(-5, 5)
            a21 = random.randint(-5, 5)
            a22 = random.randint(-5, 5)
            
            # Ensure the determinant is non-zero
            while a11 * a22 - a12 * a21 == 0:
                a11 = random.randint(-5, 5)
                a12 = random.randint(-5, 5)
                a21 = random.randint(-5, 5)
                a22 = random.randint(-5, 5)
            
            # Calculate the right-hand side
            b1 = a11 * x_sol + a12 * y_sol
            b2 = a21 * x_sol + a22 * y_sol
            
            # Create the augmented matrix
            augmented_matrix = [[a11, a12, b1], 
                                [a21, a22, b2]]
            
            # Solution
            solution = [x_sol, y_sol]
            solution_type = "unique"
        
        elif difficulty == "medium":
            # 3×3 system with integer or no solution
            has_solution = random.choice([True, False])
            
            if has_solution:
                # System with unique solution
                x_sol = random.randint(-5, 5)
                y_sol = random.randint(-5, 5)
                z_sol = random.randint(-5, 5)
                
                # Create a system with a unique solution
                a11 = random.randint(-5, 5)
                a12 = random.randint(-5, 5)
                a13 = random.randint(-5, 5)
                a21 = random.randint(-5, 5)
                a22 = random.randint(-5, 5)
                a23 = random.randint(-5, 5)
                a31 = random.randint(-5, 5)
                a32 = random.randint(-5, 5)
                a33 = random.randint(-5, 5)
                
                # Calculate the right-hand side
                b1 = a11 * x_sol + a12 * y_sol + a13 * z_sol
                b2 = a21 * x_sol + a22 * y_sol + a23 * z_sol
                b3 = a31 * x_sol + a32 * y_sol + a33 * z_sol
                
                # Create the augmented matrix
                augmented_matrix = [[a11, a12, a13, b1], 
                                    [a21, a22, a23, b2],
                                    [a31, a32, a33, b3]]
                
                # Solution
                solution = [x_sol, y_sol, z_sol]
                solution_type = "unique"
            else:
                # System with no solution
                a11 = random.randint(-5, 5)
                a12 = random.randint(-5, 5)
                a13 = random.randint(-5, 5)
                a21 = random.randint(-5, 5)
                a22 = random.randint(-5, 5)
                a23 = random.randint(-5, 5)
                
                # Make the third row a linear combination of the first two
                scalar1 = random.randint(-3, 3)
                scalar2 = random.randint(-3, 3)
                while scalar1 == 0 and scalar2 == 0:
                    scalar1 = random.randint(-3, 3)
                    scalar2 = random.randint(-3, 3)
                
                a31 = scalar1 * a11 + scalar2 * a21
                a32 = scalar1 * a12 + scalar2 * a22
                a33 = scalar1 * a13 + scalar2 * a23
                
                # Make the system inconsistent
                b1 = random.randint(-10, 10)
                b2 = random.randint(-10, 10)
                b3 = scalar1 * b1 + scalar2 * b2 + random.randint(1, 5)  # Add a non-zero term to make it inconsistent
                
                # Create the augmented matrix
                augmented_matrix = [[a11, a12, a13, b1], 
                                    [a21, a22, a23, b2],
                                    [a31, a32, a33, b3]]
                
                # No solution
                solution = None
                solution_type = "none"
        
        else:  # hard
            # 3×3 system with infinite solutions or harder systems
            solution_type = random.choice(["infinite", "unique", "none"])
            
            if solution_type == "infinite":
                # System with infinite solutions
                # Start with two independent rows
                a11 = random.randint(-5, 5)
                a12 = random.randint(-5, 5)
                a13 = random.randint(-5, 5)
                a21 = random.randint(-5, 5)
                a22 = random.randint(-5, 5)
                a23 = random.randint(-5, 5)
                
                # Make the third row a linear combination of the first two
                scalar1 = random.randint(-3, 3)
                scalar2 = random.randint(-3, 3)
                while scalar1 == 0 and scalar2 == 0:
                    scalar1 = random.randint(-3, 3)
                    scalar2 = random.randint(-3, 3)
                
                a31 = scalar1 * a11 + scalar2 * a21
                a32 = scalar1 * a12 + scalar2 * a22
                a33 = scalar1 * a13 + scalar2 * a23
                
                # Make the system consistent
                b1 = random.randint(-10, 10)
                b2 = random.randint(-10, 10)
                b3 = scalar1 * b1 + scalar2 * b2  # Consistent with the linear dependency
                
                # Create the augmented matrix
                augmented_matrix = [[a11, a12, a13, b1], 
                                    [a21, a22, a23, b2],
                                    [a31, a32, a33, b3]]
                
                # Infinite solutions
                solution = "Infinite solutions with one free parameter"
            
            elif solution_type == "unique":
                # 4×4 system with unique solution
                solution = [random.randint(-3, 3) for _ in range(4)]
                
                # Create a random 4×4 matrix
                A = [[random.randint(-3, 3) for _ in range(4)] for _ in range(4)]
                
                # Calculate the right-hand side
                b = [sum(A[i][j] * solution[j] for j in range(4)) for i in range(4)]
                
                # Create the augmented matrix
                augmented_matrix = [A[i] + [b[i]] for i in range(4)]
            
            else:  # none
                # 4×4 system with no solution
                # Start with a 4×3 matrix
                A = [[random.randint(-3, 3) for _ in range(3)] for _ in range(4)]
                
                # Add a random fourth column
                for i in range(4):
                    A[i].append(random.randint(-3, 3))
                
                # Calculate a consistent right-hand side for the first three equations
                temp_solution = [random.randint(-3, 3) for _ in range(4)]
                b = [sum(A[i][j] * temp_solution[j] for j in range(4)) for i in range(3)]
                
                # Make the last equation inconsistent
                b.append(sum(A[3][j] * temp_solution[j] for j in range(4)) + random.randint(1, 5))
                
                # Create the augmented matrix
                augmented_matrix = [A[i] + [b[i]] for i in range(4)]
                
                solution = None
        
        # Convert the augmented matrix to NumPy for computation
        augmented_np = np.array(augmented_matrix)
        
        # Solve the system using row reduction
        reduced = mrref(augmented_np)
        
        # Create the solution steps
        steps = []
        steps.append(f"To solve this system using Gaussian elimination, we start with the augmented matrix:")
        steps.append(f"{augmented_matrix}")
        steps.append(f"After applying row reduction (Gauss-Jordan elimination), we get:")
        steps.append(f"{reduced.tolist()}")
        
        if solution_type == "unique":
            steps.append(f"This represents a system with a unique solution.")
            if len(augmented_matrix[0]) == 3:  # 2×2 system
                steps.append(f"From the reduced matrix, we can read off the solution: x = {solution[0]}, y = {solution[1]}")
            elif len(augmented_matrix[0]) == 4 and len(augmented_matrix) == 3:  # 3×3 system
                steps.append(f"From the reduced matrix, we can read off the solution: x = {solution[0]}, y = {solution[1]}, z = {solution[2]}")
            else:  # 4×4 system
                steps.append(f"From the reduced matrix, we can read off the solution: x₁ = {solution[0]}, x₂ = {solution[1]}, x₃ = {solution[2]}, x₄ = {solution[3]}")
        
        elif solution_type == "none":
            steps.append(f"This system has no solution (it is inconsistent).")
            steps.append(f"We can see this because there is a row of the form [0 0 ... 0 | k] where k ≠ 0, which represents an equation like 0 = k, which is a contradiction.")
        
        else:  # infinite
            steps.append(f"This system has infinitely many solutions.")
            steps.append(f"We can see this because there are fewer pivot variables than unknowns, resulting in free parameters.")
            steps.append(f"The general solution can be expressed in terms of these free parameters.")
        
        # Create the question
        system_size = f"{len(augmented_matrix)}×{len(augmented_matrix[0])-1}"
        question = {
            "type": "gaussian_elimination",
            "title": f"Solving a {system_size} Linear System with Gaussian Elimination",
            "question": f"Solve the following system of linear equations using Gaussian elimination:\n{augmented_matrix}",
            "parameters": {
                "augmented_matrix": augmented_matrix
            },
            "answer": {
                "solution_type": solution_type,
                "solution": solution
            },
            "solution_steps": steps
        }
        
        return question
    
    def generate_coplanar_vectors(self, difficulty="medium"):
        """Generate a coplanar vectors test question."""
        if difficulty == "easy":
            # Generate 3 vectors where the third is a linear combination of the first two
            a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            
            # Make sure a and b are not zero vectors and not collinear
            while all(v == 0 for v in a) or all(v == 0 for v in b):
                a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            
            # Check if a and b are accidentally collinear
            rank = np.linalg.matrix_rank(np.array([a, b]))
            while rank < 2:  # If rank < 2, they are collinear
                b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                rank = np.linalg.matrix_rank(np.array([a, b]))
            
            # Create a linear combination for c
            scalar1 = random.randint(-3, 3)
            scalar2 = random.randint(-3, 3)
            while scalar1 == 0 and scalar2 == 0:
                scalar1 = random.randint(-3, 3)
                scalar2 = random.randint(-3, 3)
            
            c = [scalar1 * a[i] + scalar2 * b[i] for i in range(3)]
            
            # These vectors are coplanar by construction
            are_coplanar = True
        
        elif difficulty == "medium":
            # Generate 3 vectors that might be coplanar or not
            a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            
            # Make sure a and b are not zero vectors and not collinear
            while all(v == 0 for v in a) or all(v == 0 for v in b):
                a = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
            
            # Check if a and b are accidentally collinear
            rank = np.linalg.matrix_rank(np.array([a, b]))
            while rank < 2:  # If rank < 2, they are collinear
                b = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                rank = np.linalg.matrix_rank(np.array([a, b]))
            
            if random.choice([True, False]):
                # Create a coplanar vector (linear combination)
                scalar1 = random.randint(-3, 3)
                scalar2 = random.randint(-3, 3)
                while scalar1 == 0 and scalar2 == 0:
                    scalar1 = random.randint(-3, 3)
                    scalar2 = random.randint(-3, 3)
                
                c = [scalar1 * a[i] + scalar2 * b[i] for i in range(3)]
                are_coplanar = True
            else:
                # Create a non-coplanar vector
                c = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                
                # Check if c is accidentally coplanar with a and b
                matrix = np.array([a, b, c])
                rank = np.linalg.matrix_rank(matrix)
                
                while rank < 3:  # If rank < 3, they are coplanar
                    c = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                    matrix = np.array([a, b, c])
                    rank = np.linalg.matrix_rank(matrix)
                
                are_coplanar = False
        
        else:  # hard
            # Generate 3 vectors with decimal components
            a = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            b = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            
            # Make sure a and b are not near-zero vectors
            while all(abs(v) < 0.1 for v in a) or all(abs(v) < 0.1 for v in b):
                a = [round(random.uniform(-10, 10), 2) for _ in range(3)]
                b = [round(random.uniform(-10, 10), 2) for _ in range(3)]
            
            # Check if a and b are accidentally nearly collinear
            rank = np.linalg.matrix_rank(np.array([a, b]))
            while rank < 2:  # If rank < 2, they are collinear
                b = [round(random.uniform(-10, 10), 2) for _ in range(3)]
                rank = np.linalg.matrix_rank(np.array([a, b]))
            
            if random.choice([True, False]):
                # Create a coplanar vector (linear combination with decimals)
                scalar1 = round(random.uniform(-3, 3), 2)
                scalar2 = round(random.uniform(-3, 3), 2)
                while abs(scalar1) < 0.1 and abs(scalar2) < 0.1:
                    scalar1 = round(random.uniform(-3, 3), 2)
                    scalar2 = round(random.uniform(-3, 3), 2)
                
                c = [scalar1 * a[i] + scalar2 * b[i] for i in range(3)]
                c = [round(v, 2) for v in c]  # Round to 2 decimal places
                are_coplanar = True
            else:
                # Create a non-coplanar vector
                c = [round(random.uniform(-10, 10), 2) for _ in range(3)]
                
                # Check if c is accidentally coplanar with a and b
                matrix = np.array([a, b, c])
                rank = np.linalg.matrix_rank(matrix)
                
                while rank < 3:  # If rank < 3, they are coplanar
                    c = [round(random.uniform(-10, 10), 2) for _ in range(3)]
                    matrix = np.array([a, b, c])
                    rank = np.linalg.matrix_rank(matrix)
                
                are_coplanar = False
        
        # Calculate the matrix rank for the solution
        matrix = np.array([a, b, c])
        rank = np.linalg.matrix_rank(matrix)
        
        # Create the question
        question = {
            "type": "coplanar_vectors",
            "title": "Coplanar Vectors Test",
            "question": f"Determine if the vectors a = {a}, b = {b}, and c = {c} are coplanar (lie in the same plane).",
            "parameters": {
                "vector_a": a,
                "vector_b": b,
                "vector_c": c
            },
            "answer": {
                "are_coplanar": are_coplanar
            },
            "solution_steps": [
                f"To determine if three vectors in 3D space are coplanar, we check if one is a linear combination of the other two.",
                f"1. We can do this by checking the rank of the matrix formed by these vectors.",
                f"2. The matrix is: [a, b, c] = [{a}, {b}, {c}]",
                f"3. The rank of this matrix is {rank}.",
                f"4. If the rank is less than 3, the vectors are coplanar (lie in the same plane).",
                f"5. If the rank is 3, the vectors are not coplanar (they span the entire 3D space).",
                f"6. Since the rank is {rank}, the vectors are {'coplanar' if are_coplanar else 'not coplanar'}."
            ]
        }
        
        return question
    
    def generate_matrix_product(self, difficulty="medium"):
        """Generate a matrix multiplication question."""
        if difficulty == "easy":
            # 2×2 matrices with small integer entries
            rows_a = 2
            cols_a = 2
            rows_b = 2
            cols_b = 2
            
            # Generate matrices with small integer entries
            A = [[random.randint(-3, 3) for _ in range(cols_a)] for _ in range(rows_a)]
            B = [[random.randint(-3, 3) for _ in range(cols_b)] for _ in range(rows_b)]
        
        elif difficulty == "medium":
            # 2×3 and 3×2 matrices or similar
            rows_a = random.choice([2, 3])
            cols_a = random.choice([2, 3])
            rows_b = cols_a  # For compatibility
            cols_b = random.choice([2, 3])
            
            # Generate matrices with integer entries
            A = [[random.randint(-5, 5) for _ in range(cols_a)] for _ in range(rows_a)]
            B = [[random.randint(-5, 5) for _ in range(cols_b)] for _ in range(rows_b)]
        
        else:  # hard
            # Larger matrices or matrices with decimal entries
            rows_a = random.choice([2, 3, 4])
            cols_a = random.choice([2, 3, 4])
            rows_b = cols_a  # For compatibility
            cols_b = random.choice([2, 3, 4])
            
            # Generate matrices with decimal entries
            A = [[round(random.uniform(-5, 5), 1) for _ in range(cols_a)] for _ in range(rows_a)]
            B = [[round(random.uniform(-5, 5), 1) for _ in range(cols_b)] for _ in range(rows_b)]
        
        # Calculate the matrix product
        A_np = np.array(A)
        B_np = np.array(B)
        C_np = np.dot(A_np, B_np)
        
        # Convert to list of lists and round elements
        C = [[round(float(C_np[i, j]), 4) for j in range(cols_b)] for i in range(rows_a)]
        
        # Create the question
        question = {
            "type": "matrix_product",
            "title": "Matrix Multiplication",
            "question": f"Calculate the product of matrices A = {A} and B = {B}.",
            "parameters": {
                "matrix_A": A,
                "matrix_B": B,
                "rows_A": rows_a,
                "cols_A": cols_a,
                "rows_B": rows_b,
                "cols_B": cols_b
            },
            "answer": {
                "product": C,
                "dimensions": [rows_a, cols_b]
            },
            "solution_steps": [
                f"To multiply matrices A ({rows_a}×{cols_a}) and B ({rows_b}×{cols_b}):",
                f"1. First check that multiplication is possible: the number of columns in A ({cols_a}) must equal the number of rows in B ({rows_b}).",
                f"2. The result will be a {rows_a}×{cols_b} matrix.",
                f"3. Each element C[i,j] is calculated as the dot product of row i from A and column j from B."
            ]
        }
        
        # Add specific calculation steps for smaller matrices
        if rows_a <= 3 and cols_b <= 3:
            for i in range(rows_a):
                for j in range(cols_b):
                    row_A = A[i]
                    col_B = [B[k][j] for k in range(rows_b)]
                    
                    dot_product_terms = [f"({row_A[k]}×{col_B[k]})" for k in range(cols_a)]
                    dot_product_calc = " + ".join(dot_product_terms)
                    dot_product_result = sum(row_A[k] * col_B[k] for k in range(cols_a))
                    
                    question["solution_steps"].append(f"C[{i+1},{j+1}] = {dot_product_calc} = {round(dot_product_result, 4)}")
        
        question["solution_steps"].append(f"4. Therefore, the product A×B = {C}")
        
        return question