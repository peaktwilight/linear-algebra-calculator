#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical utilities for the Streamlit Linear Algebra Calculator
Self-sufficient parsing and computation functions
"""

import numpy as np
import sympy as sym
import re
from typing import Optional, Union

class MathUtils:
    """Self-sufficient mathematical utilities for linear algebra operations"""
    
    @staticmethod
    def parse_matrix(matrix_str: str) -> Optional[np.ndarray]:
        """
        Parse a matrix from string input
        Supports formats:
        - "1,2,3; 4,5,6" (semicolon separated rows)
        - "1,2,3\n4,5,6" (newline separated rows)
        - "1 2 3\n4 5 6" (space separated)
        """
        if not matrix_str or not matrix_str.strip():
            return None
            
        try:
            # Clean the input
            matrix_str = matrix_str.strip()
            
            # Handle different row separators
            if ';' in matrix_str:
                rows = matrix_str.split(';')
            else:
                rows = matrix_str.split('\n')
            
            matrix_rows = []
            for row in rows:
                row = row.strip()
                if not row:
                    continue
                    
                # Handle different column separators
                if ',' in row:
                    elements = [float(x.strip()) for x in row.split(',') if x.strip()]
                else:
                    # Space separated
                    elements = [float(x.strip()) for x in row.split() if x.strip()]
                
                if elements:
                    matrix_rows.append(elements)
            
            if not matrix_rows:
                return None
                
            return np.array(matrix_rows)
            
        except (ValueError, TypeError) as e:
            print(f"Error parsing matrix: {e}")
            return None
    
    @staticmethod
    def parse_vector(vector_str: str) -> Optional[np.ndarray]:
        """
        Parse a vector from string input
        Supports formats:
        - "1,2,3" (comma separated)
        - "[1,2,3]" (bracket notation)
        - "1 2 3" (space separated)
        """
        if not vector_str or not vector_str.strip():
            return None
            
        try:
            # Clean the input
            vector_str = vector_str.strip()
            
            # Remove brackets if present
            vector_str = vector_str.strip('[](){}')
            
            # Handle different separators
            if ',' in vector_str:
                elements = [float(x.strip()) for x in vector_str.split(',') if x.strip()]
            else:
                # Space separated
                elements = [float(x.strip()) for x in vector_str.split() if x.strip()]
            
            if not elements:
                return None
                
            return np.array(elements)
            
        except (ValueError, TypeError) as e:
            print(f"Error parsing vector: {e}")
            return None
    
    @staticmethod
    def format_number_latex(num: Union[int, float, np.number]) -> str:
        """Format a number for LaTeX display"""
        if isinstance(num, (int, np.integer)):
            return str(num)
        elif isinstance(num, (float, np.floating)):
            if np.isclose(num, round(num)):
                return str(int(round(num)))
            else:
                return f"{num:.6f}".rstrip('0').rstrip('.')
        else:
            return str(num)
    
    @staticmethod
    def format_matrix_latex(matrix: np.ndarray) -> str:
        """Format a matrix for LaTeX display"""
        if matrix is None or matrix.size == 0:
            return "(Empty matrix)"
        
        # Clean near-zero values
        matrix_cleaned = np.where(np.isclose(matrix, 0, atol=1e-10), 0, matrix)
        
        rows_str = []
        for row_idx in range(matrix_cleaned.shape[0]):
            elements = [MathUtils.format_number_latex(x) for x in matrix_cleaned[row_idx, :]]
            rows_str.append(" & ".join(elements))
        
        return r"\begin{pmatrix} " + r" \\ ".join(rows_str) + r" \end{pmatrix}"
    
    @staticmethod
    def format_vector_latex(vector: np.ndarray) -> str:
        """Format a vector for LaTeX display"""
        if vector is None or vector.size == 0:
            return "(Empty vector)"
        
        # Clean near-zero values
        vector_cleaned = np.where(np.isclose(vector, 0, atol=1e-10), 0, vector)
        elements = [MathUtils.format_number_latex(x) for x in vector_cleaned]
        
        return r"\begin{pmatrix} " + r" \\ ".join(elements) + r" \end{pmatrix}"
    
    @staticmethod
    def vector_magnitude(vector: np.ndarray) -> float:
        """Calculate vector magnitude/norm"""
        return np.linalg.norm(vector)
    
    @staticmethod
    def normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
        """Normalize a vector to unit length"""
        magnitude = MathUtils.vector_magnitude(vector)
        if magnitude == 0:
            return None
        return vector / magnitude
    
    @staticmethod
    def vector_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate dot product of two vectors"""
        return np.dot(v1, v2)
    
    @staticmethod
    def vector_cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Calculate cross product of two vectors"""
        # Extend to 3D if needed
        if len(v1) == 2:
            v1 = np.append(v1, 0)
        if len(v2) == 2:
            v2 = np.append(v2, 0)
        
        return np.cross(v1, v2)
    
    @staticmethod
    def vector_angle(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> float:
        """Calculate angle between two vectors"""
        cos_angle = MathUtils.vector_dot_product(v1, v2) / (
            MathUtils.vector_magnitude(v1) * MathUtils.vector_magnitude(v2)
        )
        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        if degrees:
            return np.degrees(angle_rad)
        return angle_rad
    
    @staticmethod
    def vector_projection(v_onto: np.ndarray, v_project: np.ndarray) -> np.ndarray:
        """Project v_project onto v_onto"""
        return (MathUtils.vector_dot_product(v_project, v_onto) / 
                MathUtils.vector_dot_product(v_onto, v_onto)) * v_onto
    
    @staticmethod
    def matrix_determinant(matrix: np.ndarray) -> float:
        """Calculate matrix determinant"""
        return np.linalg.det(matrix)
    
    @staticmethod
    def matrix_inverse(matrix: np.ndarray) -> Optional[np.ndarray]:
        """Calculate matrix inverse if it exists"""
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None
    
    @staticmethod
    def matrix_eigenvalues_eigenvectors(matrix: np.ndarray) -> tuple:
        """Calculate eigenvalues and eigenvectors"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """Solve linear system Ax = b"""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
    
    @staticmethod
    def triangle_area_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate triangle area from three points using cross product"""
        # Vectors from p1 to p2 and p1 to p3
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Cross product magnitude gives twice the area
        cross_prod = MathUtils.vector_cross_product(v1, v2)
        
        # For 2D, cross product is a scalar; for 3D, take magnitude
        if isinstance(cross_prod, np.ndarray):
            area = 0.5 * np.linalg.norm(cross_prod)
        else:
            area = 0.5 * abs(cross_prod)
        
        return area
    
    @staticmethod
    def point_line_distance(point_on_line: np.ndarray, line_direction: np.ndarray, point: np.ndarray) -> float:
        """Calculate distance from a point to a line defined by a point and direction"""
        # Vector from point on line to the point
        vec_to_point = point - point_on_line
        
        # Cross product gives area of parallelogram, divide by base length for height (distance)
        cross_prod = MathUtils.vector_cross_product(line_direction, vec_to_point)
        
        # Distance is |cross_product| / |direction|
        if isinstance(cross_prod, np.ndarray):
            distance = np.linalg.norm(cross_prod) / np.linalg.norm(line_direction)
        else:
            distance = abs(cross_prod) / np.linalg.norm(line_direction)
        
        return distance