#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plane Operations Mixin for Streamlit Linear Algebra Calculator
Handles 3D plane calculations including plane from 3 points, point-to-plane distance, etc.
"""

import streamlit as st
import numpy as np
from typing import Optional, Tuple
from .st_math_utils import MathUtils

class PlaneOperationsMixin:
    """Mixin class for plane-related operations in 3D space"""
    
    def plane_from_three_points(self, point_input_A: str, point_input_B: str, point_input_C: str):
        """Calculate plane equation from three points"""
        st.subheader("Plane from 3 Points")
        
        try:
            # Parse input points
            A = MathUtils.parse_vector(point_input_A)
            B = MathUtils.parse_vector(point_input_B)
            C = MathUtils.parse_vector(point_input_C)
            
            if A is None or B is None or C is None:
                st.error("Invalid point format. Please enter points as: x, y, z")
                return None
            
            if len(A) != 3 or len(B) != 3 or len(C) != 3:
                st.error("Points must be 3-dimensional (x, y, z)")
                return None
            
            # Display input points
            st.markdown("### Input Points")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Point A:** $${self._vector_to_latex(A)}$$")
            with col2:
                st.markdown(f"**Point B:** $${self._vector_to_latex(B)}$$")
            with col3:
                st.markdown(f"**Point C:** $${self._vector_to_latex(C)}$$")
            
            # Calculate vectors in the plane
            AB = B - A
            AC = C - A
            
            st.markdown("---")
            st.markdown("### Step 1: Vectors in the Plane")
            st.markdown(f"$$\\vec{{AB}} = B - A = {self._vector_to_latex(B)} - {self._vector_to_latex(A)} = {self._vector_to_latex(AB)}$$")
            st.markdown(f"$$\\vec{{AC}} = C - A = {self._vector_to_latex(C)} - {self._vector_to_latex(A)} = {self._vector_to_latex(AC)}$$")
            
            # Calculate normal vector using cross product
            n = np.cross(AB, AC)
            
            st.markdown("---")
            st.markdown("### Step 2: Normal Vector")
            st.markdown(f"$$\\vec{{n}} = \\vec{{AB}} \\times \\vec{{AC}} = {self._vector_to_latex(AB)} \\times {self._vector_to_latex(AC)} = {self._vector_to_latex(n)}$$")
            
            # Check if points are collinear
            if np.allclose(n, 0):
                st.error("⚠️ The points are collinear! They do not define a unique plane.")
                st.markdown("The normal vector is the zero vector, which means the three points lie on the same line.")
                return None
            
            # Calculate the constant term (k = n · A)
            k = np.dot(n, A)
            
            st.markdown("---")
            st.markdown("### Step 3: Plane Equation")
            st.markdown(f"$$k = \\vec{{n}} \\cdot A = {self._vector_to_latex(n)} \\cdot {self._vector_to_latex(A)} = {k:.4f}$$")
            
            # Display the plane equation
            self._display_plane_equation(n, k)
            
            # Verification: Check that all three points satisfy the equation
            st.markdown("---")
            st.markdown("### Verification")
            st.markdown("Let's verify that all three points satisfy the plane equation:")
            
            for point_name, point in [("A", A), ("B", B), ("C", C)]:
                result = np.dot(n, point)
                check_mark = "✅" if np.isclose(result, k) else "❌"
                st.markdown(f"- **Point {point_name}**: $\\vec{{n}} \\cdot {point_name} = {result:.4f}$ {check_mark}")
            
            return {"normal": n, "constant": k, "points": [A, B, C]}
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    
    def point_to_plane_distance(self, point_input: str, normal_input: str, constant_input: str):
        """Calculate distance from a point to a plane"""
        st.subheader("Point-to-Plane Distance")
        
        try:
            # Parse inputs
            point = MathUtils.parse_vector(point_input)
            normal = MathUtils.parse_vector(normal_input)
            
            if point is None or normal is None:
                st.error("Invalid vector format. Please enter vectors as: x, y, z")
                return None
            
            if len(point) != 3 or len(normal) != 3:
                st.error("Point and normal vector must be 3-dimensional")
                return None
            
            try:
                k = float(constant_input)
            except ValueError:
                st.error("Invalid constant value. Please enter a number.")
                return None
            
            # Display inputs
            st.markdown("### Given Information")
            st.markdown(f"**Point D:** $${self._vector_to_latex(point)}$$")
            st.markdown(f"**Plane normal vector:** $$\\vec{{n}} = {self._vector_to_latex(normal)}$$")
            st.markdown(f"**Plane equation:** $${self._format_plane_equation(normal, k)}$$")
            
            # Calculate distance using formula: |n·p - k| / ||n||
            st.markdown("### Distance Calculation")
            st.markdown("**Formula:** $d = \\frac{|\\vec{n} \\cdot \\vec{p} - k|}{||\\vec{n}||}$")
            
            # Calculate numerator
            dot_product = np.dot(normal, point)
            numerator = abs(dot_product - k)
            
            st.markdown(f"**Numerator:** $|\\vec{{n}} \\cdot \\vec{{p}} - k| = |{self._vector_to_latex(normal)} \\cdot {self._vector_to_latex(point)} - {k}|$")
            st.markdown(f"$= |{dot_product:.4f} - {k}| = |{dot_product - k:.4f}| = {numerator:.4f}$")
            
            # Calculate denominator
            magnitude_n = np.linalg.norm(normal)
            
            st.markdown(f"**Denominator:** $||\\vec{{n}}|| = \\sqrt{{{self._format_norm_squared(normal)}}} = {magnitude_n:.4f}$")
            
            # Calculate final distance
            if magnitude_n == 0:
                st.error("Invalid normal vector: zero vector cannot be a normal vector")
                return None
            
            distance = numerator / magnitude_n
            
            st.markdown(f"**Distance:** $d = \\frac{{{numerator:.4f}}}{{{magnitude_n:.4f}}} = {distance:.4f}$")
            
            # Display result prominently
            st.success(f"🎯 **The distance from the point to the plane is: {distance:.4f}**")
            
            return distance
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    
    def line_plane_intersection(self, line_point_input: str, line_direction_input: str, 
                              plane_normal_input: str, plane_constant_input: str):
        """Calculate intersection point of a line and a plane"""
        st.subheader("Line-Plane Intersection")
        
        try:
            # Parse inputs
            line_point = MathUtils.parse_vector(line_point_input)
            line_direction = MathUtils.parse_vector(line_direction_input)
            plane_normal = MathUtils.parse_vector(plane_normal_input)
            
            if line_point is None or line_direction is None or plane_normal is None:
                st.error("Invalid vector format. Please enter vectors as: x, y, z")
                return None
            
            if len(line_point) != 3 or len(line_direction) != 3 or len(plane_normal) != 3:
                st.error("All vectors must be 3-dimensional")
                return None
            
            try:
                k = float(plane_constant_input)
            except ValueError:
                st.error("Invalid constant value. Please enter a number.")
                return None
            
            # Display inputs
            st.markdown("### Given Information")
            st.markdown(f"**Line point:** $$\\vec{{P_0}} = {self._vector_to_latex(line_point)}$$")
            st.markdown(f"**Line direction:** $$\\vec{{d}} = {self._vector_to_latex(line_direction)}$$")
            st.markdown(f"**Plane normal:** $$\\vec{{n}} = {self._vector_to_latex(plane_normal)}$$")
            st.markdown(f"**Plane equation:** $${self._format_plane_equation(plane_normal, k)}$$")
            
            # Line parametric equation: P(t) = P₀ + t*d
            st.markdown("---")
            st.markdown("### Method")
            st.markdown("**Line parametric equation:** $$\\vec{P}(t) = \\vec{P_0} + t \\cdot \\vec{d}$$")
            st.markdown("**Plane equation:** $$\\vec{n} \\cdot \\vec{r} = k$$")
            st.markdown("**Substitute line into plane:** $$\\vec{n} \\cdot (\\vec{P_0} + t \\cdot \\vec{d}) = k$$")
            
            # Calculate intersection
            st.markdown("---")
            st.markdown("### Calculation")
            
            # Check if line is parallel to plane
            n_dot_d = np.dot(plane_normal, line_direction)
            
            st.markdown(f"**Step 1:** Calculate $\\vec{{n}} \\cdot \\vec{{d}}$")
            st.markdown(f"$$\\vec{{n}} \\cdot \\vec{{d}} = {self._vector_to_latex(plane_normal)} \\cdot {self._vector_to_latex(line_direction)} = {n_dot_d:.4f}$$")
            
            if abs(n_dot_d) < 1e-10:
                # Line is parallel to plane
                n_dot_p0 = np.dot(plane_normal, line_point)
                if abs(n_dot_p0 - k) < 1e-10:
                    st.warning("🔄 **The line lies entirely in the plane!**")
                    st.markdown("Since $\\vec{n} \\cdot \\vec{d} = 0$ and the line point satisfies the plane equation, every point on the line is also on the plane.")
                else:
                    st.error("❌ **The line is parallel to the plane and does not intersect it!**")
                    st.markdown("Since $\\vec{n} \\cdot \\vec{d} = 0$ but the line point does not satisfy the plane equation, the line never touches the plane.")
                return None
            
            # Calculate parameter t
            n_dot_p0 = np.dot(plane_normal, line_point)
            t = (k - n_dot_p0) / n_dot_d
            
            st.markdown(f"**Step 2:** Calculate parameter $t$")
            st.markdown(f"$$\\vec{{n}} \\cdot \\vec{{P_0}} = {self._vector_to_latex(plane_normal)} \\cdot {self._vector_to_latex(line_point)} = {n_dot_p0:.4f}$$")
            st.markdown(f"$$t = \\frac{{k - \\vec{{n}} \\cdot \\vec{{P_0}}}}{{\\vec{{n}} \\cdot \\vec{{d}}}} = \\frac{{{k:.4f} - {n_dot_p0:.4f}}}{{{n_dot_d:.4f}}} = {t:.4f}$$")
            
            # Calculate intersection point
            intersection = line_point + t * line_direction
            
            st.markdown(f"**Step 3:** Calculate intersection point")
            st.markdown(f"$$\\vec{{P}}({t:.4f}) = {self._vector_to_latex(line_point)} + {t:.4f} \\cdot {self._vector_to_latex(line_direction)}$$")
            st.markdown(f"$$= {self._vector_to_latex(line_point)} + {self._vector_to_latex(t * line_direction)}$$")
            st.markdown(f"$$= {self._vector_to_latex(intersection)}$$")
            
            # Verification
            st.markdown("---")
            st.markdown("### Verification")
            verification = np.dot(plane_normal, intersection)
            check_mark = "✅" if abs(verification - k) < 1e-10 else "❌"
            st.markdown(f"**Check:** $\\vec{{n}} \\cdot \\vec{{P}} = {verification:.4f}$ vs $k = {k:.4f}$ {check_mark}")
            
            # Display result prominently  
            st.success(f"🎯 **Intersection Point:** $${self._vector_to_latex(intersection)}$$")
            
            return {"intersection": intersection, "parameter": t}
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    
    def plane_equation_converter(self, input_type: str, equation_input: str):
        """Convert between different plane equation formats"""
        st.subheader("Plane Equation Format Converter")
        
        try:
            if input_type == "Three Points":
                # Parse three points
                points_str = equation_input.strip().split(';')
                if len(points_str) != 3:
                    st.error("Please enter exactly three points separated by semicolons")
                    return None
                
                points = []
                for point_str in points_str:
                    point = MathUtils.parse_vector(point_str.strip())
                    if point is None or len(point) != 3:
                        st.error("Invalid point format. Use: x,y,z")
                        return None
                    points.append(point)
                
                A, B, C = points
                
                st.markdown("### Input: Three Points")
                st.markdown(f"**A:** $${self._vector_to_latex(A)}$$")
                st.markdown(f"**B:** $${self._vector_to_latex(B)}$$")
                st.markdown(f"**C:** $${self._vector_to_latex(C)}$$")
                
                # Calculate plane equation
                AB = B - A
                AC = C - A
                n = np.cross(AB, AC)
                
                if np.allclose(n, 0):
                    st.error("Points are collinear - they don't define a unique plane")
                    return None
                
                k = np.dot(n, A)
                
                st.markdown("---")
                st.markdown("### Calculated Normal Vector and Constant")
                st.markdown(f"**Normal vector:** $$\\vec{{n}} = {self._vector_to_latex(n)}$$")
                st.markdown(f"**Constant:** $$k = {k:.4f}$$")
                
            elif input_type == "Normal Vector + Point":
                # Parse normal vector and point
                parts = equation_input.strip().split(';')
                if len(parts) != 2:
                    st.error("Please enter normal vector and point separated by semicolon")
                    return None
                
                n = MathUtils.parse_vector(parts[0].strip())
                point = MathUtils.parse_vector(parts[1].strip())
                
                if n is None or point is None or len(n) != 3 or len(point) != 3:
                    st.error("Invalid format. Use: nx,ny,nz ; px,py,pz")
                    return None
                
                k = np.dot(n, point)
                
                st.markdown("### Input: Normal Vector + Point")
                st.markdown(f"**Normal vector:** $$\\vec{{n}} = {self._vector_to_latex(n)}$$")
                st.markdown(f"**Point on plane:** $$\\vec{{P}} = {self._vector_to_latex(point)}$$")
                st.markdown(f"**Calculated constant:** $$k = \\vec{{n}} \\cdot \\vec{{P}} = {k:.4f}$$")
                
            elif input_type == "Coordinate Form":
                # Parse coordinate form like "2x - 3y + z = 5"
                if '=' not in equation_input:
                    st.error("Please enter equation in form: ax + by + cz = d")
                    return None
                
                left, right = equation_input.split('=')
                k = float(right.strip())
                
                # Simple parsing for coordinate form (basic implementation)
                # This could be enhanced with more robust parsing
                left = left.replace(' ', '').replace('-', '+-')
                terms = [term for term in left.split('+') if term]
                
                n = np.zeros(3)
                for term in terms:
                    if 'x' in term:
                        coeff = term.replace('x', '')
                        n[0] = float(coeff) if coeff not in ['', '+', '-'] else (1 if coeff != '-' else -1)
                    elif 'y' in term:
                        coeff = term.replace('y', '')
                        n[1] = float(coeff) if coeff not in ['', '+', '-'] else (1 if coeff != '-' else -1)
                    elif 'z' in term:
                        coeff = term.replace('z', '')
                        n[2] = float(coeff) if coeff not in ['', '+', '-'] else (1 if coeff != '-' else -1)
                
                st.markdown("### Input: Coordinate Form")
                st.markdown(f"**Equation:** $${equation_input}$$")
                st.markdown(f"**Parsed normal vector:** $$\\vec{{n}} = {self._vector_to_latex(n)}$$")
                st.markdown(f"**Constant:** $$k = {k:.4f}$$")
            
            # Display all formats
            st.markdown("---")
            st.markdown("### All Equivalent Representations")
            
            # 1. Coordinate Form
            st.markdown("**1. Coordinate Form:**")
            self._display_plane_equation(n, k)
            
            # 2. Vector Form
            st.markdown("**2. Vector Form:**")
            st.markdown(f"$$\\vec{{n}} \\cdot \\vec{{r}} = k \\quad \\text{{where}} \\quad \\vec{{n}} = {self._vector_to_latex(n)} \\quad \\text{{and}} \\quad k = {k:.4f}$$")
            
            # 3. Parametric Form (two parameters)
            st.markdown("**3. Parametric Form:**")
            
            # Find two orthogonal vectors in the plane
            # Choose a vector not parallel to n
            if abs(n[0]) < 0.9:
                v1 = np.array([1, 0, 0])
            else:
                v1 = np.array([0, 1, 0])
            
            # Project v1 onto the plane: v1_proj = v1 - (v1·n/|n|²)n
            v1_proj = v1 - (np.dot(v1, n) / np.dot(n, n)) * n
            v1_proj = v1_proj / np.linalg.norm(v1_proj)  # normalize
            
            # Second vector: v2 = n × v1_proj
            v2_proj = np.cross(n, v1_proj)
            v2_proj = v2_proj / np.linalg.norm(v2_proj)  # normalize
            
            # Find a point on the plane (use the first point if available, or calculate one)
            if 'A' in locals():
                point_on_plane = A
            else:
                # Find a point on the plane by setting two coordinates and solving for the third
                if abs(n[2]) > 1e-10:
                    point_on_plane = np.array([0, 0, k/n[2]])
                elif abs(n[1]) > 1e-10:
                    point_on_plane = np.array([0, k/n[1], 0])
                else:
                    point_on_plane = np.array([k/n[0], 0, 0])
            
            st.markdown(f"$$\\vec{{r}}(s,t) = {self._vector_to_latex(point_on_plane)} + s \\cdot {self._vector_to_latex(v1_proj)} + t \\cdot {self._vector_to_latex(v2_proj)}$$")
            st.markdown("where $s, t \\in \\mathbb{R}$ are parameters")
            
            return {"normal": n, "constant": k, "point_on_plane": point_on_plane}
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    
    def _display_plane_equation(self, normal: np.ndarray, constant: float):
        """Display the plane equation in coordinate form"""
        n_x, n_y, n_z = normal
        
        # Create the equation string
        parts = []
        if n_x != 0:
            if n_x == 1:
                parts.append("x")
            elif n_x == -1:
                parts.append("-x")
            else:
                parts.append(f"{n_x:.4f}x".rstrip('0').rstrip('.'))
        
        if n_y != 0:
            if n_y > 0 and parts:
                if n_y == 1:
                    parts.append("+ y")
                else:
                    parts.append(f"+ {n_y:.4f}y".rstrip('0').rstrip('.'))
            else:
                if n_y == 1:
                    parts.append("y")
                elif n_y == -1:
                    parts.append("-y")
                else:
                    parts.append(f"{n_y:.4f}y".rstrip('0').rstrip('.'))
        
        if n_z != 0:
            if n_z > 0 and parts:
                if n_z == 1:
                    parts.append("+ z")
                else:
                    parts.append(f"+ {n_z:.4f}z".rstrip('0').rstrip('.'))
            else:
                if n_z == 1:
                    parts.append("z")
                elif n_z == -1:
                    parts.append("-z")
                else:
                    parts.append(f"{n_z:.4f}z".rstrip('0').rstrip('.'))
        
        equation = " ".join(parts).replace("+ -", "- ")
        constant_str = f"{constant:.4f}".rstrip('0').rstrip('.')
        
        st.markdown(f"**Coordinate Form:** $${equation} = {constant_str}$$")
        
        # Also show vector form
        st.markdown(f"**Vector Form:** $$\\vec{{n}} \\cdot \\vec{{r}} = k \\text{{ where }} \\vec{{n}} = {self._vector_to_latex(normal)} \\text{{ and }} k = {constant:.4f}$$")
    
    def _format_plane_equation(self, normal: np.ndarray, constant: float) -> str:
        """Format plane equation as string for display"""
        n_x, n_y, n_z = normal
        parts = []
        
        if n_x != 0:
            if n_x == 1:
                parts.append("x")
            elif n_x == -1:
                parts.append("-x")
            else:
                parts.append(f"{n_x:.2f}x")
        
        if n_y != 0:
            if n_y > 0 and parts:
                if n_y == 1:
                    parts.append("+ y")
                else:
                    parts.append(f"+ {n_y:.2f}y")
            else:
                if n_y == 1:
                    parts.append("y")
                elif n_y == -1:
                    parts.append("-y")
                else:
                    parts.append(f"{n_y:.2f}y")
        
        if n_z != 0:
            if n_z > 0 and parts:
                if n_z == 1:
                    parts.append("+ z")
                else:
                    parts.append(f"+ {n_z:.2f}z")
            else:
                if n_z == 1:
                    parts.append("z")
                elif n_z == -1:
                    parts.append("-z")
                else:
                    parts.append(f"{n_z:.2f}z")
        
        equation = " ".join(parts).replace("+ -", "- ")
        return f"{equation} = {constant:.2f}"
    
    def _format_norm_squared(self, vector: np.ndarray) -> str:
        """Format the squared norm expression for display"""
        parts = []
        for i, component in enumerate(vector):
            var = ["x", "y", "z"][i] if i < 3 else f"x_{i+1}"
            if component != 0:
                if component == 1:
                    parts.append(f"{var}^2")
                elif component == -1:
                    parts.append(f"(-{var})^2")
                else:
                    parts.append(f"({component:.4f})^2".rstrip('0').rstrip('.'))
        
        return " + ".join(parts)
    
    def _vector_to_latex(self, vector: np.ndarray) -> str:
        """Convert vector to LaTeX format"""
        if len(vector) == 0:
            return "\\begin{pmatrix}\\end{pmatrix}"
        
        # Format each component
        formatted_components = []
        for component in vector:
            if np.isclose(component, round(component), atol=1e-10):
                formatted_components.append(str(int(round(component))))
            else:
                formatted_components.append(f"{component:.4f}".rstrip('0').rstrip('.'))
        
        return "\\begin{pmatrix}" + " \\\\ ".join(formatted_components) + "\\end{pmatrix}"