#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Vector Operations Mixin
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .st_visualization_utils import display_vector_visualization
from .st_utils import StreamOutput

class VectorOperationsMixin:
    def normalize_vector(self, vector_input):
        """Normalize a vector to unit length."""
        # Create an Args object with the necessary attributes
        class Args:
            def __init__(self, vector):
                self.vector = vector
        
        args = Args(vector_input)
        
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.normalize_vector(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the vector from the input
        vector = self.framework.parse_vector(vector_input)
        
        # Calculate vector length
        length = np.linalg.norm(vector)
        
        # Create a formatted display of the calculations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Create a more structured output
            st.markdown("**Original vector:**")
            st.markdown(f"$v = {vector.tolist()}$")
            
            st.markdown("**Length calculation:**")
            length_formula = " + ".join([f"({v})^2" for v in vector])
            st.markdown(f"$|v| = \\sqrt{{{length_formula}}} = {length:.4f}$")
            
            st.markdown("**Normalized vector:**")
            st.markdown(f"$\\hat{{v}} = \\frac{{v}}{{|v|}} = \\frac{{{vector.tolist()}}}{{{length:.4f}}} = {result.tolist()}$")
            
            st.markdown("**Verification:**")
            normalized_length = np.linalg.norm(result)
            st.markdown(f"$|\\hat{{v}}| = {normalized_length:.4f} \\approx 1.0$ ✓")
        
        with col2:
            # Display a visualization of the original and normalized vectors
            st.markdown("### Visualization")
            display_vector_visualization([vector, result], names=["Original Vector", "Normalized Vector"])
        
        return result
    
    def vector_shadow(self, vector_a, vector_b):
        """Calculate the shadow (projection) of one vector onto another."""
        # Create an Args object with the necessary attributes
        class Args:
            def __init__(self, vector_a, vector_b):
                self.vector_a = vector_a
                self.vector_b = vector_b
        
        args = Args(vector_a, vector_b) # Reverted: framework should now calculate proj of b onto a
        
        # Get the result from the framework
        result = self.framework.vector_shadow(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the vectors from input
        a = self.framework.parse_vector(vector_a)
        b = self.framework.parse_vector(vector_b)
        vector_proj = result["vector_proj"]
        scalar_proj = result["scalar_proj"]
        
        # Calculate dot products and normalization for the formulas
        dot_product = np.dot(a, b)
        norm_a_squared = np.dot(a, a)
        norm_a = np.linalg.norm(a)
        
        # Create a formatted display of the calculations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Create a more structured output
            st.markdown("**Input vectors:**")
            st.markdown(f"$\\vec{{a}} = {a.tolist()}$") 
            st.markdown(f"$\\vec{{b}} = {b.tolist()}$")
            
            st.markdown("**Dot product calculation:**")
            dot_formula = " + ".join([f"({a[i]} \\cdot {b[i]})" for i in range(len(a))])
            st.markdown(f"$\\vec{{a}} \\cdot \\vec{{b}} = {dot_formula} = {dot_product:.4f}$")
            
            st.markdown("**Magnitude calculations:**")
            st.markdown(f"$|\\vec{{a}}|^2 = {' + '.join([f'({v})^2' for v in a])} = {norm_a_squared:.4f}$")
            st.markdown(f"$\\|\\vec{{a}}\\| = \\sqrt{{{norm_a_squared:.4f}}} = {norm_a:.4f}$") # Changed to ||a||
            
            st.markdown("**Scalar projection calculation:**")
            st.markdown(f"$\\text{{proj}}_{{\\text{{scalar}}}} = \\frac{{\\vec{{a}} \\cdot \\vec{{b}}}}{{\\|\\vec{{a}}\\|}} = \\frac{{{dot_product:.4f}}}{{{norm_a:.4f}}} = {scalar_proj:.4f}$") # Changed to ||a|| in denominator
            
            st.markdown("**Vector projection calculation:**")
            # Format list elements for direct LaTeX rendering
            latex_formatted_a_list = f"[{', '.join(f'{x:.4g}' for x in a.tolist())}]"
            latex_formatted_vector_proj_list = f"[{', '.join(f'{x:.4g}' for x in vector_proj.tolist())}]"
            # Use st.latex() for robust rendering. Changed label from \text{vector_proj} to \mathbf{P}_{	ext{proj}}
            vector_proj_latex_str = f"\\mathbf{{P}}_{{\\text{{proj}}}} = \\frac{{\\vec{{a}} \\cdot \\vec{{b}}}}{{\\vec{{a}} \\cdot \\vec{{a}}}} \\vec{{a}} = \\frac{{{dot_product:.4f}}}{{{norm_a_squared:.4f}}} \\cdot {latex_formatted_a_list} = {latex_formatted_vector_proj_list}"
            st.latex(vector_proj_latex_str)
            
            # Add explanation of what the projection means
            st.markdown("**Interpretation:**")
            st.markdown( # Corrected multi-line string
            """
            The projection of vector b onto vector a represents how much of vector b points in the direction of a.
            - **Scalar projection**: Length of the shadow of vector b when cast onto the line along vector a
            - **Vector projection**: The resulting vector along the direction of a
            """)
        
        with col2:
            st.markdown("### Visualization")
            
            # Define the construction line for the shadow
            # It goes from the tip of b to the tip of vector_proj
            # The points are b (tip of vector b) and vector_proj (tip of projection vector, which is itself a vector from origin)
            # So the line segment is from point b to point vector_proj.
            shadow_line_style = dict(dash='dash', color='rgba(220, 220, 220, 0.6)', width=2) # Lighter color for dash
            construction_lines = [
                (b, vector_proj, "Projection Line", shadow_line_style)
            ]
            
            display_vector_visualization(
                [a, b, vector_proj], 
                names=["Vector a", "Vector b", "Projection of b onto a"],
                construction_lines=construction_lines
            )
            
            # Add projection magnitude display
            projection_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 16px; text-align: center;">
                    <strong>Scalar Projection:</strong> {scalar_proj:.4f}<br>
                    <strong>Vector Projection:</strong> {vector_proj.tolist()}
                </p>
            </div>
            """
            st.markdown(projection_box, unsafe_allow_html=True)
        
        return result
    
    def vector_angle(self, vector_a, vector_b):
        """Calculate the angle between two vectors."""
        # Create an Args object with the necessary attributes
        class Args:
            def __init__(self, vector_a, vector_b):
                self.vector_a = vector_a
                self.vector_b = vector_b
        
        args = Args(vector_a, vector_b)
        
        # Get the result from the framework
        result = self.framework.vector_angle(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the vectors from input
        a = self.framework.parse_vector(vector_a)
        b = self.framework.parse_vector(vector_b)
        
        # Calculate values for displaying in the formula
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_angle = dot_product / (norm_a * norm_b)
        angle_rad = result["radians"]
        angle_deg = result["degrees"]
        
        # Create a formatted display of the calculations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Create a more structured output
            st.markdown("**Input vectors:**")
            st.markdown(f"$\\vec{{a}} = {a.tolist()}$")
            st.markdown(f"$\\vec{{b}} = {b.tolist()}$")
            
            st.markdown("**Dot product calculation:**")
            dot_formula = " + ".join([f"({a[i]} \\cdot {b[i]})" for i in range(len(a))])
            st.markdown(f"$\\vec{{a}} \\cdot \\vec{{b}} = {dot_formula} = {dot_product:.4f}$")
            
            st.markdown("**Vector magnitudes:**")
            st.markdown(f"$|\\vec{{a}}| = \\sqrt{{{' + '.join([f'({v})^2' for v in a])}}} = {norm_a:.4f}$")
            st.markdown(f"$|\\vec{{b}}| = \\sqrt{{{' + '.join([f'({v})^2' for v in b])}}} = {norm_b:.4f}$")
            
            st.markdown("**Angle calculation:**")
            st.markdown(f"$\\cos(\\theta) = \\frac{{\\vec{{a}} \\cdot \\vec{{b}}}}{{|\\vec{{a}}| \\cdot |\\vec{{b}}|}} = \\frac{{{dot_product:.4f}}}{{{norm_a:.4f} \\cdot {norm_b:.4f}}} = {cos_angle:.4f}$")
            
            st.markdown("**Final angle:**")
            st.markdown(f"$\\theta = \\arccos({cos_angle:.4f}) = {angle_rad:.4f}$ radians")
            st.markdown(f"$\\theta = {angle_rad:.4f} \\cdot \\frac{{180^\\circ}}{{\\pi}} = {angle_deg:.4f}^\\circ$")
        
        with col2:
            st.markdown("### Visualization")
            display_vector_visualization([a, b], names=["Vector a", "Vector b"])
        
            # Add mathematical representation of the angle
            angle_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 18px; text-align: center;">
                    Angle between vectors: {angle_deg:.2f}° ({angle_rad:.4f} radians)
                </p>
            </div>
            """
            st.markdown(angle_box, unsafe_allow_html=True)
        
        return result
        
    def cross_product(self, args):
        """Calculate the cross product of two vectors."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.cross_product(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse vectors from args (assuming args.vector_a and args.vector_b are string inputs)
        a = self.framework.parse_vector(args.vector_a)
        b = self.framework.parse_vector(args.vector_b)
        
        # The result from framework.cross_product is a numpy array
        # Create a dictionary with the cross product and its magnitude (for area calculations)
        cross = result  # The actual cross product
        area = np.linalg.norm(cross) / 2  # Half the magnitude is the triangle area
        
        # Validate vector dimensions
        if len(a) != 3 or len(b) != 3:
            st.error("Cross product is defined for 3D vectors only.")
            return result
        
        # Create a formatted display of the calculations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            st.markdown("**Input vectors:**")
            st.markdown(f"$\\vec{{a}} = {a.tolist()}$")
            st.markdown(f"$\\vec{{b}} = {b.tolist()}$")
            
            st.markdown("**Cross product formula (determinant method):**")
            st.latex(r"""
            \vec{a} \times \vec{b} = 
            \begin{vmatrix}
            \mathbf{i} & \mathbf{j} & \mathbf{k} \\
            a_1 & a_2 & a_3 \\
            b_1 & b_2 & b_3 
            \end{vmatrix}
            = (a_2 b_3 - a_3 b_2)\mathbf{i} - (a_1 b_3 - a_3 b_1)\mathbf{j} + (a_1 b_2 - a_2 b_1)\mathbf{k}
            """)
            
            st.markdown("**Calculation:**")
            st.markdown(f"$x = ({a[1]} \\cdot {b[2]}) - ({a[2]} \\cdot {b[1]}) = {cross[0]}$")
            st.markdown(f"$y = - (({a[0]} \\cdot {b[2]}) - ({a[2]} \\cdot {b[0]})) = {cross[1]}$")
            st.markdown(f"$z = ({a[0]} \\cdot {b[1]}) - ({a[1]} \\cdot {b[0]}) = {cross[2]}$")
            
            st.markdown("**Resulting vector:**")
            st.markdown(f"$\\vec{{a}} \\times \\vec{{b}} = {cross.tolist()}$")
            
            st.markdown("**Area of parallelogram:**")
            st.markdown(f"Area = $|\\vec{{a}} \\times \\vec{{b}}| = {area:.4f}$")
        
        with col2:
            st.markdown("### Visualization")
            
            # Define construction lines for the parallelogram
            # Origin O, vector a from O, vector b from O, vector a+b from O
            # Parallelogram vertices: O, a, b, a+b
            # Line 1: from a to a+b (parallel to b)
            # Line 2: from b to a+b (parallel to a)
            origin = np.array([0,0,0])
            a_plus_b = a + b
            
            line_style = dict(dash='dash', color='rgba(200, 200, 200, 0.7)', width=2)
            construction_lines = [
                (a, a_plus_b, "Side (a to a+b)", line_style),
                (b, a_plus_b, "Side (b to a+b)", line_style)
            ]

            if len(cross) == 3 and (np.any(cross) or np.any(a) or np.any(b)): # Only visualize if cross product is 3D and non-zero, or a/b non-zero
                display_vector_visualization(
                    [a, b, cross],
                    names=["Vector a", "Vector b", "Cross Product (a × b)"],
                    construction_lines=construction_lines
                )
            else:
                # For 2D vectors (z=0), create a simpler visualization
                display_vector_visualization([a, b, cross], names=["Vector a", "Vector b", "Cross Product (a × b)"])
            
            # Add result info box
            result_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 16px; text-align: center;">
                    <strong>Cross Product:</strong> {cross.tolist()}<br>
                    <strong>Area of Parallelogram:</strong> {area:.4f}
                </p>
            </div>
            """
            st.markdown(result_box, unsafe_allow_html=True)
        
        return result

    def triangle_area(self, args):
        """Calculate the area of a triangle defined by three points."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.triangle_area(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # The args object has point_a, point_b, and point_c attributes
        area = result["area"] if isinstance(result, dict) else 0
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Parse the points from args
            p1 = self.framework.parse_vector(args.point_a)
            p2 = self.framework.parse_vector(args.point_b)
            p3 = self.framework.parse_vector(args.point_c)
            
            st.markdown("**Input points:**")
            st.markdown(f"$P_1 = {p1.tolist()}$")
            st.markdown(f"$P_2 = {p2.tolist()}$")
            st.markdown(f"$P_3 = {p3.tolist()}$")
            
            # Form vectors from points
            vec1 = p2 - p1
            vec2 = p3 - p1
            st.markdown("**Forming vectors from points:**")
            st.markdown(f"$\\vec{{v_1}} = P_2 - P_1 = {vec1.tolist()}$")
            st.markdown(f"$\\vec{{v_2}} = P_3 - P_1 = {vec2.tolist()}$")
            
            st.markdown("**Area calculation (using cross product for 3D, or determinant for 2D):**")
            if len(p1) == 3: # 3D case
                cross_prod = np.cross(vec1, vec2)
                st.markdown(f"$\\text{{Area}} = 0.5 \\cdot |\\vec{{v_1}} \\times \\vec{{v_2}}| = 0.5 \\cdot |{cross_prod.tolist()}| = 0.5 \\cdot {np.linalg.norm(cross_prod):.4f} = {area:.4f}$")
            elif len(p1) == 2: # 2D case
                # Using Shoelace formula / determinant method for 2D triangle area
                # Area = 0.5 |(x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2))|
                # Or using the determinant of vectors formed from edges: 0.5 * |x1*y2 - x2*y1|
                determinant = vec1[0]*vec2[1] - vec1[1]*vec2[0]
                st.markdown(f"$\\text{{Area}} = 0.5 \\cdot |(v_{{1x}}v_{{2y}} - v_{{1y}}v_{{2x}})| = 0.5 \\cdot |({vec1[0]} \\cdot {vec2[1]}) - ({vec1[1]} \\cdot {vec2[0]})| = 0.5 \\cdot |{determinant}| = {area:.4f}$")
            else:
                st.warning("Visualization and detailed steps are best for 2D or 3D points.")
        
        with col2:
            st.markdown("### Visualization")
            origin = np.zeros(len(p1))
            fig_vectors = []
            fig_names = []
            construction_lines = []
            line_style = dict(color='rgba(200, 200, 200, 0.9)', width=2)
            
            p1_arr, p2_arr, p3_arr = np.array(p1), np.array(p2), np.array(p3)
            # Plot points if they are not all at the origin (or very close)
            if not (np.allclose(p1_arr, origin) and np.allclose(p2_arr, origin) and np.allclose(p3_arr, origin)):
                # To visualize points themselves, we can create tiny vectors from origin to each point, or use scatter plot markers directly.
                # Let's ensure display_vector_visualization can handle points if given as vectors.
                # For now, let's represent points as vectors from origin for simplicity with current viz tool
                # Or, it might be better to plot the triangle sides.
                # fig_vectors.extend([p1_arr, p2_arr, p3_arr])
                # fig_names.extend(["P1", "P2", "P3"])
                pass # Points themselves are implicitly shown by triangle construction

            # Triangle sides
            # Side P1-P2
            construction_lines.append((p1_arr, p2_arr, "Side P1-P2", line_style))
            # Side P2-P3
            construction_lines.append((p2_arr, p3_arr, "Side P2-P3", line_style))
            # Side P3-P1
            construction_lines.append((p3_arr, p1_arr, "Side P3-P1", line_style))
            
            # We can also show the vectors used for area calc if desired (v1, v2 from P1)
            # fig_vectors.extend([p2_arr - p1_arr, p3_arr - p1_arr])
            # fig_names.extend(["Vector P1->P2", "Vector P1->P3"]) 
            # For clarity, let's pass the primary vectors as the ones used in calculation.
            fig_vectors = [p2_arr - p1_arr, p3_arr - p1_arr]
            fig_names = ["Vector P1P2", "Vector P1P3"]


            if fig_vectors: # Only call if there are vectors to plot
                 # Check dimension for visualization
                dim = len(fig_vectors[0]) if fig_vectors else 0
                if construction_lines and not fig_vectors:
                    dim = len(construction_lines[0][0]) # Get dim from first point of first construction line
                
                if dim == 2 or dim == 3:
                    display_vector_visualization(fig_vectors, names=fig_names, construction_lines=construction_lines)
                else:
                    st.info(f"Direct visualization is for 2D/3D. Triangle involves {dim}D inputs.")
            else:
                 st.info("No primary vectors to display for this triangle input.")

            # Add result info box
            result_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 18px; text-align: center;">
                    Area of Triangle: {area:.4f}
                </p>
            </div>
            """
            st.markdown(result_box, unsafe_allow_html=True)
            
        return result

    def point_line_distance(self, args):
        """Calculate the distance from a point to a line (defined by a point and direction vector)."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.point_line_distance(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        distance = result["distance"] if isinstance(result, dict) else 0
        
        col1, col2 = st.columns([1,1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Parse common inputs
            point_p = self.framework.parse_vector(args.point_p)
            st.markdown(f"**Point P:** ${point_p.tolist()}$")
            
            # Parse the needed vectors
            point_a = self.framework.parse_vector(args.point_a)
            direction = self.framework.parse_vector(args.direction)
            point_b = self.framework.parse_vector(args.point_b)
            
            # The line is defined by point A and direction vector
            st.markdown(f"**Line defined by point A and direction vector d:**")
            st.markdown(f"$A = {point_a.tolist()}$")
            st.markdown(f"$\\vec{{d}} = {direction.tolist()}$")
            
            # Calculate vector from point on line to point B
            vec_ap = point_b - point_a
            direction_vec_line = direction
            point_a_on_line = point_a
            
            st.markdown(f"**Vector AB (from A to B):** $\\vec{{AB}} = B - A = {vec_ap.tolist()}$")
            st.markdown(f"**Direction vector of the line:** $\\vec{{d}} = {direction_vec_line.tolist()}$")
            
            st.markdown("**Distance formula (using cross product for 3D, projection for 2D/general):**")
            # Universal formula: Distance = || (P-A) - proj_d(P-A) ||
            # Or for 3D: || AP x d || / ||d||
            # For 2D: | (P-A) . d_perp | / ||d|| where d_perp is (-dy, dx)
            
            if len(point_p) == 3 and len(direction_vec_line) == 3:
                cross_prod = np.cross(vec_ap, direction_vec_line)
                norm_d = np.linalg.norm(direction_vec_line)
                st.markdown(f"$\\text{{Distance}} = \\frac{{|\\vec{{AP}} \\times \\vec{{d}}|}}{{|\\vec{{d}}|}} = \\frac{{|{cross_prod.tolist()}|}}{{{norm_d:.4f}}} = \\frac{{{np.linalg.norm(cross_prod):.4f}}}{{{norm_d:.4f}}} = {distance:.4f}$")
            elif len(point_p) == 2 and len(direction_vec_line) == 2:
                # Using 2D specific formula: |(x_p-x_a)d_y - (y_p-y_a)d_x| / sqrt(d_x^2 + d_y^2)
                # This is equivalent to | vec_ap_x * d_y - vec_ap_y * d_x | / ||d||
                # Which is the magnitude of the 2D cross product (scalar) / magnitude of d
                numerator = abs(vec_ap[0]*direction_vec_line[1] - vec_ap[1]*direction_vec_line[0])
                denominator = np.linalg.norm(direction_vec_line)
                st.markdown(f"$\\text{{Distance}} = \\frac{{|(P_x-A_x)d_y - (P_y-A_y)d_x|}}{{\\sqrt{{d_x^2 + d_y^2}}}} = \\frac{{|({vec_ap[0]})({direction_vec_line[1]}) - ({vec_ap[1]})({direction_vec_line[0]})|}}{{{denominator:.4f}}} = \\frac{{{numerator:.4f}}}{{{denominator:.4f}}} = {distance:.4f}$")
            else:
                # General case using projection, works for any dimension
                # Distance = || vec_ap - proj_d(vec_ap) ||
                proj_d_ap = np.dot(vec_ap, direction_vec_line) / np.dot(direction_vec_line, direction_vec_line) * direction_vec_line
                vec_perp = vec_ap - proj_d_ap # Vector from line to point P, perpendicular to line
                # The distance is the magnitude of this perpendicular vector.
                st.markdown(f"Projection of AP onto d: $proj_d(AP) = \\frac{{AP \cdot d}}{{d \cdot d}} d = {proj_d_ap.tolist()}$")
                st.markdown(f"Perpendicular vector (from line to P): $AP - proj_d(AP) = {vec_perp.tolist()}$")
                st.markdown(f"Distance = $||AP - proj_d(AP)|| = {np.linalg.norm(vec_perp):.4f}$")
                if abs(np.linalg.norm(vec_perp) - distance) > 1e-3: # Check if matches CLI result
                    st.warning(f"Calculated distance {np.linalg.norm(vec_perp):.4f} differs slightly from CLI result {distance:.4f}. Using CLI result.")

        with col2:
            st.markdown("### Visualization")
            
            dim = len(point_p)
            construction_lines = []
            primary_vectors = [] # Vectors like AP, d
            primary_names = []
            
            # Line visualization: show a segment of the line
            # Create a segment of the line using the point and direction
            line_p1 = self.framework.parse_vector(args.point_a)
            # Make line_p2 by extending along direction_vec. For viz, scale d if it's too small/large
            # Let's just use A and A+d for the segment.
            line_p2 = line_p1 + direction_vec_line 
            
            line_style = dict(color='blue', width=2)
            construction_lines.append((line_p1, line_p2, "Line Segment", line_style))

            # Point P: can be visualized as a vector from origin, or just a marker if viz supports it
            # For now, let display_vector_visualization handle it if P is added to primary_vectors.
            # However, it's a point, not a vector. Let's find the closest point on the line to P.
            
            # Calculate closest point on line (Q) to P
            # Q = A + proj_d(AP)  where d is direction_vec_line, A is point_a_on_line
            if point_a_on_line is not None and direction_vec_line is not None and vec_ap is not None:
                if np.dot(direction_vec_line, direction_vec_line) == 0:
                    # Direction vector is zero, line is just a point A.
                    # Distance is ||P-A||
                    q_closest_point = point_a_on_line
                else:
                    proj_vec_ap_onto_d = (np.dot(vec_ap, direction_vec_line) / np.dot(direction_vec_line, direction_vec_line)) * direction_vec_line
                    q_closest_point = point_a_on_line + proj_vec_ap_onto_d
            
                # Line segment from P to Q (the distance itself)
                distance_line_style = dict(color='red', dash='dash', width=2)
                construction_lines.append((point_p, q_closest_point, "Distance", distance_line_style))
                
                # Show point P and Q as markers / small vectors if possible
                # Adding Q as a vector from origin for visualization
                # primary_vectors.append(point_p) # Visualizing P as vector from origin
                # primary_names.append("Point P")
                # primary_vectors.append(q_closest_point) # Visualizing Q as vector from origin
                # primary_names.append("Closest Point Q")
                # For better viz, use P, Q, and A as points in construction if possible, or as vectors from origin
                # Let's try plotting key vectors: AP, d, and the perpendicular segment PQ
                primary_vectors = [vec_ap, direction_vec_line, (point_p - q_closest_point)]
                primary_names = ["Vector AP", "Line Direction d", "Perpendicular PQ"]
                
                # To make the visualization clearer, we can also add the points A, P, Q explicitly.
                # The current display_vector_visualization is primarily for vectors from origin.
                # For points, it's better to have specific markers.
                # Let's adjust origin for the display if it helps center the view around relevant points.
                # Centroid = (point_p + point_a_on_line + q_closest_point) / 3
                # display_origin = centroid
                display_origin = np.zeros(dim) # Keep origin for now

            if dim == 2 or dim == 3:
                # Use a dummy non-empty primary_vectors if only construction_lines are present for now to trigger plot
                # if not primary_vectors and construction_lines:
                #    primary_vectors = [np.zeros(dim)] # Add a dummy zero vector if no primary vectors
                #    primary_names = ["Origin"] # This is a workaround for current display_vector_visualization
                if not primary_vectors and construction_lines:
                     # If no primary vectors, pass an empty list and let visualization handle it or focus on lines
                     display_vector_visualization([], names=[], origin=display_origin, construction_lines=construction_lines)
                else:
                    display_vector_visualization(primary_vectors, names=primary_names, origin=display_origin, construction_lines=construction_lines)

            else:
                st.info(f"Direct visualization is for 2D/3D. Current dimension is {dim}.")

            # Add result info box
            result_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 18px; text-align: center;">
                    Distance from point to line: {distance:.4f}
                </p>
            </div>
            """
            st.markdown(result_box, unsafe_allow_html=True)
            
        return result

    def check_collinear(self, args):
        """Check if three points are collinear."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.check_collinearity(args) # Note: CLI function is check_collinearity
        
        # Display the result and steps
        st.subheader("Result")
        
        p1 = self.framework.parse_vector(args.point1)
        p2 = self.framework.parse_vector(args.point2)
        p3 = self.framework.parse_vector(args.point3)
        
        result_is_collinear = result["collinear"]
        method_used = result.get("method", "area") # Default to area if method not specified
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            st.markdown("**Input points:**")
            st.markdown(f"$P_1 = {p1.tolist()}$")
            st.markdown(f"$P_2 = {p2.tolist()}$")
            st.markdown(f"$P_3 = {p3.tolist()}$")
            
            # Form vectors P1P2 and P1P3
            vec_p1p2 = p2 - p1
            vec_p1p3 = p3 - p1
            
            st.markdown("**Forming vectors:**")
            st.markdown(f"$\\vec{{P_1P_2}} = P_2 - P_1 = {vec_p1p2.tolist()}$")
            st.markdown(f"$\\vec{{P_1P_3}} = P_3 - P_1 = {vec_p1p3.tolist()}$")
            
            if method_used == "area" or (len(p1) != 3 and method_used == "cross_product"):
                st.markdown("**Method: Area of Triangle**")
                st.markdown("Points are collinear if the area of the triangle formed by them is zero.")
                if len(p1) == 2:
                    # Area = 0.5 * |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|
                    # Or using vectors: 0.5 * |(P2x-P1x)(P3y-P1y) - (P2y-P1y)(P3x-P1x)|
                    area_val = 0.5 * abs(vec_p1p2[0]*vec_p1p3[1] - vec_p1p2[1]*vec_p1p3[0])
                    st.markdown(f"Area = $0.5 \\cdot |(x_2-x_1)(y_3-y_1) - (y_2-y_1)(x_3-x_1)| = {area_val:.4f}$")
                elif len(p1) == 3:
                    # Use cross product magnitude for area in 3D
                    cross_prod = np.cross(vec_p1p2, vec_p1p3)
                    area_val = 0.5 * np.linalg.norm(cross_prod)
                    st.markdown(f"Area = $0.5 \\cdot ||\\vec{{P_1P_2}} \\times \\vec{{P_1P_3}}|| = 0.5 \\cdot ||{cross_prod.tolist()}|| = {area_val:.4f}$")
                else:
                    area_val = result.get("area", 0) # Get area from CLI if not 2D/3D for formula display
                    st.markdown(f"Area of triangle formed by points: {area_val:.4f}")
                
                st.markdown(f"Since the area is {('approximately zero' if result_is_collinear else 'not zero')}, the points are {('collinear' if result_is_collinear else 'not collinear')}.")

            elif method_used == "cross_product" and len(p1) == 3:
                st.markdown("**Method: Cross Product (for 3D points)**")
                st.markdown("Points are collinear if the cross product of $\\vec{{P_1P_2}}$ and $\\vec{{P_1P_3}}$ is the zero vector.")
                cross_prod = np.cross(vec_p1p2, vec_p1p3)
                st.markdown(f"$\\vec{{P_1P_2}} \\times \\vec{{P_1P_3}} = {cross_prod.tolist()}$")
                st.markdown(f"Since the cross product is {('the zero vector (approximately)' if result_is_collinear else 'not the zero vector')}, the points are {('collinear' if result_is_collinear else 'not collinear')}.")
            
            elif method_used == "slope":
                st.markdown("**Method: Slopes (for 2D points)**")
                st.markdown("Points are collinear if the slope of P1P2 is equal to the slope of P1P3 (or P2P3). Handle vertical lines.")
                # This part would need slope calculation logic, which can be complex with vertical lines/division by zero.
                # The CLI handles this, so we rely on its result. For display, we can just state the method.
                st.markdown(f"Comparing slopes. Based on calculations, the points are {('collinear' if result_is_collinear else 'not collinear')}.")
            
            else: # Fallback or other methods from CLI
                 st.markdown(f"**Method: {method_used.replace('_',' ').title()}**")
                 st.markdown(f"Based on the calculation, the points are {('collinear' if result_is_collinear else 'not collinear')}.")

        with col2:
            st.markdown("### Visualization")
            
            dim = len(p1)
            # Vectors from p1 to p2, and p1 to p3. If collinear, they point in same/opposite direction.
            # Or, more simply, plot the three points and a line connecting them if collinear.
            
            construction_lines = []
            line_style = dict(color='cyan', width=2)
            
            # Always draw line segments P1-P2 and P2-P3
            construction_lines.append((p1, p2, "Segment P1-P2", line_style))
            construction_lines.append((p2, p3, "Segment P2-P3", line_style))
            
            # If collinear, we can also draw the full line P1-P3 for emphasis
            if result_is_collinear:
                construction_lines.append((p1, p3, "Line P1-P3 (collinear)", dict(color='lime', width=2, dash='dot')))

            # For display_vector_visualization, it expects vectors from origin or segments. 
            # We can pass the points as vectors from origin to mark their positions.
            # Or, rely on construction_lines to draw the shape formed by points.

            # For clarity, let's pass p1, p2, p3 as vectors from origin if they are not all zero
            origin = np.zeros(dim)
            points_as_vectors = []
            points_names = []
            if not np.allclose(p1, origin):
                points_as_vectors.append(p1)
                points_names.append("P1")
            if not np.allclose(p2, origin):
                 points_as_vectors.append(p2)
                 points_names.append("P2")
            if not np.allclose(p3, origin):
                 points_as_vectors.append(p3)
                 points_names.append("P3")
            
            # If all points are at origin, add a dummy origin vector to enable plot if only construction lines present
            if not points_as_vectors and construction_lines and np.allclose(p1, origin) and np.allclose(p2, origin) and np.allclose(p3, origin):
                points_as_vectors.append(origin)
                points_names.append("Origin (all points here)")

            if dim == 2 or dim == 3:
                display_vector_visualization(points_as_vectors, names=points_names, construction_lines=construction_lines)
            else:
                st.info(f"Direct visualization is for 2D/3D. Points are in {dim}D space.")

            # Add result info box
            collinearity_status = "Collinear" if result_is_collinear else "Not Collinear"
            result_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 18px; text-align: center;">
                    The points are: <strong>{collinearity_status}</strong>
                </p>
            </div>
            """
            st.markdown(result_box, unsafe_allow_html=True)
        
        return result_is_collinear 