#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Vector Operations Mixin
"""

import streamlit as st
import numpy as np
import pandas as pd # Keep for potential future use with vector display
import plotly.express as px # Keep for potential future use
import plotly.graph_objects as go # Keep for potential future use
from .st_visualization_utils import display_vector_visualization
from .st_utils import StreamOutput

class VectorOperationsMixin:
    """
    Mixin class that provides vector operation methods for the Streamlit interface.
    This class contains methods that display and process vector operations with visualization.
    """
    def normalize_vector(self, vector_input):
        """Normalize a vector to have unit length."""
        # Capture print output from CLI function
        with StreamOutput() as output:
            # Define Args for CLI compatibility
            class Args:
                def __init__(self, vector):
                    self.vector = vector
            
            args = Args(vector_input)
            result = self.framework.normalize_vector(args)
        
        # Print the output from CLI to for debugging
        # st.text(output.get_output())
        
        # Display the results
        st.subheader("Vector Normalization")
        
        # Parse original vector for display/visualization
        vector = self.framework.parse_vector(vector_input)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Display the original vector
            st.markdown("**Original Vector:**")
            st.markdown(f"$\\vec{{v}} = {vector.tolist()}$")
            
            # Calculate and display magnitude
            magnitude = np.linalg.norm(vector)
            st.markdown(f"**Magnitude (Length):**")
            st.markdown(f"$|\\vec{{v}}| = \\sqrt{{{'+'.join([f'{x}^2' for x in vector.tolist()])}}} = {magnitude:.4f}$")
            
            # Calculate and display normalized vector
            if magnitude > 0:
                normalized = vector / magnitude
                st.markdown(f"**Normalized Vector:**")
                st.markdown(f"$\\hat{{v}} = \\frac{{\\vec{{v}}}}{{|\\vec{{v}}|}} = \\frac{{{vector.tolist()}}}{{{magnitude:.4f}}} = {normalized.tolist()}$")
                
                # Verify unit length
                new_magnitude = np.linalg.norm(normalized)
                st.markdown(f"**Verification (magnitude of normalized vector):**")
                st.markdown(f"$|\\hat{{v}}| = {new_magnitude:.4f} \\approx 1.0$")
            else:
                st.warning("Cannot normalize the zero vector (magnitude = 0)")
                normalized = vector
        
        with col2:
            st.markdown("### Visualization")
            
            # Use display_vector_visualization function to show both original and normalized vectors
            if len(vector) <= 3:  # Only visualize for 2D and 3D vectors
                display_vector_visualization(
                    [vector, normalized] if magnitude > 0 else [vector],
                    names=["Original Vector", "Normalized Vector"] if magnitude > 0 else ["Zero Vector"]
                )
            else:
                st.markdown(f"**Original Vector:** {vector.tolist()}")
                if magnitude > 0:
                    st.markdown(f"**Normalized Vector:** {normalized.tolist()}")
                else:
                    st.warning("Cannot normalize the zero vector (magnitude = 0)")
        
        return result
        
    def vector_shadow(self, vector_a_input, vector_b_input):
        """Calculate the shadow (projection) of vector b onto vector a."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            # Define Args for CLI compatibility
            class Args:
                def __init__(self, vector_a, vector_b):
                    self.vector_a = vector_a
                    self.vector_b = vector_b
            
            args = Args(vector_a_input, vector_b_input)
            result = self.framework.vector_shadow(args)
        
        cli_output = output.get_output()
        
        # Display the results
        st.subheader("Vector Projection (Shadow)")
        
        # Parse vectors for display/visualization
        vector_a = self.framework.parse_vector(vector_a_input)
        vector_b = self.framework.parse_vector(vector_b_input)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Display the original vectors
            st.markdown("**Original Vectors:**")
            st.markdown(f"$\\vec{{a}} = {vector_a.tolist()}$")
            st.markdown(f"$\\vec{{b}} = {vector_b.tolist()}$")
            
            # Calculate dot product
            dot_product = np.dot(vector_a, vector_b)
            
            # Calculate magnitudes
            magnitude_a = np.linalg.norm(vector_a)
            
            # Calculate scalar projection
            scalar_proj = dot_product / magnitude_a
            
            st.markdown("**Scalar Projection (Length of Shadow):**")
            st.markdown(f"$\\text{{proj}}_{{\\vec{{a}}}}\\vec{{b}} = \\frac{{\\vec{{a}} \\cdot \\vec{{b}}}}{{|\\vec{{a}}|}} = \\frac{{{dot_product}}}{{{magnitude_a:.4f}}} = {scalar_proj:.4f}$")
            
            # Calculate vector projection
            if magnitude_a > 0:
                vector_proj = (scalar_proj / magnitude_a) * vector_a
                st.markdown("**Vector Projection (Shadow Vector):**")
                st.markdown(f"$\\text{{proj}}_{{\\vec{{a}}}}\\vec{{b}} = \\frac{{\\vec{{a}} \\cdot \\vec{{b}}}}{{|\\vec{{a}}|^2}} \\vec{{a}} = {vector_proj.tolist()}$")
                
                # Determine angle type
                if dot_product > 0:
                    st.markdown("**Angle between vectors is acute (0° < φ < 90°)**")
                elif dot_product < 0:
                    st.markdown("**Angle between vectors is obtuse (90° < φ < 180°)**")
                else:
                    st.markdown("**Vectors are perpendicular (φ = 90°)**")
            else:
                st.warning("Cannot project onto zero vector (|a| = 0)")
                vector_proj = np.zeros_like(vector_a)
        
        with col2:
            st.markdown("### Visualization")
            
            # Use display_vector_visualization function to show both original and projection
            if len(vector_a) <= 3 and len(vector_b) <= 3:  # Only visualize for 2D and 3D vectors
                # Define construction lines for showing perpendicular component and projection line
                construction_lines = []
                
                # Add line from tip of projection to tip of b (perpendicular component)
                if magnitude_a > 0:
                    perp_component = vector_b - vector_proj
                    tip_of_proj = vector_proj
                    tip_of_b = vector_b
                    construction_lines.append((tip_of_proj, tip_of_b, "Perpendicular Component", {"color": "red", "dash": "dash"}))
                
                # Display the vectors and construction
                display_vector_visualization(
                    [vector_a, vector_b, vector_proj] if magnitude_a > 0 else [vector_a, vector_b],
                    names=["Vector a", "Vector b", "Projection of b onto a"] if magnitude_a > 0 else ["Vector a", "Vector b"],
                    construction_lines=construction_lines
                )
            else:
                st.info(f"Direct visualization is only available for 2D and 3D vectors. Your vectors are {len(vector_a)}D and {len(vector_b)}D.")
                st.markdown(f"**Vector a:** {vector_a.tolist()}")
                st.markdown(f"**Vector b:** {vector_b.tolist()}")
                if magnitude_a > 0:
                    st.markdown(f"**Projection of b onto a:** {vector_proj.tolist()}")
        
        return result
        
    def vector_angle(self, vector_a_input, vector_b_input):
        """Calculate the angle between two vectors."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            # Define Args for CLI compatibility
            class Args:
                def __init__(self, vector_a, vector_b):
                    self.vector_a = vector_a
                    self.vector_b = vector_b
            
            args = Args(vector_a_input, vector_b_input)
            result = self.framework.vector_angle(args)
        
        # Get CLI output
        cli_output = output.get_output()
        
        # Extract angle values from the result
        angle_rad = result.get("angle_rad", 0)
        angle_deg = result.get("angle_deg", 0)
        
        # Display the results
        st.subheader("Angle Between Vectors")
        
        # Parse vectors for display and visualization
        vector_a = self.framework.parse_vector(vector_a_input)
        vector_b = self.framework.parse_vector(vector_b_input)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Display the original vectors
            st.markdown("**Original Vectors:**")
            st.markdown(f"$\\vec{{a}} = {vector_a.tolist()}$")
            st.markdown(f"$\\vec{{b}} = {vector_b.tolist()}$")
            
            # Calculate magnitudes
            mag_a = np.linalg.norm(vector_a)
            mag_b = np.linalg.norm(vector_b)
            
            # Calculate dot product
            dot_product = np.dot(vector_a, vector_b)
            
            # Formula and explanation
            st.markdown("**Angle Calculation:**")
            if mag_a > 0 and mag_b > 0:  # Avoid division by zero
                # Formula: cos(θ) = (a·b)/(|a|·|b|)
                cosine = dot_product / (mag_a * mag_b)
                # Handle potential numerical issues
                if cosine > 1.0: cosine = 1.0
                if cosine < -1.0: cosine = -1.0
                
                # Display the calculation steps
                st.markdown(f"$\\cos(\\theta) = \\frac{{\\vec{{a}} \\cdot \\vec{{b}}}}{{|\\vec{{a}}| \\cdot |\\vec{{b}}|}} = \\frac{{{dot_product}}}{{{mag_a:.4f} \\cdot {mag_b:.4f}}} = \\frac{{{dot_product}}}{{{mag_a * mag_b:.4f}}} = {cosine:.4f}$")
                
                # Calculate angle
                angle = np.arccos(cosine)
                st.markdown(f"$\\theta = \\arccos({cosine:.4f}) = {angle:.4f}$ radians = ${angle * 180/np.pi:.4f}°$")
                
                # Classify the angle
                if np.isclose(angle, 0, atol=1e-10):
                    st.markdown("**The vectors are parallel (pointing in the same direction)**")
                elif np.isclose(angle, np.pi, atol=1e-10):
                    st.markdown("**The vectors are antiparallel (pointing in opposite directions)**")
                elif np.isclose(angle, np.pi/2, atol=1e-10):
                    st.markdown("**The vectors are perpendicular (orthogonal)**")
                elif angle < np.pi/2:
                    st.markdown("**The vectors form an acute angle (less than 90°)**")
                else:
                    st.markdown("**The vectors form an obtuse angle (greater than 90°)**")
            else:
                st.warning("Cannot calculate angle involving zero vector(s)")
        
        with col2:
            st.markdown("### Visualization")
            
            # Use visualization function to display vectors 
            if len(vector_a) <= 3 and len(vector_b) <= 3:  # Only visualize 2D and 3D vectors
                display_vector_visualization(
                    [vector_a, vector_b],
                    names=["Vector a", "Vector b"]
                )
                
                # Show the angle
                if mag_a > 0 and mag_b > 0:
                    # Create a mock gauge chart for angle visualization
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = angle_deg,
                        title = {'text': "Angle (degrees)"},
                        gauge = {
                            'axis': {'range': [0, 180], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 90], 'color': "lightgreen"},
                                {'range': [90, 180], 'color': "lightyellow"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': angle_deg
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=30, r=30, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                    
                    st.plotly_chart(fig)
            else:
                st.info(f"Direct visualization is only available for 2D and 3D vectors. Your vectors are {len(vector_a)}D and {len(vector_b)}D.")
        
        return result

    def cross_product(self, args):
        """Calculate the cross product of two vectors."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.cross_product(args)
        
        # Since the framework's cross_product method returns the cross product directly (not in a dict)
        cross_prod = result if isinstance(result, np.ndarray) else np.array([0, 0, 0])
        
        # Display the results
        st.subheader("Cross Product")
        
        # Parse vectors for display/visualization
        vector_a = self.framework.parse_vector(args.vector_a)
        vector_b = self.framework.parse_vector(args.vector_b)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            # Display the original vectors
            st.markdown("**Original Vectors:**")
            st.markdown(f"$\\vec{{a}} = {vector_a.tolist()}$")
            st.markdown(f"$\\vec{{b}} = {vector_b.tolist()}$")
            
            # Handle 2D and 3D cases
            if len(vector_a) == 2 and len(vector_b) == 2:
                # For 2D, compute the 3D cross product with z=0
                a_3d = np.append(vector_a, 0)
                b_3d = np.append(vector_b, 0)
                cross_prod_3d = np.cross(a_3d, b_3d)
                
                # Show formula for 2D vectors
                st.markdown("**Converting 2D vectors to 3D (z=0) for cross product:**")
                st.markdown(f"$\\vec{{a}}_{{3D}} = [{vector_a[0]}, {vector_a[1]}, 0]$")
                st.markdown(f"$\\vec{{b}}_{{3D}} = [{vector_b[0]}, {vector_b[1]}, 0]$")
                
                st.markdown("**Cross Product Calculation:**")
                det_val = vector_a[0]*vector_b[1] - vector_a[1]*vector_b[0]
                st.markdown(f"$\\vec{{a}} \\times \\vec{{b}} = \\begin{{vmatrix}} \\vec{{i}} & \\vec{{j}} & \\vec{{k}} \\\\ {vector_a[0]} & {vector_a[1]} & 0 \\\\ {vector_b[0]} & {vector_b[1]} & 0 \\end{{vmatrix}} = [0, 0, {det_val}]$")
                
                st.markdown("**For 2D vectors, the cross product points along the z-axis:**")
                st.markdown(f"$\\vec{{a}} \\times \\vec{{b}} = [0, 0, {det_val}]$")
                
                # Compute magnitude (area of parallelogram)
                area = abs(det_val)
                st.markdown(f"**Magnitude |a×b| = {area:.4f}** (Area of parallelogram formed by a and b)")
                st.markdown(f"**Area of triangle = |a×b|/2 = {area/2:.4f}**")
                
            elif len(vector_a) == 3 and len(vector_b) == 3:
                # Compute the cross product for 3D vectors using the determinant formula
                st.markdown("**Cross Product Calculation:**")
                
                # Compute determinants for each component
                det_i = vector_a[1]*vector_b[2] - vector_a[2]*vector_b[1]
                det_j = vector_a[2]*vector_b[0] - vector_a[0]*vector_b[2]
                det_k = vector_a[0]*vector_b[1] - vector_a[1]*vector_b[0]
                
                st.markdown("$\\vec{a} \\times \\vec{b} = \\begin{vmatrix} \\vec{i} & \\vec{j} & \\vec{k} \\\\ " +
                           f"{vector_a[0]} & {vector_a[1]} & {vector_a[2]} \\\\ " +
                           f"{vector_b[0]} & {vector_b[1]} & {vector_b[2]} \\end{{vmatrix}}$")
                
                st.markdown(f"$\\vec{{a}} \\times \\vec{{b}} = [{det_i}, {det_j}, {det_k}]$")
                
                # Compute magnitude (area of parallelogram)
                area = np.linalg.norm(cross_prod)
                st.markdown(f"**Magnitude |a×b| = {area:.4f}** (Area of parallelogram formed by a and b)")
                st.markdown(f"**Area of triangle = |a×b|/2 = {area/2:.4f}**")
            else:
                st.warning(f"Cross product visualization is primarily designed for 3D vectors. Your vectors are {len(vector_a)}D and {len(vector_b)}D.")
        
        with col2:
            st.markdown("### Visualization")
            
            # Create vectors for visualization
            if (len(vector_a) == 2 and len(vector_b) == 2) or (len(vector_a) == 3 and len(vector_b) == 3):
                # For 2D vectors, use the 3D representation with z=0
                a = vector_a if len(vector_a) == 3 else np.append(vector_a, 0)
                b = vector_b if len(vector_b) == 3 else np.append(vector_b, 0)
                
                # Create a 3D visualization
                vectors_to_plot = [a, b, cross_prod]
                vector_names = ["Vector a", "Vector b", "a × b"]
                
                # Add construction lines to form a parallelogram
                a_plus_b = a + b
                construction_lines = []
                line_style = {"color": "rgba(200, 200, 200, 0.7)", "width": 2}
                
                # Draw the parallelogram
                construction_lines = [
                    (a, a_plus_b, "Side (a to a+b)", line_style),
                    (b, a_plus_b, "Side (b to a+b)", line_style)
                ]
                
                if len(cross_prod) == 3 and (np.any(cross_prod) or np.any(a) or np.any(b)): # Only visualize if cross product is 3D and non-zero, or a/b non-zero
                    display_vector_visualization(
                        [a, b, cross_prod],
                        names=["Vector a", "Vector b", "Cross Product (a × b)"],
                        construction_lines=construction_lines
                    )
                else:
                    # For 2D vectors (z=0), create a simpler visualization
                    display_vector_visualization([a, b, cross_prod], names=["Vector a", "Vector b", "Cross Product (a × b)"])
                
                # Add result info box
                result_box = f"""
                <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                    <p style="font-size: 16px; text-align: center;">
                        <strong>Cross Product:</strong> {cross_prod.tolist()}<br>
                        <strong>Area of Parallelogram:</strong> {area:.4f}
                    </p>
                </div>
                """
                st.markdown(result_box, unsafe_allow_html=True)
        
        return result

    def triangle_area(self, args):
        """Calculate the area of a triangle defined by three points."""
        # Parse the points from args
        p1 = self.framework.parse_vector(args.point_a)
        p2 = self.framework.parse_vector(args.point_b)
        p3 = self.framework.parse_vector(args.point_c)
        
        # Form vectors from points
        vec1 = p2 - p1
        vec2 = p3 - p1
        
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.triangle_area(args)
        
        # The framework's triangle_area method returns the area directly (not in a dict)
        area = result if isinstance(result, (int, float)) else 0
        
        # Calculate the area ourselves as a double-check
        if len(p1) == 3:  # 3D case
            cross_prod = np.cross(vec1, vec2)
            calculated_area = np.linalg.norm(cross_prod) / 2
        elif len(p1) == 2:  # 2D case
            determinant = vec1[0]*vec2[1] - vec1[1]*vec2[0]
            calculated_area = abs(determinant) / 2
        else:
            calculated_area = 0
            
        # Use the calculated area if it differs significantly from the returned area
        if abs(calculated_area - area) > 1e-6:
            area = calculated_area
        
        # Display the result and steps
        st.subheader("Result")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            st.markdown("**Input points:**")
            st.markdown(f"$P_1 = {p1.tolist()}$")
            st.markdown(f"$P_2 = {p2.tolist()}$")
            st.markdown(f"$P_3 = {p3.tolist()}$")
            
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
            point_b = self.framework.parse_vector(args.point_b)
            st.markdown(f"**Point B:** ${point_b.tolist()}$")
            
            # Parse the needed vectors
            point_a = self.framework.parse_vector(args.point_a)
            direction = self.framework.parse_vector(args.direction)
            
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
            
            if len(point_b) == 3 and len(direction_vec_line) == 3:
                cross_prod = np.cross(vec_ap, direction_vec_line)
                norm_d = np.linalg.norm(direction_vec_line)
                st.markdown(f"$\\text{{Distance}} = \\frac{{|\\vec{{AP}} \\times \\vec{{d}}|}}{{|\\vec{{d}}|}} = \\frac{{|{cross_prod.tolist()}|}}{{{norm_d:.4f}}} = \\frac{{{np.linalg.norm(cross_prod):.4f}}}{{{norm_d:.4f}}} = {distance:.4f}$")
            elif len(point_b) == 2 and len(direction_vec_line) == 2:
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
            
            dim = len(point_b)
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
            point_p_as_vec = point_b # For viz, treat P as vector from origin
            
            # Calculate closest point on line (Q) to P
            # Q = A + proj_d(AP)
            if np.dot(direction_vec_line, direction_vec_line) == 0:
                # Direction vector is zero, line is just a point A
                q_closest_point = point_a_on_line
            else:
                proj_vec_ap_onto_d = (np.dot(vec_ap, direction_vec_line) / np.dot(direction_vec_line, direction_vec_line)) * direction_vec_line
                q_closest_point = point_a_on_line + proj_vec_ap_onto_d
            
            # Line from P to Q (shortest distance)
            distance_line_style = dict(color='red', dash='dash', width=2)
            construction_lines.append((point_b, q_closest_point, "Distance", distance_line_style))
            
            # Show P and Q as points
            origin = np.zeros(dim)
            # primary_vectors.append(point_p_as_vec) # P as vector from origin
            # primary_names.append("Point P")
            # primary_vectors.append(q_closest_point) # Q as vector from origin
            # primary_names.append("Closest Point Q")
            
            # Show key vectors: AP, d, perpendicular P-Q
            primary_vectors = [vec_ap, direction_vec_line, (point_b - q_closest_point)]
            primary_names = ["Vector AB", "Direction d", "Perpendicular Distance"]
            
            # Visualize if dimensions allow
            if dim == 2 or dim == 3:
                display_vector_visualization(primary_vectors, names=primary_names, construction_lines=construction_lines)
            else:
                st.info(f"Direct visualization is for 2D/3D. Current dimension is {dim}.")
                
            # Show result
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
        """Check if vectors are collinear."""
        # Get the vectors from the arguments
        vectors = args.vectors
        
        # Parse each vector
        parsed_vectors = [self.framework.parse_vector(v) for v in vectors if v]
        
        # Implement the collinearity check directly since framework doesn't have it
        is_collinear = False
        
        # Use matrix rank method to determine collinearity
        if len(parsed_vectors) >= 2:
            # Create a matrix where each row is a vector
            matrix = np.vstack(parsed_vectors)
            # Calculate the rank - if 1, the vectors are collinear
            rank = np.linalg.matrix_rank(matrix)
            is_collinear = (rank == 1)
            
            # Store the results
            result = {"collinear": is_collinear, "method": "rank"}
        else:
            st.warning("Need at least 2 vectors to check collinearity.")
            result = {"collinear": None, "method": None}
        
        # Display the result and steps
        st.subheader("Result")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Step-by-step Calculation")
            
            st.markdown("**Input vectors:**")
            for i, vector in enumerate(parsed_vectors):
                st.markdown(f"$\\mathbf{{v}}_{i+1} = {vector.tolist()}$")
            
            st.markdown("**Method: Matrix Rank**")
            st.markdown("Vectors are collinear if they are scalar multiples of each other. This can be checked by creating a matrix where each row is a vector and checking if its rank is 1.")
            
            st.markdown(f"Matrix of vectors:")
            if len(parsed_vectors) >= 2:
                # Handling the matrix display without f-string issues
                row1 = ' & '.join(map(str, parsed_vectors[0]))
                row2 = ' & '.join(map(str, parsed_vectors[1]))
                additional_rows = ""
                if len(parsed_vectors) > 2:
                    additional_rows = " \\\\ " + " \\\\ ".join([' & '.join(map(str, v)) for v in parsed_vectors[2:]])
                st.markdown(f"$\\begin{{pmatrix}} {row1} \\\\ {row2}{additional_rows} \\end{{pmatrix}}$")
                st.markdown(f"Rank of matrix: {rank}")
                st.markdown(f"Since the rank is {('1' if is_collinear else 'greater than 1')}, the vectors are {('collinear' if is_collinear else 'not collinear')}.")
            else:
                st.warning("Not enough vectors to determine collinearity.")

        with col2:
            st.markdown("### Visualization")
            
            if len(parsed_vectors) >= 2:
                dim = len(parsed_vectors[0])  # Dimension of vectors
                
                # Visualize vectors from origin
                vector_names = [f"Vector {i+1}" for i in range(len(parsed_vectors))]
                
                # For visualization, show vectors from origin
                if dim == 2 or dim == 3:
                    display_vector_visualization(parsed_vectors, names=vector_names)
                else:
                    st.info(f"Direct visualization is for 2D/3D. Vectors are in {dim}D space.")
            else:
                st.info("Not enough vectors to visualize collinearity.")

            # Add result info box
            if len(parsed_vectors) >= 2:
                result_box = f"""
                <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                    <p style="font-size: 18px; text-align: center;">
                        Vectors are {('collinear!' if is_collinear else 'not collinear.')}
                    </p>
                </div>
                """
                st.markdown(result_box, unsafe_allow_html=True)
            
        return result