#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Linear Algebra Calculator Operations
"""

import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Not directly used in LinAlgCalculator, but good to keep if visualizations expand
import plotly.express as px
import plotly.graph_objects as go
from given_reference.core import mrref # mnull, eliminate are used by linalg_cli, not directly here
from .st_visualization_utils import display_vector_visualization, display_matrix_heatmap

# Import CLI functionality to reuse functions
from linalg_cli import LinearAlgebraExerciseFramework

# Import Utilities
from .st_utils import StreamOutput

class LinAlgCalculator:
    def __init__(self):
        self.framework = LinearAlgebraExerciseFramework()
    
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
        
        # Parse the vectors from input
        a = self.framework.parse_vector(args.vector_a)
        b = self.framework.parse_vector(args.vector_b)
        cross = result  # The cross product result
        
        # Ensure vectors are at least 3D (pad with zeros if needed)
        if len(a) < 3:
            a = np.pad(a, (0, 3 - len(a)), 'constant')
        if len(b) < 3:
            b = np.pad(b, (0, 3 - len(b)), 'constant')
            
        # Calculate magnitudes and angle for the formulas
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        dot_product = np.dot(a, b)
        cos_angle = dot_product / (norm_a * norm_b) if norm_a * norm_b > 1e-10 else 0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to avoid domain errors
        angle_rad = np.arccos(cos_angle)
        angle_deg = angle_rad * 180 / np.pi
        sin_angle = np.sin(angle_rad)
        cross_magnitude = np.linalg.norm(cross)
        
        # Check orthogonality with both input vectors
        dot_a = np.dot(cross, a)
        dot_b = np.dot(cross, b)
        
        # Create a formatted display of the calculations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Mathematical Calculation")
            
            # Create a more structured output with LaTeX formatting
            st.markdown("**Input vectors:**")
            st.markdown(f"$\\vec{{a}} = {a.tolist()}$")
            st.markdown(f"$\\vec{{b}} = {b.tolist()}$")
            
            st.markdown("**Cross product formula:**")
            st.markdown(r"""
            For 3D vectors $\vec{a} = [a_1, a_2, a_3]$ and $\vec{b} = [b_1, b_2, b_3]$, the cross product is:
            
            $\vec{a} \times \vec{b} = \begin{vmatrix} 
            \vec{i} & \vec{j} & \vec{k} \\
            a_1 & a_2 & a_3 \\
            b_1 & b_2 & b_3
            \end{vmatrix}$
            
            $= \vec{i} \begin{vmatrix} a_2 & a_3 \\ b_2 & b_3 \end{vmatrix} - 
            \vec{j} \begin{vmatrix} a_1 & a_3 \\ b_1 & b_3 \end{vmatrix} + 
            \vec{k} \begin{vmatrix} a_1 & a_2 \\ b_1 & b_2 \end{vmatrix}$
            """)
            
            # Component-by-component calculation
            st.markdown("**Component calculations:**")
            c1 = a[1]*b[2] - a[2]*b[1]
            c2 = a[2]*b[0] - a[0]*b[2]
            c3 = a[0]*b[1] - a[1]*b[0]
            
            st.markdown(f"$c_1 = a_2b_3 - a_3b_2 = ({a[1]})({b[2]}) - ({a[2]})({b[1]}) = {c1:.4f}$")
            st.markdown(f"$c_2 = a_3b_1 - a_1b_3 = ({a[2]})({b[0]}) - ({a[0]})({b[2]}) = {c2:.4f}$")
            st.markdown(f"$c_3 = a_1b_2 - a_2b_1 = ({a[0]})({b[1]}) - ({a[1]})({b[0]}) = {c3:.4f}$")
            
            st.markdown("**Result:**")
            st.markdown(f"$\\vec{{a}} \\times \\vec{{b}} = [{c1:.4f}, {c2:.4f}, {c3:.4f}]$")
            
            # Magnitude and orientation properties
            st.markdown("**Properties of the cross product:**")
            
            st.markdown("**1. Magnitude:**")
            st.markdown(f"$|\\vec{{a}} \\times \\vec{{b}}| = |\\vec{{a}}||\\vec{{b}}|\\sin(\\theta)$")
            st.markdown(f"$= {norm_a:.4f} \\cdot {norm_b:.4f} \\cdot \\sin({angle_deg:.2f}°)$")
            st.markdown(f"$= {norm_a:.4f} \\cdot {norm_b:.4f} \\cdot {sin_angle:.4f} = {norm_a * norm_b * sin_angle:.4f}$")
            st.markdown(f"Actual magnitude: $|\\vec{{a}} \\times \\vec{{b}}| = {cross_magnitude:.4f}$")
            
            st.markdown("**2. Orthogonality:**")
            st.markdown(f"$\\vec{{a}} \\cdot (\\vec{{a}} \\times \\vec{{b}}) = {dot_a:.10f} \\approx 0$ ✓")
            st.markdown(f"$\\vec{{b}} \\cdot (\\vec{{a}} \\times \\vec{{b}}) = {dot_b:.10f} \\approx 0$ ✓")
            
        with col2:
            st.markdown("### Visualization")
            
            if a[2] != 0 or b[2] != 0 or cross[2] != 0:  # At least one vector has a z component
                # For 3D vectors, visualize in 3D space
                # Create a 3D visualization using plotly
                fig = go.Figure()
                
                # Add the vectors
                vectors = [a, b, cross]
                names = ["Vector a", "Vector b", "Cross Product (a × b)"]
                colors = ["blue", "green", "red"]
                
                for i, vec in enumerate(vectors):
                    fig.add_trace(go.Scatter3d(
                        x=[0, vec[0]],
                        y=[0, vec[1]],
                        z=[0, vec[2]],
                        mode='lines+markers',
                        name=names[i],
                        line=dict(width=6, color=colors[i]),
                        marker=dict(size=[0, 8], color=colors[i])
                    ))
                
                # Add a semi-transparent plane to show perpendicularity
                # Get corners of a square in the plane containing a and b
                max_val = max([max(abs(v)) for v in vectors])
                    
                # Create a mesh for the plane (if vectors aren't collinear)
                if sin_angle > 0.01:  # Only create plane if angle is substantial
                    # Use a and b to define the plane
                    normal = cross / np.linalg.norm(cross)
                    
                    # Create a grid of points for the plane
                    u = np.linspace(-max_val, max_val, 10)
                    v = np.linspace(-max_val, max_val, 10)
                    U, V = np.meshgrid(u, v)
                    
                    # Create orthogonal vectors in the plane
                    v1 = a / np.linalg.norm(a)
                    v2 = np.cross(normal, v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # Compute the points on the plane
                    X = U[:,:,np.newaxis] * v1[np.newaxis,np.newaxis,:] + V[:,:,np.newaxis] * v2[np.newaxis,np.newaxis,:]
                    
                    # Add the plane to the figure
                    fig.add_trace(go.Surface(
                        x=X[:,:,0], y=X[:,:,1], z=X[:,:,2],
                        opacity=0.2,
                        colorscale=[[0, 'purple'], [1, 'purple']],
                        showscale=False,
                        name="Plane containing a and b"
                    ))
                    
                # Configure the layout
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            range=[-max_val * 1.2, max_val * 1.2], 
                            title="X",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        yaxis=dict(
                            range=[-max_val * 1.2, max_val * 1.2], 
                            title="Y",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        zaxis=dict(
                            range=[-max_val * 1.2, max_val * 1.2], 
                            title="Z",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        aspectmode='cube',
                    ),
                    title="Cross Product Visualization",
                    title_font_color="white",
                    width=700,
                    height=700,
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color="white")),
                )
                
                st.plotly_chart(fig)
            else:
                # For 2D vectors (z=0), create a simpler visualization
                display_vector_visualization([a, b, cross], names=["Vector a", "Vector b", "Cross Product (a × b)"])
            
            # Add result info box
            result_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 16px; text-align: center;">
                    <strong>Cross Product:</strong> {cross.tolist()}<br>
                    <strong>Magnitude:</strong> {cross_magnitude:.4f}<br>
                    <strong>Angle between vectors:</strong> {angle_deg:.2f}°
                </p>
                <p style="font-size: 14px; text-align: center; font-style: italic;">
                    The cross product is perpendicular to both input vectors,<br>
                    following the right-hand rule orientation.
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
        
        # Parse the points from input
        A = self.framework.parse_vector(args.point_a)
        B = self.framework.parse_vector(args.point_b)
        C = self.framework.parse_vector(args.point_c)
        
        # Calculate vectors for calculation and visualization
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = np.linalg.norm(cross) / 2
        
        # Create a formatted display of the calculations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Mathematical Calculation")
            
            # Create a more structured output with LaTeX formatting
            st.markdown("**Input points:**")
            st.markdown(f"$A = {A.tolist()}$")
            st.markdown(f"$B = {B.tolist()}$")
            st.markdown(f"$C = {C.tolist()}$")
            
            # Show the calculation method
            st.markdown("**Method: Cross Product**")
            st.markdown("For a triangle defined by three points, we can calculate its area using vectors and the cross product:")
            
            # Step 1: Create vectors
            st.markdown("**Step 1: Create vectors from point A to B and C**")
            st.markdown(f"$\\vec{{AB}} = B - A = {B.tolist()} - {A.tolist()} = {AB.tolist()}$")
            st.markdown(f"$\\vec{{AC}} = C - A = {C.tolist()} - {A.tolist()} = {AC.tolist()}$")
            
            # Step 2: Cross Product
            st.markdown("**Step 2: Calculate the cross product $\\vec{AB} \\times \\vec{AC}$**")
            
            # For 3D vectors, show the detailed calculation
            if len(AB) >= 3 and len(AC) >= 3:
                c1 = AB[1]*AC[2] - AB[2]*AC[1]
                c2 = AB[2]*AC[0] - AB[0]*AC[2]
                c3 = AB[0]*AC[1] - AB[1]*AC[0]
                
                st.markdown(f"$\\vec{{AB}} \\times \\vec{{AC}} = \\begin{{vmatrix}} \\vec{{i}} & \\vec{{j}} & \\vec{{k}} \\\\ {AB[0]} & {AB[1]} & {AB[2]} \\\\ {AC[0]} & {AC[1]} & {AC[2]} \\end{{vmatrix}}$")
                st.markdown(f"$= \\vec{{i}}\\begin{{vmatrix}} {AB[1]} & {AB[2]} \\\\ {AC[1]} & {AC[2]} \\end{{vmatrix}} - \\vec{{j}}\\begin{{vmatrix}} {AB[0]} & {AB[2]} \\\\ {AC[0]} & {AC[2]} \\end{{vmatrix}} + \\vec{{k}}\\begin{{vmatrix}} {AB[0]} & {AB[1]} \\\\ {AC[0]} & {AC[1]} \\end{{vmatrix}}$")
                st.markdown(f"$= [{c1}, {c2}, {c3}]$")
            else:
                # For 2D vectors, we just show the formula for the area
                st.markdown(f"For 2D points, we can compute the cross product using the determinant:")
                st.markdown(f"$\\vec{{AB}} \\times \\vec{{AC}} = \\begin{{vmatrix}} {AB[0]} & {AB[1]} \\\\ {AC[0]} & {AC[1]} \\end{{vmatrix}} = {AB[0]}\\cdot{AC[1]} - {AB[1]}\\cdot{AC[0]} = {AB[0]*AC[1] - AB[1]*AC[0]}$")
                st.markdown(f"The cross product in 2D yields a scalar, which is the z-component of the 3D cross product.")
            
            # Step 3: Calculate the area
            st.markdown("**Step 3: Calculate the area as half the magnitude of the cross product**")
            st.markdown(f"$\\text{{Area}} = \\frac{{|\\vec{{AB}} \\times \\vec{{AC}}|}}{{2}} = \\frac{{{np.linalg.norm(cross):.4f}}}{{2}} = {area:.4f}$")
            
            # Explain why this works
            st.markdown("**Why This Works**")
            st.markdown("""
            The magnitude of the cross product $|\\vec{AB} \\times \\vec{AC}|$ equals the area of the 
            parallelogram formed by the two vectors. Since a triangle is half of a parallelogram, 
            we divide by 2 to get the triangle area.
            """)
        
        with col2:
            st.markdown("### Visualization")
        
        # Create a figure based on dimensionality
        if len(A) == 2 and len(B) == 2 and len(C) == 2:
            # 2D Triangle
            fig = go.Figure()
            
            # Add the triangle vertices
            fig.add_trace(go.Scatter(
                x=[A[0], B[0], C[0], A[0]],  # Close the triangle by repeating the first point
                y=[A[1], B[1], C[1], A[1]],
                mode='lines+markers',
                name='Triangle',
                fill='toself',  # Fill the triangle
                line=dict(width=2),
                marker=dict(size=8)
            ))
                
            # Add the vectors from A to B and A to C
            fig.add_trace(go.Scatter(
                x=[A[0], A[0] + AB[0]],
                y=[A[1], A[1] + AB[1]],
                mode='lines+markers',
                name='Vector AB',
                line=dict(width=2, dash='dash', color='blue'),
                marker=dict(size=[0, 8])
            ))
            
            fig.add_trace(go.Scatter(
                x=[A[0], A[0] + AC[0]],
                y=[A[1], A[1] + AC[1]],
                mode='lines+markers',
                name='Vector AC',
                line=dict(width=2, dash='dash', color='green'),
                marker=dict(size=[0, 8])
            ))
            
            # Add labels for the points
            for i, point in enumerate([A, B, C]):
                fig.add_annotation(
                    x=point[0],
                    y=point[1],
                    text=f"Point {chr(65+i)}",  # A, B, C
                    showarrow=True,
                    arrowhead=1,
                    ax=10,
                    ay=-30
                )
            
            # Calculate the centroid for area label placement
            centroid = (A + B + C) / 3
            
            # Add area label
            fig.add_annotation(
                x=centroid[0],
                y=centroid[1],
                    text=f"Area: {area:.4f}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            
            # Set layout
            points = np.vstack([A, B, C])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            # Add some padding
            padding = max(x_max - x_min, y_max - y_min) * 0.2
            
            fig.update_layout(
                xaxis=dict(
                    range=[x_min - padding, x_max + padding],
                    title="X",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                yaxis=dict(
                    range=[y_min - padding, y_max + padding],
                    title="Y",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                title="Triangle Visualization",
                title_font_color="white",
                showlegend=True,
                width=700,
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
            
        elif len(A) == 3 and len(B) == 3 and len(C) == 3:
            # 3D Triangle
            fig = go.Figure()
            
            # Add the triangle
            fig.add_trace(go.Mesh3d(
                x=[A[0], B[0], C[0]],
                y=[A[1], B[1], C[1]],
                z=[A[2], B[2], C[2]],
                opacity=0.7,
                color='blue',
                name='Triangle'
            ))
                
            # Add the vectors from A to B and A to C
            fig.add_trace(go.Scatter3d(
                x=[A[0], A[0] + AB[0]],
                y=[A[1], A[1] + AB[1]],
                z=[A[2], A[2] + AB[2]],
                mode='lines',
                name='Vector AB',
                line=dict(width=5, dash='dash', color='blue')
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[A[0], A[0] + AC[0]],
                y=[A[1], A[1] + AC[1]],
                z=[A[2], A[2] + AC[2]],
                mode='lines',
                name='Vector AC',
                line=dict(width=5, dash='dash', color='green')
            ))
            
            # Add the vertices
            fig.add_trace(go.Scatter3d(
                x=[A[0], B[0], C[0]],
                y=[A[1], B[1], C[1]],
                z=[A[2], B[2], C[2]],
                mode='markers',
                marker=dict(size=8, color=['red', 'green', 'blue']),
                name='Vertices',
                text=['Point A', 'Point B', 'Point C']
            ))
            
            # Add edges
            edges = np.array([A, B, C, A])  # Close the loop
            fig.add_trace(go.Scatter3d(
                x=edges[:, 0],
                y=edges[:, 1],
                z=edges[:, 2],
                mode='lines',
                line=dict(width=4, color='white'),
                name='Edges'
            ))
                
            # Add normal vector (cross product)
            # Scale it for better visualization
            normal = cross / np.linalg.norm(cross) if np.linalg.norm(cross) > 0 else cross
            scale_factor = max(np.linalg.norm(AB), np.linalg.norm(AC)) * 0.5
            normal_scaled = normal * scale_factor
            
            # Position the normal vector at the centroid
            centroid = (A + B + C) / 3
            
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0] + normal_scaled[0]],
                y=[centroid[1], centroid[1] + normal_scaled[1]],
                z=[centroid[2], centroid[2] + normal_scaled[2]],
                mode='lines',
                name='Normal Vector',
                line=dict(width=5, color='red')
            ))
            
            # Configure the 3D layout
            points = np.vstack([A, B, C])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            
            # Calculate the ranges for each axis with padding
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            x_pad = (max_range - (x_max - x_min)) / 2
            y_pad = (max_range - (y_max - y_min)) / 2
            z_pad = (max_range - (z_max - z_min)) / 2
            
            padding = max_range * 0.3  # Increase padding to show vectors better
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        range=[x_min - x_pad - padding, x_max + x_pad + padding],
                        title="X",
                        color="white",
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        backgroundcolor='rgba(0,0,0,0.1)',
                    ),
                    yaxis=dict(
                        range=[y_min - y_pad - padding, y_max + y_pad + padding],
                        title="Y",
                        color="white",
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        backgroundcolor='rgba(0,0,0,0.1)',
                    ),
                    zaxis=dict(
                        range=[z_min - z_pad - padding, z_max + z_pad + padding],
                        title="Z",
                        color="white",
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        backgroundcolor='rgba(0,0,0,0.1)',
                    ),
                    aspectmode='cube',
                ),
                title=f"3D Triangle Visualization",
                title_font_color="white",
                width=700,
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
        else:
            # Higher dimensions or inconsistent dimensionality
            st.info("Triangle visualization is only available for 2D and 3D points. "
                   "Showing point coordinates instead.")
            
            # Display points as a table
            point_df = pd.DataFrame([A, B, C], index=["Point A", "Point B", "Point C"])
            st.table(point_df)
            
            # Add result info box
            result_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 18px; text-align: center; font-weight: bold;">
                    Triangle Area: {area:.4f}
                </p>
                <p style="font-size: 14px; text-align: center;">
                    Calculated using the cross product method: |AB × AC| / 2
                </p>
            </div>
            """
            st.markdown(result_box, unsafe_allow_html=True)
        
        return result
        
    def point_line_distance(self, args):
        """Calculate the distance from a point to a line."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.point_line_distance(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the points and direction vector from input
        A = self.framework.parse_vector(args.point_a)  # Point on line
        v = self.framework.parse_vector(args.direction)  # Direction vector
        B = self.framework.parse_vector(args.point_b)  # Point to find distance from
        
        # Ensure vectors are 3D (pad with zeros if needed)
        if len(A) < 3:
            A = np.pad(A, (0, 3 - len(A)), 'constant')
        if len(v) < 3:
            v = np.pad(v, (0, 3 - len(v)), 'constant')
        if len(B) < 3:
            B = np.pad(B, (0, 3 - len(B)), 'constant')
        
        # Calculate intermediate values for the formulas
        v_norm = np.linalg.norm(v)
        v_unit = v / v_norm if v_norm > 0 else v
        B_minus_A = B - A
        
        # Calculate the closest point first
        # Project B-A onto v to get the parallel component
        t_closest = np.dot(B_minus_A, v_unit)
        closest = A + t_closest * v_unit
        
        # The perpendicular vector from B to the line is B - closest
        perp_vector = B - closest
        distance = np.linalg.norm(perp_vector)
        
        # For visualization and verification, also calculate via cross product
        cross_product = np.cross(B_minus_A, v)
        cross_norm = np.linalg.norm(cross_product)
        distance_via_cross = cross_norm / v_norm
        
        # Verify both methods give same result (within numerical precision)
        if abs(distance - distance_via_cross) > 1e-10:
            st.warning("Warning: Distance calculations differ between methods. This might indicate a numerical precision issue.")
        
        # Create a formatted display of the calculations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Mathematical Calculation")
            
            # Create a more structured output with LaTeX formatting
            st.markdown("**Input parameters:**")
            st.markdown(f"$A = {A.tolist()}$ (point on the line)")
            st.markdown(f"$\\vec{{v}} = {v.tolist()}$ (direction vector of the line)")
            st.markdown(f"$B = {B.tolist()}$ (point to find distance from the line)")
            
            # Show the calculation method
            st.markdown("**Method: Cross Product Formula**")
            st.markdown("""
            To find the shortest distance from a point to a line, we use the cross product formula:
            
            $d = \\frac{|\\vec{v} \\times (B-A)|}{|\\vec{v}|}$
            
            Where:
            - $A$ is a point on the line
            - $\\vec{v}$ is the direction vector of the line
            - $B$ is the point to find the distance from
            - $\\times$ represents the cross product
            """)
            
            # Step 1: Calculate B-A
            st.markdown("**Step 1: Calculate vector from point on line to point B**")
            st.markdown(f"$B - A = {B.tolist()} - {A.tolist()} = {B_minus_A.tolist()}$")
            
            # Step 2: Calculate the cross product
            st.markdown("**Step 2: Calculate the cross product $\\vec{v} \\times (B-A)$**")
            st.markdown(f"$\\vec{{v}} \\times (B-A) = {v.tolist()} \\times {B_minus_A.tolist()} = {cross_product.tolist()}$")
            
            # Step 3: Calculate the magnitude of the cross product and direction vector
            st.markdown("**Step 3: Calculate magnitudes**")
            st.markdown(f"$|\\vec{{v}} \\times (B-A)| = {cross_norm:.4f}$")
            st.markdown(f"$|\\vec{{v}}| = {v_norm:.4f}$")
            
            # Step 4: Calculate the distance
            st.markdown("**Step 4: Calculate the distance**")
            st.markdown(f"$d = \\frac{{|\\vec{{v}} \\times (B-A)|}}{{|\\vec{{v}}|}} = \\frac{{{cross_norm:.4f}}}{{{v_norm:.4f}}} = {distance:.4f}$")
            
            # Alternative calculation using the closest point
            st.markdown("**Verification: Alternative Method**")
            st.markdown("We can also find the closest point on the line to B, then calculate the direct distance:")
            st.markdown(f"1. Find parameter $t$ such that the point $P = A + t\\vec{{v}}_{{unit}}$ is closest to B:")
            st.markdown(f"$t = (B-A) \\cdot \\vec{{v}}_{{unit}} = {B_minus_A.tolist()} \\cdot {v_unit.tolist()} = {t_closest:.4f}$")
            st.markdown(f"2. Calculate the closest point: $P = A + t\\vec{{v}}_{{unit}} = {A.tolist()} + {t_closest:.4f} \\cdot {v_unit.tolist()} = {closest.tolist()}$")
            st.markdown(f"3. Calculate direct distance: $|B - P| = |{B.tolist()} - {closest.tolist()}| = {np.linalg.norm(B - closest):.4f}$")
            
        with col2:
            st.markdown("### Visualization")
        
        # We'll create either a 2D or 3D visualization based on inputs
        if A[2] == 0 and v[2] == 0 and B[2] == 0:
            # 2D visualization (all z-coordinates are 0)
            fig = go.Figure()
            
            # Create the line by extending in both directions
            # Find appropriate line segment length based on the distance to point B
            line_length = max(10, 3 * np.linalg.norm(B - A))
            
            # Create line points extending in both directions
            t_vals = np.linspace(-line_length, line_length, 100)
            line_points = np.array([A + t * v_unit for t in t_vals])
            
            # Add the line
            fig.add_trace(go.Scatter(
                x=line_points[:, 0],
                y=line_points[:, 1],
                mode='lines',
                name='Line',
                line=dict(width=2, color='blue')
            ))
            
            # Add point A (on the line)
            fig.add_trace(go.Scatter(
                x=[A[0]],
                y=[A[1]],
                mode='markers',
                name='Point A (on line)',
                marker=dict(size=10, color='blue')
            ))
                
            # Add the direction vector
            scaled_v = v_unit * min(5, line_length / 3)  # Scale for better visualization
            fig.add_trace(go.Scatter(
                x=[A[0], A[0] + scaled_v[0]],
                y=[A[1], A[1] + scaled_v[1]],
                mode='lines+markers',
                name='Direction Vector v',
                line=dict(width=2, color='purple', dash='dot'),
                marker=dict(size=[0, 8], color='purple')
            ))
            
            # Add point B (to find distance from)
            fig.add_trace(go.Scatter(
                x=[B[0]],
                y=[B[1]],
                mode='markers',
                name='Point B',
                marker=dict(size=10, color='red')
            ))
            
            # Add the closest point on the line
            fig.add_trace(go.Scatter(
                x=[closest[0]],
                y=[closest[1]],
                mode='markers',
                    name='Closest Point P',
                marker=dict(size=8, color='green')
            ))
            
            # Draw the distance line (perpendicular to the original line)
            fig.add_trace(go.Scatter(
                x=[B[0], closest[0]],
                y=[B[1], closest[1]],
                mode='lines',
                    name=f'Distance: {distance:.4f}',
                line=dict(width=2, color='red', dash='dash')
            ))
            
            # Set layout
            points = np.vstack([A, B, closest])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            # Add some padding
            padding_x = max(1, (x_max - x_min) * 0.2)
            padding_y = max(1, (y_max - y_min) * 0.2)
            
            fig.update_layout(
                xaxis=dict(
                    range=[x_min - padding_x, x_max + padding_x],
                    title="X",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                yaxis=dict(
                    range=[y_min - padding_y, y_max + padding_y],
                    title="Y",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                title="Point-Line Distance Visualization",
                title_font_color="white",
                showlegend=True,
                width=700,
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                legend=dict(font=dict(color="white")),
            )
            
            # Add formula annotation
            fig.add_annotation(
                x=(B[0] + closest[0]) / 2,
                y=(B[1] + closest[1]) / 2 + padding_y * 0.3,
                    text=f"Distance = {distance:.4f}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            
            st.plotly_chart(fig)
            
        else:
            # 3D visualization
            fig = go.Figure()
            
            # Create the line by extending in both directions
            line_length = max(10, 3 * np.linalg.norm(B - A))
            
            # Create line points extending in both directions
            t_vals = np.linspace(-line_length, line_length, 100)
            line_points = np.array([A + t * v_unit for t in t_vals])
            
            # Add the line
            fig.add_trace(go.Scatter3d(
                x=line_points[:, 0],
                y=line_points[:, 1],
                z=line_points[:, 2],
                mode='lines',
                name='Line',
                line=dict(width=4, color='blue')
            ))
            
            # Add point A (on the line)
            fig.add_trace(go.Scatter3d(
                x=[A[0]],
                y=[A[1]],
                z=[A[2]],
                mode='markers',
                name='Point A (on line)',
                marker=dict(size=8, color='blue')
            ))
                
            # Add the direction vector
            scaled_v = v_unit * min(5, line_length / 3)  # Scale for better visualization
            fig.add_trace(go.Scatter3d(
                x=[A[0], A[0] + scaled_v[0]],
                y=[A[1], A[1] + scaled_v[1]],
                z=[A[2], A[2] + scaled_v[2]],
                mode='lines',
                name='Direction Vector v',
                line=dict(width=5, color='purple', dash='dot')
            ))
            
            # Add point B (to find distance from)
            fig.add_trace(go.Scatter3d(
                x=[B[0]],
                y=[B[1]],
                z=[B[2]],
                mode='markers',
                name='Point B',
                marker=dict(size=8, color='red')
            ))
            
            # Add the closest point on the line
            fig.add_trace(go.Scatter3d(
                x=[closest[0]],
                y=[closest[1]],
                z=[closest[2]],
                mode='markers',
                    name='Closest Point P',
                marker=dict(size=6, color='green')
            ))
            
            # Draw the distance line (perpendicular to the original line)
            fig.add_trace(go.Scatter3d(
                x=[B[0], closest[0]],
                y=[B[1], closest[1]],
                z=[B[2], closest[2]],
                mode='lines',
                    name=f'Distance: {distance:.4f}',
                line=dict(width=5, color='red', dash='dash')
            ))
            
            # Configure the 3D layout
            points = np.vstack([A, B, closest])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            
            # Calculate the ranges for each axis with padding
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            x_pad = (max_range - (x_max - x_min)) / 2
            y_pad = (max_range - (y_max - y_min)) / 2
            z_pad = (max_range - (z_max - z_min)) / 2
            
            padding = max_range * 0.2
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        range=[x_min - x_pad - padding, x_max + x_pad + padding],
                        title="X",
                        color="white",
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        backgroundcolor='rgba(0,0,0,0.1)',
                    ),
                    yaxis=dict(
                        range=[y_min - y_pad - padding, y_max + y_pad + padding],
                        title="Y",
                        color="white",
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        backgroundcolor='rgba(0,0,0,0.1)',
                    ),
                    zaxis=dict(
                        range=[z_min - z_pad - padding, z_max + z_pad + padding],
                        title="Z",
                        color="white",
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        backgroundcolor='rgba(0,0,0,0.1)',
                    ),
                    aspectmode='cube',
                ),
                    title=f"Point-Line Distance in 3D",
                title_font_color="white",
                width=700,
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
        
            # Add result info box
            result_box = f"""
            <div style="padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 18px; text-align: center; font-weight: bold;">
                    Distance from Point to Line: {distance:.4f}
                </p>
                <p style="font-size: 14px; text-align: center;">
                    Calculated using the formula: |v × (B-A)| / |v|
                </p>
            </div>
            """
            st.markdown(result_box, unsafe_allow_html=True)
            
            # Add some key facts about point-line distance
            st.markdown("### Key Facts")
            st.info("""
            **Properties of Point-Line Distance:**
            
            1. The shortest distance from a point to a line is always along the perpendicular path.
            2. The cross product formula gives the exact distance regardless of where point A is on the line.
            3. This distance remains constant for any point on the line to point B.
            """)
        
        return result
        
    def check_collinear(self, args):
        """Check if vectors are collinear (linearly dependent)."""
        # Get the boolean result from the framework first
        # We are not capturing the full StreamOutput anymore, as we'll format the explanation manually.
        # However, we still need the core logic to determine collinearity.
        # For this, we'll temporarily use StreamOutput to suppress prints if any, 
        # but primarily rely on the direct return value if the framework method supports it.
        # Assuming self.framework.check_collinear(args) returns a boolean or a structure from which boolean can be derived.
        
        # Let's simulate the args parsing that the framework would do, if needed, or pass it directly.
        # The framework.check_collinear(args) is expected to return a boolean.
        # If it prints and doesn't return, this part needs to be adapted to parse its print output for the boolean result.
        # For now, assuming it returns a boolean as 'result_is_collinear'.
        
        # It seems the original `self.framework.check_collinear(args)` was designed to print to console.
        # We need its boolean result. Let's assume it returns a boolean, or adapt if it only prints.
        # Re-evaluating the original: `result = self.framework.check_collinear(args)` captures the *return value*.
        # The StreamOutput was for capturing *prints*. 
        # So, the original `result` variable should hold the boolean True/False.

        result_is_collinear = self.framework.check_collinear(args)

        st.subheader("Result")
        
        # Parse the vectors from input for display
        parsed_vectors = []
        for vector_str in args.vectors:
            parsed_vectors.append(self.framework.parse_vector(vector_str))

        # Display input vectors
        st.markdown("#### Input Vectors:")
        vector_display_cols = st.columns(len(parsed_vectors))
        for i, vec in enumerate(parsed_vectors):
            with vector_display_cols[i]:
                st.markdown(f"$v_{i+1} = {vec.tolist()}$")

        st.markdown("---")
        st.markdown("#### Collinearity Check Explanation:")

        if len(parsed_vectors) < 2:
            st.warning("At least two vectors are needed to check for collinearity.")
            return

        # General explanation
        st.markdown(
            """Vectors are **collinear** if they all lie on the same line when starting from the origin. 
            This means they are linearly dependent."""
        )

        # Explanation based on number of vectors
        if len(parsed_vectors) == 2:
            st.markdown(
                """For two vectors, $v_1$ and $v_2$, they are collinear if one is a scalar multiple of the other:
                $v_2 = k \\cdot v_1$ for some scalar $k$.
                """)
            # Attempt to find k
            v1 = parsed_vectors[0]
            v2 = parsed_vectors[1]
            k = None
            if result_is_collinear:
                # Find first non-zero component in v1 to calculate k
                for i in range(len(v1)):
                    if abs(v1[i]) > 1e-9: # Avoid division by zero or near-zero
                        if abs(v2[i]) > 1e-9: # Ensure corresponding component in v2 is also non-zero for a meaningful ratio
                            k_candidate = v2[i] / v1[i]
                            # Verify with other components if they exist
                            is_consistent_k = True
                            for j in range(len(v1)):
                                if abs(v2[j] - k_candidate * v1[j]) > 1e-9:
                                    is_consistent_k = False
                                    break
                            if is_consistent_k:
                                k = k_candidate
                                break
                        else: # v1[i] is non-zero, but v2[i] is zero. If v2 is not all zeros, they can't be collinear with this k
                            if np.any(v2):
                                k = None # Mark as not found or inconsistent
                            else: # v2 is a zero vector, k can be 0
                                k = 0
                            break
                    elif abs(v2[i]) > 1e-9: # v1[i] is zero, but v2[i] is non-zero. Cannot be v2 = k*v1 unless v2 is zero vector
                        k = None # Mark as not found or inconsistent
                        break
                    # If both v1[i] and v2[i] are zero, continue to next component

        else: # More than 2 vectors
            st.markdown(
                """For multiple vectors, one common way to check for collinearity is to form a matrix where 
                each vector is a row (or column). The vectors are collinear if the **rank** of this matrix is 1 
                (assuming not all vectors are zero vectors)."""
            )
            # Display the matrix
            matrix_str = "\\\\begin{bmatrix}\n" 
            for vec in parsed_vectors:
                matrix_str += " & ".join(map(str, vec.tolist())) + "\\\\\\\\ \n"
            matrix_str += "\\end{bmatrix}"
            st.markdown(f"Matrix formed by the vectors (as rows): $M = {matrix_str}$")
            st.markdown("If rank(M) = 1, the vectors are collinear.")

        st.markdown("---")

        # Display the final result message
        if result_is_collinear:
            result_message = "The vectors are **collinear** (linearly dependent)."
            if len(parsed_vectors) == 2 and k is not None:
                result_message += f" Specifically, $v_2 \\approx {k:.4f} \\cdot v_1$."
            elif len(parsed_vectors) == 2 and k is None and not (np.all(parsed_vectors[0]==0) or np.all(parsed_vectors[1]==0)):
                 result_message += " They are collinear (e.g., one might be a zero vector, or they align but a simple scalar k was not uniquely determined by the method above)." 
            st.success(result_message)
            st.write("This means they all lie along the same line through the origin.")
        else:
            st.error("The vectors are **not collinear** (they are linearly independent).")
            st.write("This means they point in different directions and do not all lie on the same line through the origin.")
        
        # Display a visualization of the vectors
        st.subheader("Visualization")
        
        # Determine the dimensionality for visualization
        max_dim = max([len(v) for v in parsed_vectors])
        if max_dim > 3:
            st.info(f"Cannot visualize {max_dim}-dimensional vectors directly. Using tabular representation instead.")
            vector_df = pd.DataFrame(parsed_vectors, index=[f"Vector {i+1}" for i in range(len(parsed_vectors))])
            st.table(vector_df)
        else:
            # Pad vectors to consistent dimensionality if needed
            padded_vectors = []
            for v in parsed_vectors:
                if len(v) < max_dim:
                    padded_vectors.append(np.pad(v, (0, max_dim - len(v)), 'constant'))
                else:
                    padded_vectors.append(v)
            
            # Create a visualization based on dimensionality
            if max_dim <= 2:
                # Add a zero for 2D visualization
                vis_vectors = []
                for v_pad in padded_vectors: # Corrected variable name
                    if len(v_pad) == 1:
                        vis_vectors.append(np.array([v_pad[0], 0]))
                    else:
                        vis_vectors.append(v_pad)
                
                # Create 2D visualization
                fig = go.Figure()
                
                # Add vectors as arrows from origin
                max_val_vis = 0 # Renamed to avoid conflict
                for i, vec_vis in enumerate(vis_vectors):
                    max_val_vis = max(max_val_vis, max(abs(vec_vis)) if len(vec_vis) > 0 else 0)
                    fig.add_trace(go.Scatter(
                        x=[0, vec_vis[0]],
                        y=[0, vec_vis[1]],
                        mode='lines+markers',
                        name=f"Vector {i+1}",
                        line=dict(width=3),
                        marker=dict(size=[0, 10])
                    ))
                
                # Determine the layout extent
                # max_val = max([max(abs(vec)) for vec in vis_vectors]) * 1.2 # Original line causing error if vis_vectors is empty or contains empty arrays
                max_val_layout = max(1, max_val_vis) * 1.2 # Use max_val_vis and ensure it's at least 1

                # Add the collinearity result to the title
                collinear_status = "collinear ✓" if result_is_collinear else "not collinear ✗"
                
                # Configure the layout
                fig.update_layout(
                    xaxis=dict(
                        range=[-max_val_layout, max_val_layout],
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='white',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        color='white',
                    ),
                    yaxis=dict(
                        range=[-max_val_layout, max_val_layout],
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='white',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        color='white',
                    ),
                    title=f"Vector Visualization - Vectors are {collinear_status}",
                    title_font_color="white",
                    showlegend=True,
                    width=700,
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.1)',
                    legend=dict(font=dict(color="white")),
                )
                
                st.plotly_chart(fig)
                
                # Add explanation
                if result_is_collinear:
                    ratio_text = ""
                    if len(parsed_vectors) == 2:
                        # Calculate the ratio between the vectors if there are only 2
                        v1, v2 = parsed_vectors[0], parsed_vectors[1]
                        # Find the first non-zero component to calculate ratio
                        for i in range(len(v1)):
                            if abs(v1[i]) > 1e-10 and abs(v2[i]) > 1e-10:
                                ratio = v2[i] / v1[i]
                                ratio_text = f"Vector 2 = {ratio:.4f} · Vector 1"
                                break
                    
                    # st.success(f"The vectors are collinear (linearly dependent). {ratio_text}") # Covered by main success message
                    # st.write("This means they all lie along the same line through the origin, or in other words, one is a scalar multiple of the other.")
                else:
                    # st.error("The vectors are not collinear (they are linearly independent).") # Covered by main error message
                    pass # No additional text needed here as it's covered by the main error message.
            
            else:  # 3D visualization
                # Create 3D visualization
                fig = go.Figure()
                
                # Add vectors as arrows from origin
                max_val_vis_3d = 0 # Renamed
                for i, vec_pad_3d in enumerate(padded_vectors):
                    max_val_vis_3d = max(max_val_vis_3d, max(abs(vec_pad_3d)) if len(vec_pad_3d) > 0 else 0)
                    fig.add_trace(go.Scatter3d(
                        x=[0, vec_pad_3d[0]],
                        y=[0, vec_pad_3d[1]],
                        z=[0, vec_pad_3d[2]],
                        mode='lines+markers',
                        name=f"Vector {i+1}",
                        line=dict(width=6),
                        marker=dict(size=[0, 8])
                    ))
                
                # Determine the layout extent
                # max_val = max([max(abs(vec)) for vec in padded_vectors]) * 1.2 # Original line
                max_val_layout_3d = max(1, max_val_vis_3d) * 1.2 # Use max_val_vis_3d and ensure it's at least 1
                
                # Add the collinearity result to the title
                collinear_status = "collinear ✓" if result_is_collinear else "not collinear ✗"
                
                # Configure the layout
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            range=[-max_val_layout_3d, max_val_layout_3d],
                            title="X",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        yaxis=dict(
                            range=[-max_val_layout_3d, max_val_layout_3d],
                            title="Y",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        zaxis=dict(
                            range=[-max_val_layout_3d, max_val_layout_3d],
                            title="Z",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        aspectmode='cube',
                    ),
                    title=f"3D Vector Visualization - Vectors are {collinear_status}",
                    title_font_color="white",
                    width=700,
                    height=700,
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color="white")),
                )
                
                st.plotly_chart(fig)
                
                # Add explanation for 3D case, similar to 2D
                if result_is_collinear:
                    ratio_text = ""
                    if len(parsed_vectors) == 2:
                        # Calculate the ratio between the vectors if there are only 2
                        v1, v2 = parsed_vectors[0], parsed_vectors[1]
                        # Find the first non-zero component to calculate ratio (simplified, assumes same dim)
                        for i in range(len(v1)):
                            if abs(v1[i]) > 1e-10 and abs(v2[i]) > 1e-10: # Basic check
                                ratio = v2[i] / v1[i]
                                # A more robust check would verify this ratio across all components
                                # For simplicity, we use the k calculated earlier if available or just state collinearity.
                                # The main result message already handles the k display.
                                break
                    # st.success(f"The vectors are collinear. {ratio_text}") # Covered by main success message
                else:
                    # st.error("The vectors are not collinear.") # Covered by main error message
                    pass # No additional text needed here.

        
        return result_is_collinear