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
# import sympy as sym # Not directly used in LinAlgCalculator

# Import CLI functionality to reuse functions
from linalg_cli import LinearAlgebraExerciseFramework

# Import Utilities
from .st_utils import StreamOutput

class LinAlgCalculator:
    def __init__(self):
        self.framework = LinearAlgebraExerciseFramework()
    
    def display_vector_visualization(self, vectors, names=None, origin=None):
        """Create a visualization for vectors."""
        if names is None:
            names = [f"Vector {i+1}" for i in range(len(vectors))]
        
        # Make sure all vectors are of the same dimensionality (2D or 3D)
        dim = len(vectors[0])
        for vec in vectors:
            if len(vec) != dim:
                st.warning(f"Cannot visualize vectors with different dimensions: {dim} vs {len(vec)}")
                return
        
        if origin is None:
            origin = np.zeros(dim)
        
        if dim == 2:
            # Create a 2D visualization using plotly
            fig = go.Figure()
            
            # Add vectors as arrows
            max_val = 0
            for i, vec in enumerate(vectors):
                max_val = max(max_val, max(abs(vec[0]), abs(vec[1])))
                fig.add_trace(go.Scatter(
                    x=[origin[0], origin[0] + vec[0]],
                    y=[origin[1], origin[1] + vec[1]],
                    mode='lines+markers',
                    name=names[i],
                    line=dict(width=3),
                    marker=dict(size=[0, 10])
                ))
            
            # Add grid and configuration
            fig.update_layout(
                xaxis=dict(
                    range=[-max_val * 1.2, max_val * 1.2],
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                yaxis=dict(
                    range=[-max_val * 1.2, max_val * 1.2],
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                title="Vector Visualization",
                title_font_color="white",
                showlegend=True,
                width=600,
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
        
        elif dim == 3:
            # Create a 3D visualization using plotly
            fig = go.Figure()
            
            # Add vectors as arrows
            max_val = 0
            for i, vec in enumerate(vectors):
                max_val = max(max_val, max(abs(vec[0]), abs(vec[1]), abs(vec[2])))
                fig.add_trace(go.Scatter3d(
                    x=[origin[0], origin[0] + vec[0]],
                    y=[origin[1], origin[1] + vec[1]],
                    z=[origin[2], origin[2] + vec[2]],
                    mode='lines+markers',
                    name=names[i],
                    line=dict(width=6),
                    marker=dict(size=[0, 8])
                ))
            
            # Add grid and configuration
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
                title="3D Vector Visualization",
                title_font_color="white",
                width=700,
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
        else:
            st.info(f"Cannot visualize {dim}-dimensional vectors directly. Using tabular representation instead.")
            # Display vectors as a table
            vector_df = pd.DataFrame(vectors, index=names)
            st.table(vector_df)
    
    def display_matrix_heatmap(self, matrix, title="Matrix Visualization"):
        """Create a heatmap visualization for matrices."""
        matrix = np.array(matrix)
        
        # Create a heatmap using plotly
        fig = px.imshow(
            matrix,
            labels=dict(x="Column", y="Row", color="Value"),
            x=[f"Col {i+1}" for i in range(matrix.shape[1])],
            y=[f"Row {i+1}" for i in range(matrix.shape[0])],
            text_auto=True,
            color_continuous_scale='Viridis',
            title=title
        )
        
        # Update layout for dark theme compatibility
        fig.update_layout(
            width=800, 
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            title_font_color="white",
            font=dict(color="white"),
            xaxis=dict(color="white", gridcolor='rgba(255, 255, 255, 0.2)'),
            yaxis=dict(color="white", gridcolor='rgba(255, 255, 255, 0.2)'),
        )
        st.plotly_chart(fig)
    
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
            self.display_vector_visualization([vector, result], names=["Original Vector", "Normalized Vector"])
        
        return result
    
    def vector_shadow(self, vector_a, vector_b):
        """Calculate the shadow (projection) of one vector onto another."""
        # Create an Args object with the necessary attributes
        class Args:
            def __init__(self, vector_a, vector_b):
                self.vector_a = vector_a
                self.vector_b = vector_b
        
        args = Args(vector_a, vector_b)
        
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.vector_shadow(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the vectors from input
        a = self.framework.parse_vector(vector_a)
        b = self.framework.parse_vector(vector_b)
        vector_proj = result["vector_proj"]
        
        # Display steps calculation 
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display a visualization of the vectors and the projection
        st.subheader("Visualization")
        self.display_vector_visualization([a, b, vector_proj], names=["Vector a", "Vector b", "Projection of b onto a"])
        
        
        return result
    
    def vector_angle(self, vector_a, vector_b):
        """Calculate the angle between two vectors."""
        # Create an Args object with the necessary attributes
        class Args:
            def __init__(self, vector_a, vector_b):
                self.vector_a = vector_a
                self.vector_b = vector_b
        
        args = Args(vector_a, vector_b)
        
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.vector_angle(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the vectors from input
        a = self.framework.parse_vector(vector_a)
        b = self.framework.parse_vector(vector_b)
        
        # Display steps calculation 
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display a visualization of the vectors and angle
        st.subheader("Visualization")
        
        # For 2D vectors, create a special angle visualization
        if len(a) == 2 and len(b) == 2:
            # Create a figure for the angle visualization
            fig = go.Figure()
            
            # Add the vectors
            fig.add_trace(go.Scatter(
                x=[0, a[0]], y=[0, a[1]],
                mode='lines+markers',
                name='Vector a',
                line=dict(width=3),
                marker=dict(size=[0, 10])
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, b[0]], y=[0, b[1]],
                mode='lines+markers',
                name='Vector b',
                line=dict(width=3),
                marker=dict(size=[0, 10])
            ))
            
            # Add an arc to show the angle
            angle_deg = result["degrees"]
            r = min(np.linalg.norm(a), np.linalg.norm(b)) * 0.3
            
            # Calculate reference angle to x-axis for vector a
            angle_a = np.arctan2(a[1], a[0])
            angle_b = np.arctan2(b[1], b[0])
            
            # Ensure we draw the smaller angle
            if angle_b - angle_a > np.pi:
                angle_a += 2 * np.pi
            elif angle_a - angle_b > np.pi:
                angle_b += 2 * np.pi
            
            # Create points for the arc
            if angle_a <= angle_b:
                theta = np.linspace(angle_a, angle_b, 50)
            else:
                theta = np.linspace(angle_b, angle_a, 50)
            
            x_arc = r * np.cos(theta)
            y_arc = r * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_arc, y=y_arc,
                mode='lines',
                name=f'Angle: {angle_deg:.2f}°',
                line=dict(width=2, color='red', dash='dash'),
            ))
            
            # Add a label for the angle
            mid_angle = (angle_a + angle_b) / 2
            label_r = r * 1.2
            fig.add_annotation(
                x=label_r * np.cos(mid_angle),
                y=label_r * np.sin(mid_angle),
                text=f"{angle_deg:.2f}°",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            
            # Configure the layout
            max_norm = max(np.linalg.norm(a), np.linalg.norm(b))
            fig.update_layout(
                xaxis=dict(
                    range=[-max_norm * 1.2, max_norm * 1.2],
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                yaxis=dict(
                    range=[-max_norm * 1.2, max_norm * 1.2],
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    color='white',
                ),
                title="Angle Visualization",
                title_font_color="white",
                showlegend=True,
                width=600,
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
        else:
            # For higher dimensions, just show the vectors
            self.display_vector_visualization([a, b], names=["Vector a", "Vector b"])
        
        
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
        
        # Display steps calculation 
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display a visualization of the vectors and the cross product
        st.subheader("Visualization")
        
        if len(a) == 3 and len(b) == 3:
            # For 3D vectors, visualize in 3D space
            cross = result  # The cross product result
            
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
            
            # Add grid and configuration
            max_val = max([max(abs(v)) for v in vectors])
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
            
            # Add verification notes
            st.write("### Verification")
            
            # Check orthogonality with both input vectors
            dot_a = np.dot(cross, a)
            dot_b = np.dot(cross, b)
            
            st.write(f"**Orthogonality Check:**")
            st.write(f"- Dot product with Vector a: {dot_a:.10f} ≈ 0 ✓")
            st.write(f"- Dot product with Vector b: {dot_b:.10f} ≈ 0 ✓")
            
            # Show magnitude relationship
            # Ensure denominator is not zero before division
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            dot_ab_norm = np.dot(a, b) / (norm_a * norm_b) if norm_a * norm_b > 1e-10 else 0
            # Clip to avoid domain errors with arccos
            angle_rad_for_mag = np.arccos(np.clip(dot_ab_norm, -1.0, 1.0))
            denominator_mag = norm_a * norm_b * np.sin(angle_rad_for_mag)

            if abs(denominator_mag) > 1e-10:
                magnitude_relation = np.linalg.norm(cross) / denominator_mag
                st.write(f"**Magnitude Relationship:**")
                st.write(f"- |a × b| = |a|·|b|·sin(θ) ✓ (Ratio: {magnitude_relation:.6f} ≈ 1)")
            else:
                st.write(f"**Magnitude Relationship:**")
                st.write(f"- Could not verify magnitude relationship due to zero or near-zero denominator (vectors might be collinear).")
            
        else:
            # For 2D vectors or padded vectors, show the 3 vectors
            if len(a) < 3:
                a = np.pad(a, (0, 3 - len(a)), 'constant')
            if len(b) < 3:
                b = np.pad(b, (0, 3 - len(b)), 'constant')
                
            self.display_vector_visualization([a, b, result], names=["Vector a", "Vector b", "Cross Product (a × b)"])
        
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
        
        # Display steps calculation
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display a visualization of the triangle
        st.subheader("Visualization")
        
        # Calculate vectors for visualization
        AB = B - A
        AC = C - A
        
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
                text=f"Area: {result:.4f}",
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
                title=f"3D Triangle - Area: {result:.4f}",
                title_font_color="white",
                width=700,
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
            
            # Add verification notes - show calculations
            st.write("### Verification")
            
            # Show cross product is perpendicular to both triangle sides
            cross = np.cross(AB, AC)
            dot_AB = np.dot(cross, AB)
            dot_AC = np.dot(cross, AC)
            
            st.write("**Cross Product Orthogonality Check:**")
            st.write(f"- Dot product with vector AB: {dot_AB:.10f} ≈ 0 ✓")
            st.write(f"- Dot product with vector AC: {dot_AC:.10f} ≈ 0 ✓")
            
            # Show area calculation
            st.write("**Area Calculation:**")
            st.write(f"- |AB × AC| / 2 = {np.linalg.norm(cross) / 2:.4f} ✓")
            
        else:
            # Higher dimensions or inconsistent dimensionality
            st.info("Triangle visualization is only available for 2D and 3D points. "
                   "Showing point coordinates instead.")
            
            # Display points as a table
            point_df = pd.DataFrame([A, B, C], index=["Point A", "Point B", "Point C"])
            st.table(point_df)
            
            # Still show the area
            st.success(f"Triangle area: {result:.4f}")
        
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
        
        # Display steps calculation
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display a visualization of the point and line
        st.subheader("Visualization")
        
        # We'll create either a 2D or 3D visualization based on inputs
        if A[2] == 0 and v[2] == 0 and B[2] == 0:
            # 2D visualization (all z-coordinates are 0)
            fig = go.Figure()
            
            # Create the line by extending in both directions
            # Find appropriate line segment length based on the distance to point B
            line_length = max(10, 3 * np.linalg.norm(B - A))
            
            # Normalize direction vector
            v_unit = v / np.linalg.norm(v)
            
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
            
            # Add point B (to find distance from)
            fig.add_trace(go.Scatter(
                x=[B[0]],
                y=[B[1]],
                mode='markers',
                name='Point B',
                marker=dict(size=10, color='red')
            ))
            
            # Calculate the closest point on the line to B
            # Formula: closest = A + ((B-A)·v_unit) * v_unit
            closest = A + np.dot(B - A, v_unit) * v_unit
            
            # Add the closest point on the line
            fig.add_trace(go.Scatter(
                x=[closest[0]],
                y=[closest[1]],
                mode='markers',
                name='Closest Point',
                marker=dict(size=8, color='green')
            ))
            
            # Draw the distance line (perpendicular to the original line)
            fig.add_trace(go.Scatter(
                x=[B[0], closest[0]],
                y=[B[1], closest[1]],
                mode='lines',
                name=f'Distance: {result:.4f}',
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
                text=f"Distance = {result:.4f}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            
            st.plotly_chart(fig)
            
        else:
            # 3D visualization
            fig = go.Figure()
            
            # Create the line by extending in both directions
            line_length = max(10, 3 * np.linalg.norm(B - A))
            
            # Normalize direction vector
            v_unit = v / np.linalg.norm(v)
            
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
            
            # Add point B (to find distance from)
            fig.add_trace(go.Scatter3d(
                x=[B[0]],
                y=[B[1]],
                z=[B[2]],
                mode='markers',
                name='Point B',
                marker=dict(size=8, color='red')
            ))
            
            # Calculate the closest point on the line to B
            closest = A + np.dot(B - A, v_unit) * v_unit
            
            # Add the closest point on the line
            fig.add_trace(go.Scatter3d(
                x=[closest[0]],
                y=[closest[1]],
                z=[closest[2]],
                mode='markers',
                name='Closest Point',
                marker=dict(size=6, color='green')
            ))
            
            # Draw the distance line (perpendicular to the original line)
            fig.add_trace(go.Scatter3d(
                x=[B[0], closest[0]],
                y=[B[1], closest[1]],
                z=[B[2], closest[2]],
                mode='lines',
                name=f'Distance: {result:.4f}',
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
                title=f"Point-Line Distance in 3D - Distance: {result:.4f}",
                title_font_color="white",
                width=700,
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="white")),
            )
            
            st.plotly_chart(fig)
        
        # Add verification notes
        st.write("### Verification")
        
        # Show the vector from A to B
        AB = B - A
        
        # Show the distance calculation
        st.write("**Distance Calculation:**")
        st.write(f"- |v × (B-A)| / |v| = {np.linalg.norm(np.cross(v, B - A)) / np.linalg.norm(v):.4f} ✓")
        
        # Compare to the distance between the point and the closest point on the line
        closest = A + np.dot(B - A, v_unit) * v_unit
        direct_distance = np.linalg.norm(B - closest)
        
        st.write(f"- Alternate calculation (direct distance to closest point): {direct_distance:.4f} ✓")
        
        return result
        
    def check_collinear(self, args):
        """Check if vectors are collinear (linearly dependent)."""
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.check_collinear(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the vectors from input
        vectors = []
        for vector_str in args.vectors:
            vectors.append(self.framework.parse_vector(vector_str))
        
        # Display steps calculation
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display a visualization of the vectors
        st.subheader("Visualization")
        
        # Determine the dimensionality for visualization
        max_dim = max([len(v) for v in vectors])
        if max_dim > 3:
            st.info(f"Cannot visualize {max_dim}-dimensional vectors directly. Using tabular representation instead.")
            vector_df = pd.DataFrame(vectors, index=[f"Vector {i+1}" for i in range(len(vectors))])
            st.table(vector_df)
        else:
            # Pad vectors to consistent dimensionality if needed
            padded_vectors = []
            for v in vectors:
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
                collinear_status = "collinear ✓" if result else "not collinear ✗"
                
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
                if result:
                    ratio_text = ""
                    if len(vectors) == 2:
                        # Calculate the ratio between the vectors if there are only 2
                        v1, v2 = vectors[0], vectors[1]
                        # Find the first non-zero component to calculate ratio
                        for i in range(len(v1)):
                            if abs(v1[i]) > 1e-10 and abs(v2[i]) > 1e-10:
                                ratio = v2[i] / v1[i]
                                ratio_text = f"Vector 2 = {ratio:.4f} · Vector 1"
                                break
                    
                    st.success(f"The vectors are collinear (linearly dependent). {ratio_text}")
                    st.write("This means they all lie along the same line through the origin, or in other words, one is a scalar multiple of the other.")
                else:
                    st.error("The vectors are not collinear (they are linearly independent).")
                    st.write("This means they point in different directions and cannot be expressed as scalar multiples of each other.")
            
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
                collinear_status = "collinear ✓" if result else "not collinear ✗"
                
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
                
                # Add explanation
                if result:
                    ratio_text = ""
                    if len(vectors) == 2:
                        # Calculate the ratio between the vectors if there are only 2
                        v1, v2 = vectors[0], vectors[1]
                        # Find the first non-zero component to calculate ratio
                        for i in range(len(v1)):
                            if abs(v1[i]) > 1e-10 and abs(v2[i]) > 1e-10:
                                ratio = v2[i] / v1[i]
                                ratio_text = f"Vector 2 = {ratio:.4f} · Vector 1"
                                break
                    
                    st.success(f"The vectors are collinear (linearly dependent). {ratio_text}")
                    st.write("This means they all lie along the same line through the origin, or in other words, one is a scalar multiple of the other.")
                else:
                    st.error("The vectors are not collinear (they are linearly independent).")
                    st.write("This means they point in different directions and cannot be expressed as scalar multiples of each other.")
        
        # Add verification section
        st.write("### Verification")
        
        # Show the reduced form of the matrix of vectors
        if len(vectors) <= 4:  # Only show detailed verification for smaller sets
            st.write("**Matrix of Vectors:**")
            matrix = np.array(vectors)
            st.write(pd.DataFrame(matrix))
            
            # Show the reduced row echelon form
            st.write("**Reduced Row Echelon Form:**")
            reduced_matrix = mrref(matrix) # Renamed variable
            st.write(pd.DataFrame(reduced_matrix))
            
            # Check for zero rows to determine linear dependence
            rank = np.sum([np.linalg.norm(row) > 1e-10 for row in reduced_matrix])
            st.write(f"**Rank of Matrix:** {rank}")
            
            if rank < len(vectors) and rank <= 1:
                st.write("Since the rank is less than the number of vectors and at most 1, the vectors are collinear.")
            else:
                st.write("Since the rank is greater than 1, the vectors are not collinear.")
        
        return result
    
    def matrix_operations(self, operation, matrix_a, matrix_b=None, scalar=None):
        """Perform basic matrix operations."""
        # Create an Args object with the necessary attributes
        class Args:
            def __init__(self, operation, matrix_a, matrix_b=None, scalar=None):
                self.operation = operation
                self.matrix_a = matrix_a
                self.matrix_b = matrix_b
                self.scalar = scalar
        
        args = Args(operation, matrix_a, matrix_b, scalar)
        
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.matrix_operations(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the matrices from input
        A = self.framework.parse_matrix(matrix_a)
        
        # Display steps calculation 
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display matrix visualizations
        st.subheader("Visualization")
        
        # Display input matrices
        col1, col2 = st.columns(2)
        with col1:
            self.display_matrix_heatmap(A, "Matrix A")
        
        if operation in ["add", "subtract"] and matrix_b:
            B = self.framework.parse_matrix(matrix_b)
            with col2:
                self.display_matrix_heatmap(B, "Matrix B")
        
        # Display result matrix
        if result is not None:
            operation_name = {
                "add": "A + B",
                "subtract": "A - B",
                "multiply_scalar": f"{scalar} * A",
                "transpose": "A^T (transpose)"
            }.get(operation, "Result")
            
            self.display_matrix_heatmap(result, f"Result: {operation_name}")
        
        
        return result
    
    def solve_gauss(self, matrix_input):
        """Solve a system of linear equations using Gaussian elimination."""
        # Create an Args object with the necessary attributes
        class Args:
            def __init__(self, matrix):
                self.matrix = matrix
        
        args = Args(matrix_input)
        
        # Capture print output from the CLI function
        with StreamOutput() as output:
            result = self.framework.solve_gauss(args)
        
        # Display the result and steps
        st.subheader("Result")
        
        # Parse the matrix from input
        augmented = self.framework.parse_matrix(matrix_input)
        
        # Display steps calculation 
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display matrix visualizations
        st.subheader("Visualization")
        
        # Display the augmented matrix
        self.display_matrix_heatmap(augmented, "Augmented Matrix [A|b]")
        
        # Display the reduced row echelon form
        reduced_matrix_gauss = mrref(augmented) # Renamed
        self.display_matrix_heatmap(reduced_matrix_gauss, "Reduced Row Echelon Form")
        
        # For 2D or 3D systems, visualize the solution space
        if augmented.shape[1] - 1 in [2, 3]:
            st.markdown("**Solution Space Visualization:**")
            
            # Get the system information
            A_gauss = augmented[:, :-1] # Renamed
            b_gauss = augmented[:, -1] # Renamed
            
            # Check if the system has a unique solution
            if result is not None and not isinstance(result, dict):
                # This is a unique solution
                if len(result) == 2:
                    # 2D solution
                    x, y = result
                    
                    # Create a 2D plot for the system
                    fig = go.Figure()
                    
                    # For each equation, plot the line
                    for i in range(A_gauss.shape[0]):
                        a_eq, b_val = A_gauss[i, 0], A_gauss[i, 1]
                        c_eq = b_gauss[i]
                        
                        if abs(b_val) > 1e-10:
                            # b_val is not zero, so we can plot y = (c - a*x) / b_val
                            x_vals = np.linspace(-10, 10, 100)
                            y_vals = (c_eq - a_eq * x_vals) / b_val
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=y_vals,
                                mode='lines',
                                name=f"Equation {i+1}: {a_eq}x + {b_val}y = {c_eq}"
                            ))
                        elif abs(a_eq) > 1e-10:
                            # b_val is zero, so we have a vertical line x = c/a
                            x_val = c_eq / a_eq
                            y_vals = np.linspace(-10, 10, 100)
                            x_vals_const = np.ones_like(y_vals) * x_val # Renamed
                            fig.add_trace(go.Scatter(
                                x=x_vals_const, y=y_vals,
                                mode='lines',
                                name=f"Equation {i+1}: {a_eq}x = {c_eq}"
                            ))
                    
                    # Add the solution point
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name=f"Solution: ({x:.4f}, {y:.4f})"
                    ))
                    
                    # Configure the layout
                    fig.update_layout(
                        title="2D System Solution",
                        xaxis_title="x",
                        yaxis_title="y",
                        showlegend=True,
                        width=700,
                        height=500,
                    )
                    
                    st.plotly_chart(fig)
                
                elif len(result) == 3:
                    # 3D solution
                    st.info("3D system visualization is not implemented yet.")
            
            elif result is not None and isinstance(result, dict) and "particular" in result and "null_space" in result:
                # This is a system with a particular solution and null space
                if len(result["particular"]) == 2:
                    # 2D system with infinite solutions (a line)
                    particular = result["particular"]
                    null_space = result["null_space"].flatten()
                    
                    # Create a 2D plot showing the solution line
                    fig = go.Figure()
                    
                    # Plot the solution line
                    t_vals_line = np.linspace(-5, 5, 100) # Renamed
                    x_vals_line = particular[0] + t_vals_line * null_space[0]
                    y_vals_line = particular[1] + t_vals_line * null_space[1]
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals_line, y=y_vals_line,
                        mode='lines',
                        name="Solution Line",
                        line=dict(color='red', width=3)
                    ))
                    
                    # Add the particular solution point
                    fig.add_trace(go.Scatter(
                        x=[particular[0]], y=[particular[1]],
                        mode='markers',
                        marker=dict(size=12, color='blue'),
                        name=f"Particular Solution: ({particular[0]:.4f}, {particular[1]:.4f})"
                    ))
                    
                    # For each equation, plot the line
                    for i in range(A_gauss.shape[0]):
                        a_eq_inf, b_val_inf = A_gauss[i, 0], A_gauss[i, 1] # Renamed
                        c_eq_inf = b_gauss[i] # Renamed
                        
                        if abs(b_val_inf) > 1e-10:
                            # b_val is not zero, so we can plot y = (c - a*x) / b_val
                            x_vals_eq = np.linspace(-10, 10, 100)
                            y_vals_eq = (c_eq_inf - a_eq_inf * x_vals_eq) / b_val_inf
                            fig.add_trace(go.Scatter(
                                x=x_vals_eq, y=y_vals_eq,
                                mode='lines',
                                name=f"Equation {i+1}: {a_eq_inf}x + {b_val_inf}y = {c_eq_inf}"
                            ))
                        elif abs(a_eq_inf) > 1e-10:
                            # b_val is zero, so we have a vertical line x = c/a
                            x_val_eq_vert = c_eq_inf / a_eq_inf # Renamed
                            y_vals_eq_vert = np.linspace(-10, 10, 100)
                            x_vals_eq_const = np.ones_like(y_vals_eq_vert) * x_val_eq_vert # Renamed
                            fig.add_trace(go.Scatter(
                                x=x_vals_eq_const, y=y_vals_eq_vert,
                                mode='lines',
                                name=f"Equation {i+1}: {a_eq_inf}x = {c_eq_inf}"
                            ))
                    
                    # Configure the layout
                    fig.update_layout(
                        title="2D System Solution (Infinite Solutions)",
                        xaxis_title="x",
                        yaxis_title="y",
                        showlegend=True,
                        width=700,
                        height=500,
                    )
                    
                    st.plotly_chart(fig)
                
                elif len(result["particular"]) == 3:
                    # 3D system with infinite solutions
                    st.info("3D system visualization is not implemented yet.")
        
        
        return result 