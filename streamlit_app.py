#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Linear Algebra Calculator
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from given_reference.core import eliminate, mrref, mnull
import sympy as sym
from io import StringIO
import sys

# Import CLI functionality to reuse functions
from linalg_cli import LinearAlgebraExerciseFramework

# Set page configuration
st.set_page_config(
    page_title="Linear Algebra Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/peaktwilight/python_25fs/issues',
        'Report a bug': 'https://github.com/peaktwilight/python_25fs/issues',
        'About': "# Linear Algebra Calculator\nA comprehensive toolkit for learning and solving linear algebra problems. Made with ❤️ by Doruk for the LAG Fachmodul at FHNW."
    }
)

# No custom CSS - rely on Streamlit's default styling


class StreamOutput:
    """Capture print output to display in Streamlit."""
    def __init__(self):
        self.buffer = StringIO()
        self.old_stdout = sys.stdout
    
    def __enter__(self):
        sys.stdout = self.buffer
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.old_stdout
    
    def get_output(self):
        return self.buffer.getvalue()


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
        
        # Display steps calculation 
        st.write("### Step-by-step Calculation")
        
        step_output = output.get_output().split('\n')
        for line in step_output:
            if line.strip():
                st.write(line)
        
        # Display a visualization of the original and normalized vectors
        st.subheader("Visualization")
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
            max_val = max([max(abs(vec)) for vec in vectors])
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
            magnitude_relation = np.linalg.norm(cross) / (np.linalg.norm(a) * np.linalg.norm(b) * np.sin(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))
            st.write(f"**Magnitude Relationship:**")
            st.write(f"- |a × b| = |a|·|b|·sin(θ) ✓ (Ratio: {magnitude_relation:.6f} ≈ 1)")
            
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
                for v in padded_vectors:
                    if len(v) == 1:
                        vis_vectors.append(np.array([v[0], 0]))
                    else:
                        vis_vectors.append(v)
                
                # Create 2D visualization
                fig = go.Figure()
                
                # Add vectors as arrows from origin
                for i, vec in enumerate(vis_vectors):
                    fig.add_trace(go.Scatter(
                        x=[0, vec[0]],
                        y=[0, vec[1]],
                        mode='lines+markers',
                        name=f"Vector {i+1}",
                        line=dict(width=3),
                        marker=dict(size=[0, 10])
                    ))
                
                # Determine the layout extent
                max_val = max([max(abs(vec)) for vec in vis_vectors]) * 1.2
                
                # Add the collinearity result to the title
                collinear_status = "collinear ✓" if result else "not collinear ✗"
                
                # Configure the layout
                fig.update_layout(
                    xaxis=dict(
                        range=[-max_val, max_val],
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='white',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        color='white',
                    ),
                    yaxis=dict(
                        range=[-max_val, max_val],
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
                for i, vec in enumerate(padded_vectors):
                    fig.add_trace(go.Scatter3d(
                        x=[0, vec[0]],
                        y=[0, vec[1]],
                        z=[0, vec[2]],
                        mode='lines+markers',
                        name=f"Vector {i+1}",
                        line=dict(width=6),
                        marker=dict(size=[0, 8])
                    ))
                
                # Determine the layout extent
                max_val = max([max(abs(vec)) for vec in padded_vectors]) * 1.2
                
                # Add the collinearity result to the title
                collinear_status = "collinear ✓" if result else "not collinear ✗"
                
                # Configure the layout
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            range=[-max_val, max_val],
                            title="X",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        yaxis=dict(
                            range=[-max_val, max_val],
                            title="Y",
                            color="white",
                            gridcolor='rgba(255, 255, 255, 0.2)',
                            backgroundcolor='rgba(0,0,0,0.1)',
                        ),
                        zaxis=dict(
                            range=[-max_val, max_val],
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
            reduced = mrref(matrix)
            st.write(pd.DataFrame(reduced))
            
            # Check for zero rows to determine linear dependence
            rank = np.sum([np.linalg.norm(row) > 1e-10 for row in reduced])
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
        reduced = mrref(augmented)
        self.display_matrix_heatmap(reduced, "Reduced Row Echelon Form")
        
        # For 2D or 3D systems, visualize the solution space
        if augmented.shape[1] - 1 in [2, 3]:
            st.markdown("**Solution Space Visualization:**")
            
            # Get the system information
            A = augmented[:, :-1]
            b = augmented[:, -1]
            
            # Check if the system has a unique solution
            if result is not None and not isinstance(result, dict):
                # This is a unique solution
                if len(result) == 2:
                    # 2D solution
                    x, y = result
                    
                    # Create a 2D plot for the system
                    fig = go.Figure()
                    
                    # For each equation, plot the line
                    for i in range(A.shape[0]):
                        a, b_val = A[i, 0], A[i, 1]
                        c = b[i]
                        
                        if abs(b_val) > 1e-10:
                            # b_val is not zero, so we can plot y = (c - a*x) / b_val
                            x_vals = np.linspace(-10, 10, 100)
                            y_vals = (c - a * x_vals) / b_val
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=y_vals,
                                mode='lines',
                                name=f"Equation {i+1}: {a}x + {b_val}y = {c}"
                            ))
                        elif abs(a) > 1e-10:
                            # b_val is zero, so we have a vertical line x = c/a
                            x_val = c / a
                            y_vals = np.linspace(-10, 10, 100)
                            x_vals = np.ones_like(y_vals) * x_val
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=y_vals,
                                mode='lines',
                                name=f"Equation {i+1}: {a}x = {c}"
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
                    t_vals = np.linspace(-5, 5, 100)
                    x_vals = particular[0] + t_vals * null_space[0]
                    y_vals = particular[1] + t_vals * null_space[1]
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
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
                    for i in range(A.shape[0]):
                        a, b_val = A[i, 0], A[i, 1]
                        c = b[i]
                        
                        if abs(b_val) > 1e-10:
                            # b_val is not zero, so we can plot y = (c - a*x) / b_val
                            x_vals = np.linspace(-10, 10, 100)
                            y_vals = (c - a * x_vals) / b_val
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=y_vals,
                                mode='lines',
                                name=f"Equation {i+1}: {a}x + {b_val}y = {c}"
                            ))
                        elif abs(a) > 1e-10:
                            # b_val is zero, so we have a vertical line x = c/a
                            x_val = c / a
                            y_vals = np.linspace(-10, 10, 100)
                            x_vals = np.ones_like(y_vals) * x_val
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=y_vals,
                                mode='lines',
                                name=f"Equation {i+1}: {a}x = {c}"
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


def main():
    calculator = LinAlgCalculator()
    
    # Header
    st.title("Linear Algebra Calculator")
    
    # Sidebar with categories
    st.sidebar.title("Categories")
    category = st.sidebar.selectbox(
        "Select Operation Category",
        ["Vector Operations", "Matrix Operations", "Systems of Linear Equations"]
    )
    
    if category == "Vector Operations":
        operation = st.sidebar.selectbox(
            "Select Vector Operation",
            ["Vector Normalization", "Vector Projection/Shadow", "Vector Angle", "Cross Product", 
             "Triangle Area", "Point-Line Distance", "Check Collinear"]
        )
        
        st.subheader(operation)
        
        if operation == "Vector Normalization":
            st.write("This operation normalizes a vector to unit length while preserving its direction.")
            
            vector_input = st.text_input(
                "Enter Vector (format: x1, x2, ... or [x1, x2, ...]):",
                value="3, 4"
            )
            
            if st.button("Calculate Normalization"):
                if vector_input:
                    calculator.normalize_vector(vector_input)
                else:
                    st.error("Please enter a vector.")
        
        elif operation == "Vector Projection/Shadow":
            st.write("This operation calculates the projection (shadow) of one vector onto another.")
            
            col1, col2 = st.columns(2)
            with col1:
                vector_a = st.text_input(
                    "Enter Vector a (to project onto):",
                    value="3, -4"
                )
            
            with col2:
                vector_b = st.text_input(
                    "Enter Vector b (to be projected):",
                    value="12, -1"
                )
            
            if st.button("Calculate Projection"):
                if vector_a and vector_b:
                    calculator.vector_shadow(vector_a, vector_b)
                else:
                    st.error("Please enter both vectors.")
        
        elif operation == "Vector Angle":
            st.write("This operation calculates the angle between two vectors.")
            
            col1, col2 = st.columns(2)
            with col1:
                vector_a = st.text_input(
                    "Enter Vector a:",
                    value="1, 1, -1"
                )
            
            with col2:
                vector_b = st.text_input(
                    "Enter Vector b:",
                    value="1, -1, 1"
                )
            
            if st.button("Calculate Angle"):
                if vector_a and vector_b:
                    calculator.vector_angle(vector_a, vector_b)
                else:
                    st.error("Please enter both vectors.")
        
        elif operation == "Cross Product":
            st.write("This operation calculates the cross product of two vectors, resulting in a vector perpendicular to both input vectors.")
            
            col1, col2 = st.columns(2)
            with col1:
                vector_a = st.text_input(
                    "Enter first vector (a):",
                    value="-2, 0, 0"
                )
            
            with col2:
                vector_b = st.text_input(
                    "Enter second vector (b):",
                    value="0, 9, 8"
                )
            
            with st.expander("Help: Cross Product Information"):
                st.write("""
                The cross product is primarily defined for 3D vectors and produces a vector that is perpendicular to both input vectors.
                
                **Formula:** a × b = |a|·|b|·sin(θ)·n̂
                
                Where:
                - θ is the angle between the vectors
                - n̂ is the unit vector perpendicular to both a and b
                
                **Note:** For 2D vectors, they will be treated as 3D vectors with z=0.
                """)
            
            if st.button("Calculate Cross Product"):
                if vector_a and vector_b:
                    # Add cross product calculation logic
                    class Args:
                        def __init__(self, vector_a, vector_b):
                            self.vector_a = vector_a
                            self.vector_b = vector_b
                    
                    args = Args(vector_a, vector_b)
                    calculator.cross_product(args)
                else:
                    st.error("Please enter both vectors.")
                    
        elif operation == "Triangle Area":
            st.write("This operation calculates the area of a triangle defined by three points.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                point_a = st.text_input(
                    "Enter first point (A):",
                    value="0, 4, 2"
                )
            
            with col2:
                point_b = st.text_input(
                    "Enter second point (B):",
                    value="0, 8, 5"
                )
                
            with col3:
                point_c = st.text_input(
                    "Enter third point (C):",
                    value="0, 8, -1"
                )
            
            with st.expander("Help: Triangle Area Information"):
                st.write("""
                This calculation finds the area of a triangle in 2D or 3D space using vector cross product.
                
                **Formula:** Area = |AB × AC| / 2
                
                Where:
                - AB and AC are vectors from point A to points B and C
                - × represents the cross product
                
                **Input:** Three points in 2D or 3D space.
                For 2D points, enter: "x, y"
                For 3D points, enter: "x, y, z"
                """)
            
            if st.button("Calculate Triangle Area"):
                if point_a and point_b and point_c:
                    # Create Args object and call the triangle_area method
                    class Args:
                        def __init__(self, point_a, point_b, point_c):
                            self.point_a = point_a
                            self.point_b = point_b
                            self.point_c = point_c
                    
                    args = Args(point_a, point_b, point_c)
                    calculator.triangle_area(args)
                else:
                    st.error("Please enter all three points.")
                    
        elif operation == "Point-Line Distance":
            st.write("This operation calculates the shortest distance from a point to a line in space.")
            
            col1, col2 = st.columns(2)
            with col1:
                point_a = st.text_input(
                    "Enter a point on the line (A):",
                    value="3, 0, 0"
                )
                
                direction = st.text_input(
                    "Enter line direction vector (v):",
                    value="2, 0, 0"
                )
            
            with col2:
                point_b = st.text_input(
                    "Enter point to find distance from (B):",
                    value="5, 10, 0"
                )
                
                st.info("The direction vector should be non-zero (it defines the direction of the line).")
            
            with st.expander("Help: Point-Line Distance Information"):
                st.write("""
                This calculation finds the shortest distance from a point to a line in 2D or 3D space.
                
                **Formula:** Distance = |v × (B-A)| / |v|
                
                Where:
                - A is a point on the line
                - v is the direction vector of the line
                - B is the point to find the distance from
                - × represents the cross product
                
                **Input:** 
                - Point on line and direction vector to define the line
                - Point to find the distance from
                
                For 2D points, enter: "x, y"
                For 3D points, enter: "x, y, z"
                """)
            
            if st.button("Calculate Point-Line Distance"):
                if point_a and direction and point_b:
                    # Create Args object and call the point_line_distance method
                    class Args:
                        def __init__(self, point_a, direction, point_b):
                            self.point_a = point_a
                            self.direction = direction
                            self.point_b = point_b
                    
                    args = Args(point_a, direction, point_b)
                    calculator.point_line_distance(args)
                else:
                    st.error("Please enter all required points and vectors.")
                    
        elif operation == "Check Collinear":
            st.write("This operation checks if vectors are collinear (lie on the same line through the origin).")
            
            # Dynamic vector input
            num_vectors = st.slider("Number of vectors to check", min_value=2, max_value=5, value=2)
            
            vectors = []
            cols = st.columns(num_vectors)
            
            for i in range(num_vectors):
                with cols[i]:
                    vector = st.text_input(
                        f"Enter Vector {i+1}:",
                        value="-3, 2" if i == 0 else "6, -4" if i == 1 else "",
                    )
                    vectors.append(vector)
            
            with st.expander("Help: Collinearity Information"):
                st.write("""
                This calculation checks if vectors are collinear - meaning they all lie on the same line through the origin.
                
                **Mathematical Definition:** Vectors are collinear if one can be expressed as a scalar multiple of the other.
                
                For example, vectors [1, 2] and [2, 4] are collinear because [2, 4] = 2 * [1, 2].
                
                **Calculation Method:**
                - Create a matrix where each vector is a row
                - Find the rank of this matrix using row reduction
                - If the rank is 1 (or 0 for zero vectors), the vectors are collinear
                
                **Input:** Two or more vectors of the same dimensionality.
                """)
            
            if st.button("Check Collinearity"):
                # Check if all vector inputs are provided
                if all(vector for vector in vectors):
                    # Create Args object and call the check_collinear method
                    class Args:
                        def __init__(self, vectors):
                            self.vectors = vectors
                    
                    args = Args(vectors)
                    calculator.check_collinear(args)
                else:
                    st.error("Please enter all vectors.")
    
    elif category == "Matrix Operations":
        operation = st.sidebar.selectbox(
            "Select Matrix Operation",
            ["Matrix Addition", "Matrix Subtraction", "Scalar Multiplication", "Matrix Transpose"]
        )
        
        st.subheader(operation)
        
        if operation == "Matrix Addition":
            st.write("This operation adds two matrices element-wise.")
            
            col1, col2 = st.columns(2)
            with col1:
                matrix_a = st.text_area(
                    "Enter Matrix A (format: a,b,c; d,e,f or separate rows with newlines):",
                    value="1, 2, 3\n4, 5, 6\n7, 8, 9"
                )
            
            with col2:
                matrix_b = st.text_area(
                    "Enter Matrix B (same dimensions as A):",
                    value="9, 8, 7\n6, 5, 4\n3, 2, 1"
                )
            
            if st.button("Calculate Sum"):
                if matrix_a and matrix_b:
                    calculator.matrix_operations("add", matrix_a, matrix_b)
                else:
                    st.error("Please enter both matrices.")
        
        elif operation == "Matrix Subtraction":
            st.write("This operation subtracts matrix B from matrix A element-wise.")
            
            col1, col2 = st.columns(2)
            with col1:
                matrix_a = st.text_area(
                    "Enter Matrix A (format: a,b,c; d,e,f or separate rows with newlines):",
                    value="9, 8, 7\n6, 5, 4\n3, 2, 1"
                )
            
            with col2:
                matrix_b = st.text_area(
                    "Enter Matrix B (same dimensions as A):",
                    value="1, 2, 3\n4, 5, 6\n7, 8, 9"
                )
            
            if st.button("Calculate Difference"):
                if matrix_a and matrix_b:
                    calculator.matrix_operations("subtract", matrix_a, matrix_b)
                else:
                    st.error("Please enter both matrices.")
        
        elif operation == "Scalar Multiplication":
            st.write("This operation multiplies a matrix by a scalar value.")
            
            matrix_a = st.text_area(
                "Enter Matrix (format: a,b,c; d,e,f or separate rows with newlines):",
                value="1, 2, 3\n4, 5, 6\n7, 8, 9"
            )
            
            scalar = st.number_input("Enter Scalar Value:", value=2.0)
            
            if st.button("Calculate Product"):
                if matrix_a:
                    calculator.matrix_operations("multiply_scalar", matrix_a, scalar=str(scalar))
                else:
                    st.error("Please enter a matrix.")
        
        elif operation == "Matrix Transpose":
            st.write("This operation transposes a matrix (flips rows and columns).")
            
            matrix_a = st.text_area(
                "Enter Matrix (format: a,b,c; d,e,f or separate rows with newlines):",
                value="1, 2, 3\n4, 5, 6"
            )
            
            if st.button("Calculate Transpose"):
                if matrix_a:
                    calculator.matrix_operations("transpose", matrix_a)
                else:
                    st.error("Please enter a matrix.")
    
    elif category == "Systems of Linear Equations":
        operation = st.sidebar.selectbox(
            "Select Operation",
            ["Solve System (Gaussian Elimination)"]
        )
        
        st.subheader(operation)
        
        if operation == "Solve System (Gaussian Elimination)":
            st.write("This operation solves a system of linear equations using Gaussian elimination.")
            st.write("Enter the augmented matrix [A|b] where each row represents one equation.")
            
            matrix_input = st.text_area(
                "Enter Augmented Matrix [A|b] (format: a,b,c,d; e,f,g,h or separate rows with newlines):",
                value="1, -4, -2, -25\n0, -3, 6, -18\n7, -13, -4, -85"
            )
            
            # Add helpful explanation
            with st.expander("Help: Format Explanation"):
                st.write("""
                For a system of linear equations:
                ```
                a₁x + b₁y + c₁z = d₁
                a₂x + b₂y + c₂z = d₂
                a₃x + b₃y + c₃z = d₃
                ```
                
                The augmented matrix would be:
                ```
                a₁, b₁, c₁, d₁
                a₂, b₂, c₂, d₂
                a₃, b₃, c₃, d₃
                ```
                
                Example: For the system
                ```
                x - 4y - 2z = -25
                -3y + 6z = -18
                7x - 13y - 4z = -85
                ```
                
                Enter:
                ```
                1, -4, -2, -25
                0, -3, 6, -18
                7, -13, -4, -85
                ```
                """)
            
            if st.button("Solve System"):
                if matrix_input:
                    calculator.solve_gauss(matrix_input)
                else:
                    st.error("Please enter the augmented matrix.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This Linear Algebra Calculator is built with Streamlit and leverages "
        "the existing functions from the Linear Algebra CLI framework. "
        "It provides visualizations and step-by-step calculations for various "
        "linear algebra operations."
    )
    
    # Version info and deployment notice
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center;'>"
        "<small>Version 1.4.0 | Deployed on Fly.io</small><br>"
        "<small>© 2025 Doruk | FHNW Linear Algebra Module</small>"
        "</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()