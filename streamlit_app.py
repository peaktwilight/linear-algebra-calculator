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
            ["Vector Normalization", "Vector Projection/Shadow", "Vector Angle"]
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


if __name__ == "__main__":
    main()