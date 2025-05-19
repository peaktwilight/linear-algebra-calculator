#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Linear Algebra Calculator
"""

import streamlit as st
# import numpy as np # No longer directly used here
# import sympy as sym # No longer directly used here
# import random # No longer directly used here
# import json # No longer directly used here

# Import CLI functionality to reuse functions
# from linalg_cli import LinearAlgebraExerciseFramework # No longer directly used here

# Import Quiz generator
# from linear_algebra_quiz import LinearAlgebraQuiz # No longer directly used here

# Import Utilities and Components from st_components package
from st_components.st_utils import StreamOutput # StreamOutput is not used directly in streamlit_app.py, but LinAlgCalculator needs it.
from st_components.st_calculator_operations import LinAlgCalculator
from st_components.st_quiz_ui import QuizComponent
from st_components.st_learning_resources import render_learning_resources

# Set page configuration
st.set_page_config(
    page_title="Linear Algebra Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/peaktwilight/linear-algebra-calculator/issues',
        'Report a bug': 'https://github.com/peaktwilight/linear-algebra-calculator/issues',
        'About': "# Linear Algebra Calculator\nA comprehensive toolkit for learning and solving linear algebra problems. Made with ❤️ by Doruk for the LAG Fachmodul at FHNW.\n\nVersion 1.4.0 | Open source at https://github.com/peaktwilight/linear-algebra-calculator"
    }
)

# No custom CSS - rely on Streamlit's default styling


def main():
    calculator = LinAlgCalculator()
    quiz_component = QuizComponent()
    
    # Simple header
    st.title("Linear Algebra Calculator")
    
    # Sidebar with categories
    st.sidebar.title("Categories")
    category = st.sidebar.selectbox(
        "Select Operation Category",
        ["Vector Operations", "Matrix Operations", "Systems of Linear Equations", "Quiz Mode", "Learning Resources"]
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
    
    elif category == "Quiz Mode":
        # Render the quiz component
        quiz_component.render()
    
    elif category == "Learning Resources":
        render_learning_resources()
    
    # How to use this calculator section
    st.sidebar.markdown("---") # Separator

    with st.sidebar.expander("Quick Start Guide", expanded=False):
        st.markdown(
            """
            - Select an operation category from the sidebar.
            - Choose a specific operation.
            - Enter your values in the input fields.
            - Click the calculate button.
            - View the step-by-step solution and visual representation.
            """
        )

    st.sidebar.markdown("##### Tips") # Using H5 for "Tips"
    st.sidebar.markdown(
        """
        - Hover over visualizations for more details.
        - Expand the "Help" sections for formula explanations.
        - For vector/matrix inputs, use format: "x1, x2, ..." or separate rows with newlines.
        """
    )

    st.sidebar.caption("Made with ❤️ by Doruk")
    st.sidebar.caption("Version 1.4 | FHNW Linear Algebra Module")
    st.sidebar.caption("Open source at [github.com/peaktwilight/linear-algebra-calculator](https://github.com/peaktwilight/linear-algebra-calculator)")


if __name__ == "__main__":
    main()
