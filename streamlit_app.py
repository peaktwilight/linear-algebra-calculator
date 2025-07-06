#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Linear Algebra Calculator
"""

import streamlit as st
import numpy as np
from st_components.st_utils import StreamOutput # StreamOutput is not used directly in streamlit_app.py, but LinAlgCalculator needs it.
from st_components.st_calculator_operations import LinAlgCalculator
from st_components.st_learning_resources import render_learning_resources_page
from st_components.st_linearity_operations import LinearityOperations
from st_components.st_batch_calculator import BatchMatrixCalculator
from st_components.st_summation_calculator import SummationCalculator
from st_components.st_complex_operations import ComplexOperations
from st_components.st_determinant_operations import DeterminantOperations
from st_components.st_cramers_rule import CramersRule
from st_components.st_math_utils import MathUtils

# Set page configuration
st.set_page_config(
    page_title="Linear Algebra Calculator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS is now loaded from the .streamlit/custom.css file


def apply_styling():
    """Apply styling with reliable animations"""
    # Define simple animations directly in the code to ensure availability
    st.markdown("""
    <style>
    /* Critical animations */
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    @keyframes slideInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
      from { opacity: 0; transform: translateX(-20px); }
      to { opacity: 1; transform: translateX(0); }
    }
    
    /* Core animation classes */
    .animate-title {
      animation: slideInUp 0.8s ease-out forwards;
      opacity: 0;
      color: #ffffff;
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 1.5rem;
      text-align: center;
    }
    
    .animate-subheader {
      animation: slideInLeft 0.6s ease-out forwards;
      opacity: 0;
      color: #f0f2f6;
      font-size: 1.8rem;
      font-weight: 600;
      margin: 1.5rem 0 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
      display: inline-block !important;
      background-color: rgba(55, 55, 60, 0.7) !important;
      color: white !important;
      border: 1px solid rgba(255, 255, 255, 0.2) !important;
      border-radius: 4px !important;
      padding: 0.3rem 1rem !important;
      margin: 0.5rem 0 !important;
      visibility: visible !important;
      /* opacity controlled by animation */
    }
    
    /* Basic element animations */
    h1, h2, h3, h4 {
      animation: fadeIn 0.5s ease-out forwards;
      opacity: 0;
    }
    
    p {
      animation: fadeIn 0.5s ease-out forwards;
      opacity: 0;
      animation-delay: 0.1s;
    }
    
    /* Alert styling to ensure they're visible */
    .stAlert, div[data-testid="stAlert"], [data-testid="stAlertContainer"] {
      opacity: 1 !important;
    }
    
    /* Staggered waterfall animations */
    .animate-title { animation-delay: 0.1s; }
    .animate-subheader { animation-delay: 0.35s; }
    h1 { animation-delay: 0.1s; }
    h2 { animation-delay: 0.35s; }
    h3 { animation-delay: 0.45s; }
    h4 { animation-delay: 0.55s; }
    p { animation-delay: 0.65s; }
    .stButton > button { animation: fadeIn 0.4s ease-out forwards; animation-delay: 0.75s; opacity: 0; }
    
    /* Make animations complete */
    h1, h2, h3, h4, p, .animate-title, .animate-subheader, .stButton > button {
      animation-fill-mode: forwards;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Then load the full CSS file - make sure this runs last to override any other styles
    try:
        with open('.streamlit/custom.css', 'r') as f:
            custom_css = f.read()
                    # Add scripts to ensure animations work correctly
            enhanced_css = custom_css + """
            <script>
            // Script to ensure animations have fully loaded
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(function() {
                    document.body.classList.add('animations-ready');
                    
                    // Fix alerts visibility
                    const fixAlerts = function() {
                        document.querySelectorAll('.stAlert, [data-testid="stAlert"], [data-testid="stAlertContainer"]')
                            .forEach(function(el) {
                                el.style.opacity = '1';
                                el.style.visibility = 'visible';
                            });
                    };
                    
                    // Enhance title rainbow wave effect
                    const enhanceTitleEffect = function() {
                        const title = document.querySelector('.animate-title');
                        if (title) {
                            // Set initial state
                            title.style.opacity = "0"; // Start invisible
                            
                            // Force a repaint to ensure animation runs smoothly
                            setTimeout(() => {
                                title.style.opacity = "1"; // Make visible with animation
                                console.log("Rainbow wave animation activated");
                            }, 50);
                        }
                    };
                    
                    // Run initial fixes and enhancements
                    fixAlerts();
                    enhanceTitleEffect();
                    
                    // Keep checking for new alerts
                    setInterval(fixAlerts, 200);
                }, 100);
            });
            </script>
            """
            st.markdown(f'<style>{enhanced_css}</style>', unsafe_allow_html=True)
    except Exception:
        # Silently continue with the basic animations defined above
        pass


def main():
    calculator = LinAlgCalculator()
    linearity_checker = LinearityOperations()
    batch_calculator = BatchMatrixCalculator()
    summation_calculator = SummationCalculator()
    complex_calculator = ComplexOperations()
    determinant_calculator = DeterminantOperations()
    cramers_rule_calculator = CramersRule()
    
    # Apply minimal styling approach 
    apply_styling()
    
    # Animated header with GitHub tag 
    st.markdown('''
    <div class="title-container">
        <h1 class="animate-title">Linear Algebra Calculator</h1>
    </div>
    <div class="github-tag-container">
        <a href="https://github.com/peaktwilight/linear-algebra-calculator" target="_blank" class="github-version-tag">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style="margin-right: 6px;" stroke="#f0f2f6" stroke-width="2">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
            </svg>
            <span class="rainbow-text">GitHub v1.9.4</span>
        </a>
    </div>
    ''', unsafe_allow_html=True)
    
    # Define all available operations for search
    all_operations = {
        "Vector Operations": ["Vector Normalization", "Vector Projection/Shadow", "Vector Angle", "Cross Product", 
                             "Triangle Area", "Point-Line Distance", "Check Collinear"],
        "Lines and Planes": ["Plane from 3 Points", "Point-to-Plane Distance", "Line-Plane Intersection", "Plane Equation"],
        "Matrix Operations": ["Batch Expression Calculator", "Matrix Addition", "Matrix Subtraction", "Scalar Multiplication", "Matrix Transpose", 
                             "Matrix Multiplication", "Matrix Determinant", "Matrix Inverse", "Gauss-Jordan System Solver", "Special Matrices", 
                             "Eigenvalues and Eigenvectors"],
        "Determinants": ["2×2 Determinant & Parallelogram Area", "3×3 Determinant & Sarrus Rule", 
                        "General Determinant"],
        "Systems of Linear Equations": ["Solve System (Gaussian Elimination)", "Cramer's Rule", "Standard Form Analysis", 
                                       "Row Operations Analysis", "Free Parameter Analysis", 
                                       "Homogeneous/Inhomogeneous Solutions", "Check Vector Solutions",
                                       "Calculate Null Space (Basis)", "Geometric Interpretation"],
        "Linear Mappings": ["Check Linearity", "Matrix Representation", "Polynomial Mappings", "Trigonometric Mappings", 
                           "Dot Product Mappings", "Quadratic Forms", "Matrix-Vector Multiplication"],
        "Complex Numbers": ["Addition", "Multiplication", "Division", "Conjugate", "Polar Form", 
                           "Polar Arithmetic", "Powers & Roots", "Polynomial Equations",
                           "Gaussian Plane Visualization"],
        "Series & Summations": ["General Summation", "Geometric Series", "Arithmetic Series", "Pattern Recognition"]
    }
    
    # Flatten list for searching
    all_operations_flat = [(cat, op) for cat in all_operations for op in all_operations[cat]]
    
    # Sidebar with search and categories
    st.sidebar.title("Linear Algebra Tools")
    
    # Search box in sidebar
    search_query = st.sidebar.text_input("🔍 Search operations...", 
                                       placeholder="E.g., matrix multiply, eigenvalues...")
    
    # Process search if entered
    if search_query:
        matches = []
        for category_name, operation_name in all_operations_flat:
            if (search_query.lower() in operation_name.lower() or 
                search_query.lower() in category_name.lower()):
                matches.append((category_name, operation_name))
        
        if matches:
            st.sidebar.success(f"Found {len(matches)} matching operations")
            search_category, search_operation = st.sidebar.selectbox(
                "Select an operation:",
                matches,
                format_func=lambda x: f"{x[1]} (in {x[0]})"
            )
            # Auto-select this category and operation
            category = search_category
            # We'll use this later to auto-select the operation
            selected_operation_from_search = search_operation
        else:
            st.sidebar.warning("No matches found. Try different keywords.")
            selected_operation_from_search = None
            st.sidebar.markdown("---")
            category = st.sidebar.selectbox(
                "Select Operation Category",
                ["Vector Operations", "Lines and Planes", "Matrix Operations", "Determinants", "Systems of Linear Equations", "Linear Mappings", "Complex Numbers", "Series & Summations", "Learning Resources"]
            )
    else:
        selected_operation_from_search = None
        st.sidebar.markdown("---")
        # Default categories selection if no search
        category = st.sidebar.selectbox(
            "Select Operation Category",
            ["Vector Operations", "Lines and Planes", "Matrix Operations", "Determinants", "Systems of Linear Equations", "Linear Mappings", "Complex Numbers", "Series & Summations", "Learning Resources"]
        )
    
    if category == "Vector Operations":
        vector_operations = ["Vector Normalization", "Vector Projection/Shadow", "Vector Angle", "Cross Product", 
                           "Triangle Area", "Point-Line Distance", "Check Collinear"]
        
        # Auto-select operation if it came from search
        default_index = 0
        if selected_operation_from_search:
            # Find the closest matching operation
            for i, op in enumerate(vector_operations):
                if selected_operation_from_search in op or op in selected_operation_from_search:
                    default_index = i
                    break
                    
        operation = st.sidebar.selectbox(
            "Select Vector Operation",
            vector_operations,
            index=default_index
        )
        
        st.markdown(f'<h2 class="animate-subheader">{operation}</h2>', unsafe_allow_html=True)
        
        if operation == "Vector Normalization":
            st.write("This operation normalizes a vector to unit length while preserving its direction.")
            
            vector_input = st.text_input(
                "Enter Vector (format: x1, x2, ... or [x1, x2, ...]):",
                value="3, 4"
            )
            
            # Regular Streamlit button with a key
            if st.button("Calculate Normalization", key="norm_button"):
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
                    value="4, 1"
                )
            
            with col2:
                vector_b = st.text_input(
                    "Enter Vector b (to be projected):",
                    value="1, 8"
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
    
    elif category == "Lines and Planes":
        plane_operations = ["Plane from 3 Points", "Point-to-Plane Distance", "Line-Plane Intersection", "Plane Equation"]
        
        # Auto-select operation if it came from search
        default_index = 0
        if selected_operation_from_search:
            for i, op in enumerate(plane_operations):
                if selected_operation_from_search in op or op in selected_operation_from_search:
                    default_index = i
                    break
        
        operation = st.selectbox(
            "Select Operation:",
            plane_operations,
            index=default_index
        )
        
        if operation == "Plane from 3 Points":
            st.write("This operation calculates the plane equation from three non-collinear points.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                point_A = st.text_input(
                    "Enter Point A (x, y, z):",
                    value="3, 1, -2",
                    help="Enter coordinates separated by commas"
                )
            with col2:
                point_B = st.text_input(
                    "Enter Point B (x, y, z):",
                    value="4, 1, -1",
                    help="Enter coordinates separated by commas"
                )
            with col3:
                point_C = st.text_input(
                    "Enter Point C (x, y, z):",
                    value="-4, 5, -2",
                    help="Enter coordinates separated by commas"
                )
            
            with st.expander("Help: Plane from 3 Points Information"):
                st.write("""
                This calculation finds the plane equation passing through three given points.
                
                **Method:**
                1. Calculate two vectors in the plane: AB = B - A and AC = C - A
                2. Find the normal vector: n = AB × AC (cross product)
                3. Use point A to find the constant: k = n · A
                4. The plane equation is: n₁x + n₂y + n₃z = k
                
                **Requirements:**
                - Points must be 3-dimensional (x, y, z)
                - Points must not be collinear (not all on the same line)
                """)
            
            if st.button("Calculate Plane Equation"):
                if point_A and point_B and point_C:
                    calculator.plane_from_three_points(point_A, point_B, point_C)
                else:
                    st.error("Please enter all three points.")
        
        elif operation == "Point-to-Plane Distance":
            st.write("This operation calculates the shortest distance from a point to a plane.")
            
            col1, col2 = st.columns(2)
            with col1:
                point_D = st.text_input(
                    "Enter Point (x, y, z):",
                    value="-3, 21, -10",
                    help="Point to find distance from"
                )
                plane_normal = st.text_input(
                    "Enter Plane Normal Vector (nx, ny, nz):",
                    value="8, -4, 0",
                    help="Normal vector to the plane"
                )
            with col2:
                plane_constant = st.text_input(
                    "Enter Plane Constant (k):",
                    value="18",
                    help="Constant in plane equation nx + ny + nz = k"
                )
                
                st.markdown("**Alternative: Use Plane from 3 Points**")
                st.info("💡 Tip: First use 'Plane from 3 Points' to get the normal vector and constant, then use those values here!")
            
            with st.expander("Help: Point-to-Plane Distance Information"):
                st.write("""
                This calculation finds the shortest distance from a point to a plane.
                
                **Formula:** d = |n · p - k| / ||n||
                
                Where:
                - n is the normal vector to the plane
                - p is the point
                - k is the constant in the plane equation nx + ny + nz = k
                - ||n|| is the magnitude of the normal vector
                
                **Input Format:**
                - Point: x, y, z coordinates
                - Normal vector: nx, ny, nz components
                - Constant: single number k
                """)
            
            if st.button("Calculate Distance"):
                if point_D and plane_normal and plane_constant:
                    calculator.point_to_plane_distance(point_D, plane_normal, plane_constant)
                else:
                    st.error("Please enter the point, normal vector, and constant.")
        
        elif operation == "Line-Plane Intersection":
            st.write("This operation finds the intersection point of a line and a plane.")
            
            col1, col2 = st.columns(2)
            with col1:
                line_point = st.text_input(
                    "Enter Line Point (x, y, z):",
                    value="1, 0, 0",
                    help="A point on the line"
                )
                line_direction = st.text_input(
                    "Enter Line Direction (dx, dy, dz):",
                    value="1, 1, 1", 
                    help="Direction vector of the line"
                )
            with col2:
                plane_normal_int = st.text_input(
                    "Enter Plane Normal (nx, ny, nz):",
                    value="8, -4, 0",
                    help="Normal vector to the plane"
                )
                plane_constant_int = st.text_input(
                    "Enter Plane Constant (k):",
                    value="18",
                    help="Constant in plane equation nx + ny + nz = k"
                )
            
            with st.expander("Help: Line-Plane Intersection Information"):
                st.write("""
                This calculation finds where a line intersects a plane in 3D space.
                
                **Method:**
                1. Line equation: P(t) = P₀ + t·d
                2. Plane equation: n·r = k
                3. Substitute line into plane: n·(P₀ + t·d) = k
                4. Solve for parameter t: t = (k - n·P₀)/(n·d)
                5. Calculate intersection: P(t)
                
                **Special Cases:**
                - If n·d = 0: Line is parallel to plane (no intersection or line lies in plane)
                - If n·d ≠ 0: Unique intersection point exists
                
                **Input Format:**
                - Line point: x, y, z coordinates of any point on the line
                - Line direction: dx, dy, dz components of direction vector
                - Plane normal: nx, ny, nz components of normal vector
                - Plane constant: single number k from equation nx + ny + nz = k
                """)
            
            if st.button("Find Intersection"):
                if line_point and line_direction and plane_normal_int and plane_constant_int:
                    calculator.line_plane_intersection(line_point, line_direction, plane_normal_int, plane_constant_int)
                else:
                    st.error("Please enter all required values.")
        
        elif operation == "Plane Equation":
            st.write("This operation converts between different representations of plane equations.")
            
            input_type = st.selectbox(
                "Select input format:",
                ["Three Points", "Normal Vector + Point", "Coordinate Form"]
            )
            
            if input_type == "Three Points":
                equation_input = st.text_input(
                    "Enter three points (format: x1,y1,z1 ; x2,y2,z2 ; x3,y3,z3):",
                    value="3,1,-2 ; 4,1,-1 ; -4,5,-2",
                    help="Three points separated by semicolons"
                )
                st.info("💡 Example: 3,1,-2 ; 4,1,-1 ; -4,5,-2")
                
            elif input_type == "Normal Vector + Point":
                equation_input = st.text_input(
                    "Enter normal vector and point (format: nx,ny,nz ; px,py,pz):",
                    value="1,2,-1 ; 3,0,5",
                    help="Normal vector and point separated by semicolon"
                )
                st.info("💡 Example: 1,2,-1 ; 3,0,5")
                
            elif input_type == "Coordinate Form":
                equation_input = st.text_input(
                    "Enter plane equation (format: ax + by + cz = d):",
                    value="2x - 3y + z = 5",
                    help="Standard coordinate form of plane equation"
                )
                st.info("💡 Example: 2x - 3y + z = 5")
            
            with st.expander("Help: Plane Equation Formats"):
                st.write("""
                This tool converts between different ways to represent a plane:
                
                **1. Three Points:** Three non-collinear points that lie on the plane
                - Input: x1,y1,z1 ; x2,y2,z2 ; x3,y3,z3
                - Method: Calculate normal vector using cross product
                
                **2. Normal Vector + Point:** Normal vector and any point on the plane
                - Input: nx,ny,nz ; px,py,pz  
                - Method: Use n·(r-p) = 0 to get n·r = n·p
                
                **3. Coordinate Form:** Standard algebraic equation
                - Input: ax + by + cz = d
                - Method: Extract coefficients (a,b,c) as normal vector
                
                **Output:** All three equivalent representations plus parametric form
                """)
            
            if st.button("Convert Plane Equation"):
                if equation_input:
                    calculator.plane_equation_converter(input_type, equation_input)
                else:
                    st.error("Please enter the plane information.")
    
    elif category == "Matrix Operations":
        matrix_operations = ["Batch Expression Calculator", "Matrix Addition", "Matrix Subtraction", "Scalar Multiplication", "Matrix Transpose", 
                           "Matrix Multiplication", "Matrix Determinant", "Matrix Inverse", "Gauss-Jordan System Solver", "Special Matrices", 
                           "Eigenvalues and Eigenvectors"]
        
        # Auto-select operation if it came from search
        default_index = 0
        if selected_operation_from_search:
            # Find the closest matching operation
            for i, op in enumerate(matrix_operations):
                if selected_operation_from_search in op or op in selected_operation_from_search:
                    default_index = i
                    break
                    
        operation = st.sidebar.selectbox(
            "Select Matrix Operation",
            matrix_operations,
            index=default_index
        )
        
        st.markdown(f'<h2 class="animate-subheader">{operation}</h2>', unsafe_allow_html=True)
        
        if operation == "Batch Expression Calculator":
            batch_calculator.render_batch_calculator()
            
        elif operation == "Matrix Addition":
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
                    calculator.matrix_operations("multiply_scalar", matrix_a, None, str(scalar))
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
        
        elif operation == "Matrix Multiplication":
            st.write("This operation multiplies two matrices together.")
            
            col1, col2 = st.columns(2)
            with col1:
                matrix_a = st.text_area(
                    "Enter Matrix A (format: a,b,c; d,e,f or separate rows with newlines):",
                    value="1, 2, 3\n4, 5, 6"
                )
            
            with col2:
                matrix_b = st.text_area(
                    "Enter Matrix B (format: a,b,c; d,e,f or separate rows with newlines):",
                    value="7, 8\n9, 10\n11, 12"
                )
            
            with st.expander("Help: Matrix Multiplication"):
                st.write("""
                Matrix multiplication requires that the number of columns in the first matrix (A) equals 
                the number of rows in the second matrix (B).
                
                If A is a m×n matrix and B is a n×p matrix, their product C = A×B will be a m×p matrix.
                
                Example:
                - A = 2×3 matrix (2 rows, 3 columns)
                - B = 3×2 matrix (3 rows, 2 columns)
                - C = A×B will be a 2×2 matrix
                
                Each element c_{ij} in the product is calculated as:
                c_{ij} = Σ(a_{ik} * b_{kj}) for k = 1 to n
                """)
            
            if st.button("Calculate Product"):
                if matrix_a and matrix_b:
                    calculator.matrix_multiply(matrix_a, matrix_b)
                else:
                    st.error("Please enter both matrices.")
        
        elif operation == "Matrix Determinant":
            st.write("This operation calculates the determinant of a square matrix.")
            
            matrix = st.text_area(
                "Enter Matrix (format: a,b,c; d,e,f or separate rows with newlines):",
                value="1, 2\n3, 4"
            )
            
            with st.expander("Help: Matrix Determinant"):
                st.write("""
                The determinant is a scalar value calculated from a square matrix.
                
                Properties:
                - Only defined for square matrices
                - Determinant of 2×2 matrix [[a, b], [c, d]] is ad - bc
                - For larger matrices, calculated using minors and cofactors
                - If determinant = 0, the matrix is singular (not invertible)
                - Determinant of a product = product of determinants
                
                The determinant has geometric meaning: it represents the scaling factor of the transformation 
                represented by the matrix.
                """)
            
            if st.button("Calculate Determinant"):
                if matrix:
                    calculator.matrix_determinant(matrix)
                else:
                    st.error("Please enter a matrix.")
        
        elif operation == "Matrix Inverse":
            st.write("This operation calculates the inverse of a square matrix.")
            
            matrix = st.text_area(
                "Enter Matrix (format: a,b,c; d,e,f or separate rows with newlines):",
                value="1, 2\n3, 4"
            )
            
            with st.expander("Help: Matrix Inverse"):
                st.write("""
                The inverse of a matrix A is denoted A⁻¹ and satisfies: A × A⁻¹ = A⁻¹ × A = I (identity matrix).
                
                Properties:
                - Only defined for square matrices with non-zero determinant
                - For a 2×2 matrix [[a, b], [c, d]], the inverse is:
                  [[d/(ad-bc), -b/(ad-bc)], [-c/(ad-bc), a/(ad-bc)]]
                - If A is invertible, the system Ax = b has unique solution x = A⁻¹b
                
                A matrix without an inverse is called singular or non-invertible.
                """)
            
            if st.button("Calculate Inverse"):
                if matrix:
                    calculator.matrix_inverse(matrix)
                else:
                    st.error("Please enter a matrix.")
                    
        elif operation == "Gauss-Jordan System Solver":
            st.write("Solve linear systems using Gauss-Jordan elimination with step-by-step visualization.")
            
            input_method = st.radio("Input method:", ["Augmented Matrix", "Separate A and b"])
            
            if input_method == "Augmented Matrix":
                matrix_input = st.text_area(
                    "Enter augmented matrix [A|b] (rows separated by semicolons):",
                    value="2, 1, -1, 8; -3, -1, 2, -11; -2, 1, 2, -3",
                    height=100
                )
                
                if st.button("Solve using Gauss-Jordan"):
                    try:
                        augmented = MathUtils.parse_matrix(matrix_input)
                        rows, cols = augmented.shape
                        
                        if rows != cols - 1:
                            st.error(f"Augmented matrix should have {rows} rows and {rows + 1} columns for a {rows}×{rows} system. Got {rows}×{cols}")
                            return
                        
                        calculator.gauss_jordan_solver(augmented)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            else:  # Separate A and b
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    a_input = st.text_area(
                        "Coefficient Matrix A:",
                        value="2, 1, -1; -3, -1, 2; -2, 1, 2",
                        height=100
                    )
                
                with col2:
                    b_input = st.text_input(
                        "Constants Vector b:",
                        value="8; -11; -3"
                    )
                
                if st.button("Solve using Gauss-Jordan"):
                    try:
                        A = MathUtils.parse_matrix(a_input)
                        
                        # Parse b vector
                        if ';' in b_input:
                            b = np.array([float(x.strip()) for x in b_input.split(';')])
                        else:
                            b = np.array([float(x.strip()) for x in b_input.split(',')])
                        
                        if A.shape[0] != A.shape[1]:
                            st.error("Coefficient matrix A must be square")
                            return
                        
                        if len(b) != A.shape[0]:
                            st.error("Vector b must have same number of elements as rows in A")
                            return
                        
                        # Create augmented matrix
                        augmented = np.hstack([A, b.reshape(-1, 1)])
                        calculator.gauss_jordan_solver(augmented)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with st.expander("Help: Gauss-Jordan Elimination"):
                st.write("""
                Gauss-Jordan elimination is an extension of Gaussian elimination that produces 
                the reduced row echelon form (RREF) of a matrix.
                
                **Steps:**
                1. Forward elimination: Create zeros below each pivot
                2. Backward elimination: Create zeros above each pivot  
                3. Scale: Make each pivot equal to 1
                
                **Result:** The solution can be read directly from the final augmented matrix.
                
                **Advantages:**
                - No back-substitution needed
                - Solution is immediately visible
                - Can handle systems with multiple solutions
                """)
                    
        elif operation == "Special Matrices":
            st.write("This operation generates and explains special matrices with specific properties.")
            
            if not hasattr(calculator, 'special_matrices_generator'):
                st.error("Special matrices generator not implemented yet. We'll add it soon!")
            else:
                calculator.special_matrices_generator()
                
            with st.expander("Help: Special Matrices"):
                st.write("""
                Special matrices have particular forms or properties that make them useful in various applications:
                
                - **Identity Matrix**: Has 1s on the diagonal and 0s elsewhere. Acts as the multiplicative identity.
                - **Zero Matrix**: All elements are zero. Acts as the additive identity.
                - **Diagonal Matrix**: Only has non-zero entries on the main diagonal.
                - **Triangular Matrix**: Either upper (zeros below diagonal) or lower (zeros above diagonal).
                - **Symmetric Matrix**: Equal to its transpose (A = A^T).
                - **Orthogonal Matrix**: Its transpose equals its inverse (Q^T = Q^(-1)).
                - **Rotation Matrix**: Represents a rotation in space, preserves distances and angles.
                - **Scaling Matrix**: Stretches or shrinks vectors along coordinate axes.
                - **Reflection Matrix**: Reflects vectors across a line, plane, or point.
                
                Special matrices simplify calculations and have applications in graphics, physics, statistics, and more.
                """)
        
        elif operation == "Eigenvalues and Eigenvectors":
            st.write("This operation calculates the eigenvalues and eigenvectors of a square matrix.")
            
            matrix = st.text_area(
                "Enter Matrix (format: a,b,c; d,e,f or separate rows with newlines):",
                value="1, 2\n3, 4"
            )
            
            with st.expander("Help: Eigenvalues and Eigenvectors"):
                st.write("""
                An eigenvector of a square matrix A is a non-zero vector v such that when multiplied 
                by A, yields a scalar multiple of itself: Av = λv. This scalar λ is the eigenvalue 
                associated with the eigenvector v.
                
                Properties:
                - Only defined for square matrices
                - Eigenvalues are the roots of the characteristic polynomial: det(A - λI) = 0
                - A matrix with n×n dimensions has n eigenvalues (counting multiplicities)
                - Real symmetric matrices have real eigenvalues and orthogonal eigenvectors
                
                Applications:
                - Principal Component Analysis (PCA)
                - Stability analysis in dynamical systems
                - Quantum mechanics
                - Google's PageRank algorithm
                """)
            
            if st.button("Calculate Eigenvalues and Eigenvectors"):
                if matrix:
                    calculator.eigenvalues_eigenvectors(matrix)
                else:
                    st.error("Please enter a matrix.")
    
    elif category == "Determinants":
        # Use the determinant operations component
        determinant_calculator.render_determinant_calculator()
    
    elif category == "Systems of Linear Equations":
        system_operations = ["Solve System (Gaussian Elimination)", "Cramer's Rule", "Standard Form Analysis", "Row Operations Analysis", 
                           "Free Variable / Pivot Analysis", "Homogeneous/Inhomogeneous Solutions", "Check Vector Solutions",
                           "Calculate Null Space (Basis)", "Geometric Interpretation"]
        
        # Auto-select operation if it came from search
        default_index = 0
        if selected_operation_from_search:
            # Find the closest matching operation
            for i, op in enumerate(system_operations):
                if selected_operation_from_search in op or op in selected_operation_from_search:
                    default_index = i
                    break
                    
        operation = st.sidebar.selectbox(
            "Select Operation",
            system_operations,
            index=default_index
        )
        
        st.markdown(f'<h2 class="animate-subheader">{operation}</h2>', unsafe_allow_html=True)
        
        # Unified matrix input for all operations in this category.
        # Label will guide user on what to input based on the operation.
        matrix_input = st.text_area(
            "Enter Matrix (for Ax=b systems, use [A|b] format; for Null Space, enter only coefficient matrix A):",
            value="-5, 2, 0, -9\n3, 0, 3, 0",
            key="system_matrix_unified_input" # A unique key for this unified input
        )
        
        # The expander provides general help for [A|b] but users are guided by the main label for Null Space.
        with st.expander("Help: Matrix Format Explanation (primarily for [A|b] systems)"):
            st.write("""
            For a system of linear equations like:
            ```
            a₁x + b₁y + c₁z = d₁
            a₂x + b₂y + c₂z = d₂
            a₃x + b₃y + c₃z = d₃
            ```
            The **augmented matrix [A|b]** would be:
            ```
            a₁, b₁, c₁, d₁
            a₂, b₂, c₂, d₂
            a₃, b₃, c₃, d₃
            ```
            When calculating a **Null Space (Ax=0)**, you only need to enter the **coefficient matrix A** into the field above:
            ```
            a₁, b₁, c₁
            a₂, b₂, c₂
            a₃, b₃, c₃
            ```
            """)
        
        equations_input = None # Initialize, only used by Standard Form Analysis if checkbox is ticked

        if operation == "Solve System (Gaussian Elimination)":
            st.write("This operation solves a system of linear equations using Gaussian elimination.")
            if st.button("Solve System"):
                if matrix_input:
                    calculator.solve_gauss(matrix_input)
                else:
                    st.error("Please enter the augmented matrix [A|b].")
        
        elif operation == "Cramer's Rule":
            # Use the Cramer's Rule component
            cramers_rule_calculator.render_cramers_rule_calculator()
                    
        elif operation == "Standard Form Analysis":
            st.write("This operation analyzes a system and presents it in the standard form (Ax = b).")
            use_equations = st.checkbox("Enter as equations instead of matrix")
            
            if use_equations:
                equations_input = st.text_area(
                    "Enter system of equations (one per line, format: 2x1 - 3x2 + 5x3 = 7):",
                    value="x1 - 4x2 - 2x3 = -25\n-3x2 + 6x3 = -18\n7x1 - 13x2 - 4x3 = -85",
                    key="system_equations_input"
                )
            
            if st.button("Analyze Standard Form"):
                if use_equations and equations_input:
                    calculator.standard_form(None, equations_input)
                elif not use_equations and matrix_input:
                    calculator.standard_form(matrix_input, None)
                else:
                    st.error("Please enter the matrix [A|b] or provide equations.")
                    
        elif operation == "Row Operations Analysis":
            st.write("This operation shows step-by-step row operations to solve the system.")
            if st.button("Analyze Row Operations"):
                if matrix_input:
                    calculator.row_operations_analysis(matrix_input)
                else:
                    st.error("Please enter the augmented matrix [A|b].")
                    
        elif operation == "Free Variable / Pivot Analysis":
            st.write("This operation identifies free parameters and expresses the solution in parametric form.")
            if st.button("Analyze Free Parameters"):
                if matrix_input:
                    calculator.free_parameter_analysis(matrix_input)
                else:
                    st.error("Please enter the augmented matrix [A|b].")
                    
        elif operation == "Homogeneous/Inhomogeneous Solutions":
            st.write("This operation compares the solutions of Ax = b and Ax = 0 for the same coefficient matrix.")
            if st.button("Analyze Solution Relationship"):
                if matrix_input:
                    calculator.homogeneous_inhomogeneous_solutions(matrix_input)
                else:
                    st.error("Please enter the augmented matrix [A|b].")

        elif operation == "Check Vector Solutions":
            st.write("This operation checks if given vectors are particular solutions, homogeneous solutions, or not solutions at all.")
            st.info("💡 **For this problem**: Use matrix `-5,2,0,-9; 3,0,3,0` and vectors `1,-2,-1`, `2,5,-2`, `3,3,-3`")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**Vectors to check (one per line):**")
                vectors_input = st.text_area(
                    "Enter vectors to check:",
                    value="1, -2, -1\n2, 5, -2\n3, 3, -3",
                    help="Enter one vector per line, comma-separated components. These should match the dimension of your system.",
                    key="vectors_to_check"
                )
            
            with col2:
                with st.expander("Help: Understanding Solution Types"):
                    st.write("""
                    **Particular Solution**: A vector x such that Ax = b (solves the inhomogeneous system)
                    
                    **Homogeneous Solution**: A vector x such that Ax = 0 (solves the homogeneous system)
                    
                    **Not a Solution**: A vector that satisfies neither condition
                    
                    **Note**: If a vector is both a particular solution AND the system has b ≠ 0, 
                    then it cannot simultaneously be a homogeneous solution.
                    """)
            
            if st.button("Check Vector Solutions"):
                if matrix_input and vectors_input:
                    calculator.check_vector_solutions(matrix_input, vectors_input)
                else:
                    st.error("Please enter both the augmented matrix [A|b] and vectors to check.")

        elif operation == "Calculate Null Space (Basis)":
            st.write("This operation calculates a basis for the Null Space (Kernel) of a given matrix A (solutions to Ax = 0).")
            st.caption("Ensure you enter only the coefficient matrix A into the input field above.")
            
            # No separate input field here; it uses the common matrix_input from the category.
            # The help text for matrix_input and the caption above guide the user.
            
            with st.expander("Help: Null Space (Kernel) - Quick Reminder"):
                st.write("""
                The Null Space of a matrix A contains all vectors **x** such that A**x** = **0**.
                This operation finds a basis for that space using the matrix A you provide in the field above.
                """)

            if st.button("Calculate Null Space Basis"):
                if matrix_input: 
                    calculator.calculate_null_space_basis(matrix_input) # Uses the common matrix_input
                else:
                    st.error("Please enter the coefficient matrix A in the field above.")
        
        elif operation == "Geometric Interpretation":
            st.write("This operation visualizes the geometric meaning of the system as intersecting lines/planes.")
            if st.button("Show Geometric Interpretation"):
                if matrix_input: 
                    calculator.geometric_interpretation(matrix_input)
                else:
                    st.error("Please enter the augmented matrix [A|b].")
    
    elif category == "Linear Mappings":
        # Use the linearity checker component (it has its own header)
        linearity_checker.render_linearity_checker()
    
    elif category == "Complex Numbers":
        # Use the complex operations component
        complex_calculator.render_complex_calculator()
    
    elif category == "Series & Summations":
        summation_calculator.render_summation_calculator()
    
    elif category == "Learning Resources":
        render_learning_resources_page()
    

    # Quick Start & Tips Expander
    with st.sidebar.expander("💡 Quick Start & Tips", expanded=False):
        st.markdown("""
        #### How to use this calculator:
        1.  **Search or Browse:** Use the search bar in the sidebar or browse categories.
        2.  **Select an Operation:** Choose the specific calculation you need.
        3.  **Enter Inputs:** Provide vectors/matrices in the specified format (e.g., `1,2,3` or `1,2;3,4`).
        4.  **Get Results:** Outputs and explanations will appear below the inputs.
        5.  **Visit Learning Resources:** For guides, examples, and concept explanations.
        """)

    # Enhanced footer with animations (CSS now in custom.css)
    st.sidebar.markdown('''
    <div class="footer-container">
        <div class="footer-text">
            Made with <span class="heart-pulse">❤️</span> by <a href="https://doruk.ch" target="_blank">Doruk</a>
        </div>
        <div class="footer-text">FHNW Linear Algebra Module</div>
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
