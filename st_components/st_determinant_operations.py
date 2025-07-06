#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Enhanced Determinant Operations
Implements operations from Week 21: Determinants with geometric interpretations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Optional, List, Dict, Union
from .st_math_utils import MathUtils
from .st_visualization_utils import display_matrix_heatmap

class DeterminantOperations:
    """
    Component for enhanced determinant operations with geometric interpretations.
    Includes 2D determinant (parallelogram area), 3D determinant (Sarrus rule),
    and educational visualizations.
    """
    
    def __init__(self):
        self.matrix = None
        
    def render_determinant_calculator(self):
        """Render the main determinant calculator interface."""
        st.header("📐 Determinant Calculator & Geometric Interpretation")
        st.write("Calculate determinants with step-by-step explanations and geometric visualizations.")
        
        # Operation selector
        operation = st.selectbox(
            "Select Operation",
            ["2×2 Determinant & Parallelogram Area", "3×3 Determinant & Sarrus Rule", 
             "General Determinant"]
        )
        
        if operation == "2×2 Determinant & Parallelogram Area":
            self._render_2d_determinant()
        elif operation == "3×3 Determinant & Sarrus Rule":
            self._render_3d_determinant()
        elif operation == "General Determinant":
            self._render_general_determinant()
    
    def _render_2d_determinant(self):
        """Render 2×2 determinant with parallelogram area interpretation."""
        st.subheader("2×2 Determinant & Parallelogram Area")
        
        # Input method selector
        input_method = st.radio("Input method:", ["Matrix", "Two Vectors"])
        
        if input_method == "Matrix":
            matrix_input = st.text_area(
                "Enter 2×2 matrix (rows separated by semicolons):",
                value="3, 1; 1, 2",
                height=100
            )
            
            if st.button("Calculate Determinant"):
                try:
                    matrix = MathUtils.parse_matrix(matrix_input)
                    if matrix.shape != (2, 2):
                        st.error("Please enter a 2×2 matrix")
                        return
                    
                    self._calculate_2d_determinant(matrix)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
        else:  # Two Vectors
            col1, col2 = st.columns(2)
            with col1:
                v1_input = st.text_input("First vector (comma-separated):", value="3, 1")
            with col2:
                v2_input = st.text_input("Second vector (comma-separated):", value="1, 2")
            
            if st.button("Calculate Determinant"):
                try:
                    v1 = np.array([float(x.strip()) for x in v1_input.split(',')])
                    v2 = np.array([float(x.strip()) for x in v2_input.split(',')])
                    
                    if len(v1) != 2 or len(v2) != 2:
                        st.error("Please enter 2D vectors")
                        return
                    
                    matrix = np.array([v1, v2])
                    self._calculate_2d_determinant(matrix)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    def _calculate_2d_determinant(self, matrix: np.ndarray):
        """Calculate and visualize 2×2 determinant."""
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        det = a * d - b * c
        
        # Handle floating point precision issues
        if abs(det) < 1e-10:
            det = 0.0
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("### Calculation")
            if det == 0:
                st.latex(f"""
                \\begin{{vmatrix}}
                {a:.4g} & {b:.4g} \\\\
                {c:.4g} & {d:.4g}
                \\end{{vmatrix}} = {a:.4g} \\cdot {d:.4g} - {b:.4g} \\cdot {c:.4g} = 0
                """)
            else:
                st.latex(f"""
                \\begin{{vmatrix}}
                {a:.4g} & {b:.4g} \\\\
                {c:.4g} & {d:.4g}
                \\end{{vmatrix}} = {a:.4g} \\cdot {d:.4g} - {b:.4g} \\cdot {c:.4g} = {det:.4g}
                """)
            
            st.write("### Geometric Interpretation")
            if det == 0:
                st.write("**Area of Parallelogram:** 0 (vectors are collinear)")
            else:
                st.write(f"**Area of Parallelogram:** |{det:.4g}| = {abs(det):.4g}")
            
            if det > 0:
                st.success("Positive determinant: Vectors have same orientation")
            elif det < 0:
                st.warning("Negative determinant: Vectors have opposite orientation")
            else:
                st.error("Zero determinant: Vectors are collinear (no area)")
            
            # Step-by-step explanation
            with st.expander("Step-by-Step Explanation"):
                st.write("**Step 1:** Identify matrix elements")
                st.latex(f"a = {a:.4g}, b = {b:.4g}, c = {c:.4g}, d = {d:.4g}")
                
                st.write("**Step 2:** Apply the formula")
                st.latex(f"\\det(A) = ad - bc")
                
                st.write("**Step 3:** Calculate products")
                st.latex(f"ad = {a:.4g} \\times {d:.4g} = {a*d:.4g}")
                st.latex(f"bc = {b:.4g} \\times {c:.4g} = {b*c:.4g}")
                
                st.write("**Step 4:** Final result")
                st.latex(f"\\det(A) = {a*d:.4g} - {b*c:.4g} = {det:.4g}")
        
        with col2:
            st.write("### Visualization")
            fig = self._create_parallelogram_plot(matrix)
            st.plotly_chart(fig)
            
            # Additional properties
            st.write("### Properties")
            st.write(f"- **Determinant:** {det:.4g}")
            st.write(f"- **Area:** {abs(det):.4g}")
            if det != 0:
                st.write(f"- **Matrix is invertible**")
            else:
                st.write(f"- **Matrix is singular (not invertible)**")
    
    def _create_parallelogram_plot(self, matrix: np.ndarray) -> go.Figure:
        """Create a plot showing the parallelogram formed by two vectors."""
        v1 = matrix[0]
        v2 = matrix[1]
        
        # Create figure
        fig = go.Figure()
        
        # Calculate bounds
        points = np.array([[0, 0], v1, v2, v1 + v2])
        max_val = np.max(np.abs(points)) * 1.2
        
        # Add coordinate axes
        fig.add_trace(go.Scatter(
            x=[-max_val, max_val], y=[0, 0],
            mode='lines',
            line=dict(color='white', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[-max_val, max_val],
            mode='lines',
            line=dict(color='white', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add vectors
        fig.add_trace(go.Scatter(
            x=[0, v1[0]], y=[0, v1[1]],
            mode='lines+markers',
            name='v₁',
            line=dict(color='#00D9FF', width=3),
            marker=dict(size=[0, 10], symbol='arrow', angleref='previous')
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, v2[0]], y=[0, v2[1]],
            mode='lines+markers',
            name='v₂',
            line=dict(color='#FF0080', width=3),
            marker=dict(size=[0, 10], symbol='arrow', angleref='previous')
        ))
        
        # Add parallelogram
        parallelogram_x = [0, v1[0], v1[0] + v2[0], v2[0], 0]
        parallelogram_y = [0, v1[1], v1[1] + v2[1], v2[1], 0]
        
        fig.add_trace(go.Scatter(
            x=parallelogram_x, y=parallelogram_y,
            mode='lines',
            name='Parallelogram',
            line=dict(color='#FFD700', width=2, dash='dot'),
            fill='toself',
            fillcolor='rgba(255, 215, 0, 0.2)',
            hoverinfo='skip'
        ))
        
        # Add annotations
        fig.add_annotation(
            x=v1[0]/2, y=v1[1]/2,
            text="v₁",
            showarrow=False,
            font=dict(color='#00D9FF', size=14),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='#00D9FF'
        )
        
        fig.add_annotation(
            x=v2[0]/2, y=v2[1]/2,
            text="v₂",
            showarrow=False,
            font=dict(color='#FF0080', size=14),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='#FF0080'
        )
        
        # Calculate area for annotation
        det = v1[0] * v2[1] - v1[1] * v2[0]
        fig.add_annotation(
            x=(v1[0] + v2[0])/2, y=(v1[1] + v2[1])/2,
            text=f"Area = |{det:.2f}| = {abs(det):.2f}",
            showarrow=False,
            font=dict(color='white', size=12),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='#FFD700'
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text="Parallelogram formed by vectors", font=dict(color="white")),
            xaxis=dict(
                range=[-max_val, max_val],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='white',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255, 255, 255, 0.1)',
                color='white',
                title='x'
            ),
            yaxis=dict(
                range=[-max_val, max_val],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='white',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255, 255, 255, 0.1)',
                color='white',
                title='y',
                scaleanchor="x",
                scaleratio=1
            ),
            width=500,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            showlegend=True,
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        return fig
    
    def _render_3d_determinant(self):
        """Render 3×3 determinant with Sarrus rule visualization."""
        st.subheader("3×3 Determinant & Sarrus Rule")
        
        matrix_input = st.text_area(
            "Enter 3×3 matrix (rows separated by semicolons):",
            value="1, 2, 3; 4, 5, 6; 7, 8, 9",
            height=100
        )
        
        if st.button("Calculate Determinant"):
            try:
                matrix = MathUtils.parse_matrix(matrix_input)
                if matrix.shape != (3, 3):
                    st.error("Please enter a 3×3 matrix")
                    return
                
                self._calculate_3d_determinant(matrix)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    def _calculate_3d_determinant(self, matrix: np.ndarray):
        """Calculate and visualize 3×3 determinant using Sarrus rule."""
        det = np.linalg.det(matrix)
        
        # Handle floating point precision issues
        if abs(det) < 1e-10:
            det = 0.0
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("### Matrix")
            st.latex(MathUtils.format_matrix_latex(matrix))
            
            st.write("### Determinant")
            if det == 0:
                st.latex("\\det(A) = 0")
            else:
                st.latex(f"\\det(A) = {det:.4g}")
            
            if det == 0:
                st.error("Zero determinant: Matrix is singular")
            else:
                st.success(f"Non-zero determinant: Matrix is invertible")
                
            # Properties
            st.write("### Properties")
            if det == 0:
                st.write("- **Volume of parallelepiped:** 0 (vectors are coplanar)")
                st.write("- **Orientation:** Undefined (singular matrix)")
            else:
                st.write(f"- **Volume of parallelepiped:** |{det:.4g}| = {abs(det):.4g}")
                if det > 0:
                    st.write("- **Orientation:** Right-handed")
                elif det < 0:
                    st.write("- **Orientation:** Left-handed")
        
        with col2:
            st.write("### Sarrus Rule Visualization")
            self._visualize_sarrus_rule(matrix)
            
            # Step-by-step calculation
            with st.expander("Step-by-Step Calculation"):
                self._show_sarrus_calculation(matrix)
    
    def _visualize_sarrus_rule(self, matrix: np.ndarray):
        """Create a visual representation of Sarrus rule."""
        # Create extended matrix for Sarrus rule
        extended = np.hstack([matrix, matrix[:, :2]])
        
        st.write("**Extended Matrix for Sarrus Rule:**")
        
        # Format the extended matrix with highlighting
        extended_latex = r"\begin{bmatrix}"
        for i in range(3):
            row = []
            for j in range(5):
                val = extended[i, j]
                if j < 3:
                    row.append(f"{val:.4g}")
                else:
                    row.append(f"\\color{{gray}}{{{val:.4g}}}")
            extended_latex += " & ".join(row)
            if i < 2:
                extended_latex += r" \\ "
        extended_latex += r"\end{bmatrix}"
        
        st.latex(extended_latex)
        
        # Show diagonal products
        st.write("**Positive diagonals (↘):**")
        pos_diag1 = matrix[0,0] * matrix[1,1] * matrix[2,2]
        pos_diag2 = matrix[0,1] * matrix[1,2] * matrix[2,0]
        pos_diag3 = matrix[0,2] * matrix[1,0] * matrix[2,1]
        
        st.latex(f"""
        \\begin{{align}}
        & {matrix[0,0]:.4g} \\times {matrix[1,1]:.4g} \\times {matrix[2,2]:.4g} = {pos_diag1:.4g} \\\\
        & {matrix[0,1]:.4g} \\times {matrix[1,2]:.4g} \\times {matrix[2,0]:.4g} = {pos_diag2:.4g} \\\\
        & {matrix[0,2]:.4g} \\times {matrix[1,0]:.4g} \\times {matrix[2,1]:.4g} = {pos_diag3:.4g}
        \\end{{align}}
        """)
        
        st.write("**Negative diagonals (↙):**")
        neg_diag1 = matrix[0,2] * matrix[1,1] * matrix[2,0]
        neg_diag2 = matrix[0,0] * matrix[1,2] * matrix[2,1]
        neg_diag3 = matrix[0,1] * matrix[1,0] * matrix[2,2]
        
        st.latex(f"""
        \\begin{{align}}
        & {matrix[0,2]:.4g} \\times {matrix[1,1]:.4g} \\times {matrix[2,0]:.4g} = {neg_diag1:.4g} \\\\
        & {matrix[0,0]:.4g} \\times {matrix[1,2]:.4g} \\times {matrix[2,1]:.4g} = {neg_diag2:.4g} \\\\
        & {matrix[0,1]:.4g} \\times {matrix[1,0]:.4g} \\times {matrix[2,2]:.4g} = {neg_diag3:.4g}
        \\end{{align}}
        """)
    
    def _show_sarrus_calculation(self, matrix: np.ndarray):
        """Show step-by-step Sarrus rule calculation."""
        # Calculate diagonal products
        pos_diag1 = matrix[0,0] * matrix[1,1] * matrix[2,2]
        pos_diag2 = matrix[0,1] * matrix[1,2] * matrix[2,0]
        pos_diag3 = matrix[0,2] * matrix[1,0] * matrix[2,1]
        
        neg_diag1 = matrix[0,2] * matrix[1,1] * matrix[2,0]
        neg_diag2 = matrix[0,0] * matrix[1,2] * matrix[2,1]
        neg_diag3 = matrix[0,1] * matrix[1,0] * matrix[2,2]
        
        pos_sum = pos_diag1 + pos_diag2 + pos_diag3
        neg_sum = neg_diag1 + neg_diag2 + neg_diag3
        det = pos_sum - neg_sum
        
        # Handle floating point precision
        if abs(det) < 1e-10:
            det = 0.0
        
        st.write("**Step 1:** Sum of positive diagonals")
        st.latex(f"{pos_diag1:.4g} + {pos_diag2:.4g} + {pos_diag3:.4g} = {pos_sum:.4g}")
        
        st.write("**Step 2:** Sum of negative diagonals")
        st.latex(f"{neg_diag1:.4g} + {neg_diag2:.4g} + {neg_diag3:.4g} = {neg_sum:.4g}")
        
        st.write("**Step 3:** Final determinant")
        if det == 0:
            st.latex(f"\\det(A) = {pos_sum:.4g} - {neg_sum:.4g} = 0")
        else:
            st.latex(f"\\det(A) = {pos_sum:.4g} - {neg_sum:.4g} = {det:.4g}")
    
    def _render_general_determinant(self):
        """Render general n×n determinant calculator."""
        st.subheader("General n×n Determinant")
        
        matrix_input = st.text_area(
            "Enter square matrix (rows separated by semicolons):",
            value="1, 2, 3; 4, 5, 6; 7, 8, 10",
            height=150
        )
        
        if st.button("Calculate Determinant"):
            try:
                matrix = MathUtils.parse_matrix(matrix_input)
                n = matrix.shape[0]
                if matrix.shape[0] != matrix.shape[1]:
                    st.error("Please enter a square matrix")
                    return
                
                det = np.linalg.det(matrix)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("### Matrix")
                    st.latex(MathUtils.format_matrix_latex(matrix))
                    
                    st.write("### Result")
                    st.latex(f"\\det(A) = {det:.6g}")
                    
                    # Properties
                    st.write("### Properties")
                    if abs(det) < 1e-10:
                        st.error("Matrix is singular (det = 0)")
                        st.write("- Not invertible")
                        st.write("- Columns are linearly dependent")
                        st.write("- Has at least one zero eigenvalue")
                    else:
                        st.success("Matrix is non-singular")
                        st.write(f"- Invertible")
                        st.write(f"- Columns are linearly independent")
                        st.write(f"- Volume scaling factor: {abs(det):.6g}")
                
                with col2:
                    st.write("### Visualization")
                    display_matrix_heatmap(matrix, title=f"{n}×{n} Matrix", center_scale=True)
                    
                    if n <= 3:
                        st.write("### Calculation Method")
                        if n == 2:
                            st.write("Direct formula: ad - bc")
                        elif n == 3:
                            st.write("Sarrus rule or cofactor expansion")
                    else:
                        st.write("### Calculation Method")
                        st.write(f"For {n}×{n} matrices, efficient numerical methods")
                        st.write("like LU decomposition are used.")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
