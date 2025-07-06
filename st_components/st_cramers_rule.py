#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Cramer's Rule
Implements operations from Week 22: Cramer's Rule for solving linear systems
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from .st_math_utils import MathUtils
from .st_visualization_utils import display_matrix_heatmap

class CramersRule:
    """
    Component for solving linear systems using Cramer's Rule.
    Provides step-by-step calculations and visualizations.
    """
    
    def __init__(self):
        self.coefficient_matrix = None
        self.constants_vector = None
        
    def render_cramers_rule_calculator(self):
        """Render the main Cramer's Rule calculator interface."""
        st.header("📊 Cramer's Rule Solver")
        st.write("Solve linear systems Ax = b using Cramer's Rule with step-by-step calculations.")
        
        # Information about Cramer's Rule
        with st.expander("ℹ️ About Cramer's Rule", expanded=False):
            st.markdown("""
            **Cramer's Rule** is a mathematical theorem used to solve systems of linear equations with as many equations as unknowns.
            
            **Requirements:**
            - Square coefficient matrix (n×n)
            - Non-zero determinant (det(A) ≠ 0)
            
            **Formula:**
            For system Ax = b, the solution is:
            $$x_i = \\frac{\\det(A_i)}{\\det(A)}$$
            
            Where $A_i$ is matrix A with column i replaced by vector b.
            """)
        
        # Input method selector
        input_method = st.radio("Input method:", ["Augmented Matrix", "Separate A and b"])
        
        if input_method == "Augmented Matrix":
            self._render_augmented_input()
        else:
            self._render_separate_input()
    
    def _render_augmented_input(self):
        """Render augmented matrix input interface."""
        st.subheader("Augmented Matrix Input [A|b]")
        
        matrix_input = st.text_area(
            "Enter augmented matrix (rows separated by semicolons):",
            value="2, 1, -1, 8; -3, -1, 2, -11; -2, 1, 2, -3",
            height=100,
            help="Format: row1; row2; row3 where each row is coefficient1, coefficient2, ..., constant"
        )
        
        if st.button("Solve using Cramer's Rule"):
            try:
                augmented = MathUtils.parse_matrix(matrix_input)
                if augmented.shape[0] != augmented.shape[1] - 1:
                    st.error("Matrix must be square coefficient matrix with one additional column for constants")
                    return
                
                A = augmented[:, :-1]
                b = augmented[:, -1]
                self._solve_cramers_rule(A, b)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    def _render_separate_input(self):
        """Render separate A and b input interface."""
        st.subheader("Separate Matrix and Vector Input")
        
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
                value="8; -11; -3",
                help="Separate values with semicolons or commas"
            )
        
        if st.button("Solve using Cramer's Rule"):
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
                    st.error("Vector b must have the same number of elements as rows in A")
                    return
                
                self._solve_cramers_rule(A, b)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    def _solve_cramers_rule(self, A: np.ndarray, b: np.ndarray):
        """Solve the system using Cramer's Rule and display step-by-step solution."""
        n = A.shape[0]
        
        # Calculate determinant of A
        det_A = np.linalg.det(A)
        
        # Handle floating point precision
        if abs(det_A) < 1e-10:
            det_A = 0.0
        
        # Check if Cramer's Rule can be applied
        if det_A == 0:
            st.error("❌ **Cramer's Rule cannot be applied!**")
            st.write("The determinant of coefficient matrix A is zero.")
            st.write("The system either has no solution or infinitely many solutions.")
            return
        
        st.success("✅ **Cramer's Rule can be applied!**")
        st.write(f"det(A) = {det_A:.6g} ≠ 0")
        
        # Create layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("### System Setup")
            
            # Display the system
            st.write("**System of equations:**")
            st.latex("A\\mathbf{x} = \\mathbf{b}")
            
            st.write("**Coefficient Matrix A:**")
            st.latex(f"A = {MathUtils.format_matrix_latex(A)}")
            
            st.write("**Constants Vector b:**")
            st.latex(f"\\mathbf{{b}} = {MathUtils.format_vector_latex(b.reshape(-1, 1))}")
            
            st.write("**Main Determinant:**")
            st.latex(f"\\det(A) = {det_A:.6g}")
        
        with col2:
            st.write("### Solution using Cramer's Rule")
            
            # Calculate solution for each variable
            solution = np.zeros(n)
            
            for i in range(n):
                # Create Ai by replacing column i with b
                A_i = A.copy()
                A_i[:, i] = b
                
                # Calculate determinant of Ai
                det_A_i = np.linalg.det(A_i)
                
                # Handle floating point precision
                if abs(det_A_i) < 1e-10:
                    det_A_i = 0.0
                
                # Calculate xi
                x_i = det_A_i / det_A if det_A != 0 else 0
                solution[i] = x_i
                
                # Display calculation
                st.write(f"**Variable x_{i+1}:**")
                st.latex(f"A_{{{i+1}}} = {MathUtils.format_matrix_latex(A_i)}")
                st.latex(f"\\det(A_{{{i+1}}}) = {det_A_i:.6g}")
                st.latex(f"x_{{{i+1}}} = \\frac{{\\det(A_{{{i+1}}})}}{{\\det(A)}} = \\frac{{{det_A_i:.6g}}}{{{det_A:.6g}}} = {x_i:.6g}")
                st.markdown("---")
        
        # Display final solution
        st.write("### 🎯 Final Solution")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Solution Vector:**")
            solution_latex = "\\mathbf{x} = " + MathUtils.format_vector_latex(solution.reshape(-1, 1))
            st.latex(solution_latex)
            
            # Component form
            st.write("**Component Form:**")
            for i, x_val in enumerate(solution):
                st.latex(f"x_{{{i+1}}} = {x_val:.6g}")
        
        with col2:
            st.write("**Verification:**")
            
            # Verify the solution
            verification = A @ solution
            st.write("Check: A × **x** = **b**")
            
            verification_latex = MathUtils.format_matrix_latex(A) + " \\cdot " + \
                               MathUtils.format_vector_latex(solution.reshape(-1, 1)) + " = " + \
                               MathUtils.format_vector_latex(verification.reshape(-1, 1))
            
            st.latex(verification_latex)
            
            # Check if solution is correct
            if np.allclose(verification, b, rtol=1e-6):
                st.success("✅ Solution verified!")
            else:
                st.warning("⚠️ Verification shows small numerical differences")
        
        # Detailed step-by-step breakdown
        with st.expander("📋 Detailed Step-by-Step Breakdown"):
            self._show_detailed_steps(A, b, solution, det_A)
    
    def _show_detailed_steps(self, A: np.ndarray, b: np.ndarray, solution: np.ndarray, det_A: float):
        """Show detailed step-by-step breakdown of Cramer's Rule."""
        n = A.shape[0]
        
        st.write("### Step-by-Step Cramer's Rule Application")
        
        st.write("**Step 1: Calculate det(A)**")
        st.latex(f"\\det(A) = \\det{MathUtils.format_matrix_latex(A)} = {det_A:.6g}")
        
        st.write("**Step 2: Create matrices A_i and calculate their determinants**")
        
        for i in range(n):
            st.write(f"**For variable x_{i+1}:**")
            
            # Create Ai
            A_i = A.copy()
            A_i[:, i] = b
            det_A_i = np.linalg.det(A_i)
            
            # Handle floating point precision
            if abs(det_A_i) < 1e-10:
                det_A_i = 0.0
            
            st.write(f"Replace column {i+1} of A with vector b:")
            st.latex(f"A_{{{i+1}}} = {MathUtils.format_matrix_latex(A_i)}")
            
            st.write(f"Calculate determinant:")
            st.latex(f"\\det(A_{{{i+1}}}) = {det_A_i:.6g}")
            
            st.write(f"Apply Cramer's Rule:")
            x_i = det_A_i / det_A
            st.latex(f"x_{{{i+1}}} = \\frac{{\\det(A_{{{i+1}}})}}{{\\det(A)}} = \\frac{{{det_A_i:.6g}}}{{{det_A:.6g}}} = {x_i:.6g}")
            
            st.markdown("---")
        
        st.write("**Step 3: Combine results**")
        solution_latex = "\\mathbf{x} = " + MathUtils.format_vector_latex(solution.reshape(-1, 1))
        st.latex(solution_latex)
    
