#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Batch Matrix Expression Calculator
Automatically evaluates multiple matrix expressions from calculation lists
"""

import streamlit as st
import numpy as np
import re
from typing import Optional, Tuple

class BatchMatrixCalculator:
    def __init__(self):
        self.matrices = {}
        
    def _format_number_latex(self, num):
        """Helper to format numbers for LaTeX, showing integers without .0"""
        if np.isclose(num, round(num)):
            return f"{int(round(num))}"
        else:
            return f"{num:.4f}".rstrip('0').rstrip('.')
    
    def _format_matrix_to_latex_string(self, matrix_val):
        """Formats a NumPy matrix into a LaTeX pmatrix string."""
        if matrix_val is None or matrix_val.size == 0:
            return "(Empty or invalid matrix)"
        
        matrix_cleaned = np.where(np.isclose(matrix_val, 0) & (np.abs(matrix_val) < 1e-9), 0, matrix_val)
        
        rows_str = []
        for row_idx in range(matrix_cleaned.shape[0]):
            elements = [self._format_number_latex(x) for x in matrix_cleaned[row_idx, :]]
            rows_str.append(" & ".join(elements))
        return r"\begin{pmatrix} " + r" \\ ".join(rows_str) + r" \end{pmatrix}"
    
    def parse_matrix_input(self, matrix_str: str) -> Optional[np.ndarray]:
        """Parse matrix input from string format"""
        try:
            # Remove extra whitespace and split by lines
            lines = [line.strip() for line in matrix_str.strip().split('\n') if line.strip()]
            
            matrix_rows = []
            for line in lines:
                # Split by comma and convert to float
                row = [float(x.strip()) for x in line.split(',')]
                matrix_rows.append(row)
            
            return np.array(matrix_rows)
        except Exception as e:
            st.error(f"Error parsing matrix: {e}")
            return None
    
    def clear_transpose_matrices(self):
        """Clear any previously computed transpose matrices"""
        keys_to_remove = [key for key in self.matrices.keys() if '_T' in key]
        for key in keys_to_remove:
            del self.matrices[key]
    
    def evaluate_expression(self, expression: str) -> Tuple[Optional[np.ndarray], str, bool]:
        """
        Evaluate a matrix expression and return result, explanation, and whether it exists
        Returns: (result_matrix, explanation, exists)
        """
        expression = expression.strip()
        
        # Clear any previous transpose matrices to avoid conflicts
        self.clear_transpose_matrices()
        
        # Check for transpose operations (A|, B|, C|)
        expression_with_transposes = expression
        original_matrices = list(self.matrices.keys())  # Create a copy of keys to avoid iteration issues
        for var in original_matrices:
            if '_T' not in var:  # Only process original matrices, not transposes
                pattern = f"{var}\\|"
                if re.search(pattern, expression):
                    expression_with_transposes = re.sub(pattern, f"{var}_T", expression_with_transposes)
        
        try:
            # Replace matrix variables with actual computation
            result_expr = expression_with_transposes
            explanation_parts = []
            
            # Handle transpose operations first
            for var in original_matrices:
                if '_T' not in var and f"{var}_T" in result_expr:
                    transposed = self.matrices[var].T
                    self.matrices[f"{var}_T"] = transposed
                    explanation_parts.append(f"{var}^T = {self._format_matrix_to_latex_string(transposed)}")
            
            # Parse the expression step by step
            if "+" in result_expr:
                parts = result_expr.split("+")
                left_var = parts[0].strip()
                right_var = parts[1].strip()
                
                if left_var in self.matrices and right_var in self.matrices:
                    left_matrix = self.matrices[left_var]
                    right_matrix = self.matrices[right_var]
                    
                    if left_matrix.shape != right_matrix.shape:
                        return None, f"Cannot add matrices with different dimensions: {left_matrix.shape} vs {right_matrix.shape}", False
                    
                    result = left_matrix + right_matrix
                    explanation = f"{expression} = {self._format_matrix_to_latex_string(left_matrix)} + {self._format_matrix_to_latex_string(right_matrix)} = {self._format_matrix_to_latex_string(result)}"
                    return result, explanation, True
                    
            elif "-" in result_expr and not result_expr.startswith("-"):
                parts = result_expr.split("-")
                left_var = parts[0].strip()
                right_var = parts[1].strip()
                
                if left_var in self.matrices and right_var in self.matrices:
                    left_matrix = self.matrices[left_var]
                    right_matrix = self.matrices[right_var]
                    
                    if left_matrix.shape != right_matrix.shape:
                        return None, f"Cannot subtract matrices with different dimensions: {left_matrix.shape} vs {right_matrix.shape}", False
                    
                    result = left_matrix - right_matrix
                    explanation = f"{expression} = {self._format_matrix_to_latex_string(left_matrix)} - {self._format_matrix_to_latex_string(right_matrix)} = {self._format_matrix_to_latex_string(result)}"
                    return result, explanation, True
                    
            elif "*" in result_expr or "×" in result_expr:
                # Handle multiplication (use both * and × symbols)
                separator = "*" if "*" in result_expr else "×"
                parts = result_expr.split(separator)
                left_var = parts[0].strip()
                right_var = parts[1].strip()
                
                if left_var in self.matrices and right_var in self.matrices:
                    left_matrix = self.matrices[left_var]
                    right_matrix = self.matrices[right_var]
                    
                    if left_matrix.shape[1] != right_matrix.shape[0]:
                        return None, f"Cannot multiply matrices: {left_matrix.shape[1]} columns ≠ {right_matrix.shape[0]} rows", False
                    
                    result = left_matrix @ right_matrix
                    explanation = f"{expression} = {self._format_matrix_to_latex_string(left_matrix)} × {self._format_matrix_to_latex_string(right_matrix)} = {self._format_matrix_to_latex_string(result)}"
                    return result, explanation, True
            
            # If it's just a single variable or transpose
            if result_expr in self.matrices:
                result = self.matrices[result_expr]
                explanation = f"{expression} = {self._format_matrix_to_latex_string(result)}"
                return result, explanation, True
                
            return None, f"Unable to parse expression: {expression}", False
            
        except Exception as e:
            return None, f"Error evaluating {expression}: {str(e)}", False
    
    def render_batch_calculator(self):
        """Render the batch matrix calculator interface"""
        st.markdown('<h2 class="animate-subheader">Batch Matrix Expression Calculator</h2>', unsafe_allow_html=True)
        
        st.write("Enter matrices and expressions to evaluate multiple calculations automatically, perfect for multiple calculations!")
        
        # Matrix input section
        st.subheader("Define Matrices")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Matrix A:**")
            matrix_a_input = st.text_area(
                "Enter Matrix A",
                value="1, -2\n-1, -3\n-2, -1",
                key="batch_matrix_a",
                help="Format: comma-separated values, one row per line"
            )
            
        with col2:
            st.write("**Matrix B:**")
            matrix_b_input = st.text_area(
                "Enter Matrix B",
                value="5, 0\n0, 1\n3, 0\n0, 2",
                key="batch_matrix_b"
            )
            
        with col3:
            st.write("**Matrix C:**")
            matrix_c_input = st.text_area(
                "Enter Matrix C",
                value="3\n-2",
                key="batch_matrix_c"
            )
        
        # Parse matrices
        if matrix_a_input:
            self.matrices['A'] = self.parse_matrix_input(matrix_a_input)
        if matrix_b_input:
            self.matrices['B'] = self.parse_matrix_input(matrix_b_input)
        if matrix_c_input:
            self.matrices['C'] = self.parse_matrix_input(matrix_c_input)
        
        # Display parsed matrices
        if self.matrices:
            st.subheader("Parsed Matrices")
            cols = st.columns(len(self.matrices))
            for i, (name, matrix) in enumerate(self.matrices.items()):
                if matrix is not None and '_T' not in name:  # Don't show transpose matrices in this section
                    with cols[i]:
                        st.write(f"**{name}** ({matrix.shape[0]}×{matrix.shape[1]}):")
                        st.latex(self._format_matrix_to_latex_string(matrix))
        
        # Expression input section
        st.subheader("Matrix Expressions to Evaluate")
        
        expressions_input = st.text_area(
            "Enter expressions (one per line)",
            value="A - A|\nA * C\nB| * B\nA| * B\nB * C|",
            key="batch_expressions",
            help="Use | for transpose (e.g., A| means A transpose). Supported operations: +, -, *, ×"
        )
        
        # Evaluate button
        if st.button("Evaluate All Expressions", key="evaluate_batch"):
            if expressions_input and self.matrices:
                expressions = [expr.strip() for expr in expressions_input.split('\n') if expr.strip()]
                
                st.subheader("Results")
                
                for i, expression in enumerate(expressions, 1):
                    st.write(f"**({chr(96+i)}) {expression}**")
                    
                    result, explanation, exists = self.evaluate_expression(expression)
                    
                    if exists and result is not None:
                        st.success("✅ Expression exists and can be calculated")
                        st.write("**Result:**")
                        st.latex(self._format_matrix_to_latex_string(result))
                        
                        # Show calculation steps
                        st.write(f"**Calculation for {expression}:**")
                        st.latex(explanation)
                            
                    else:
                        st.error("❌ Expression does not exist")
                        st.write(f"**Reason:** {explanation}")
                    
                    st.markdown("---")
            else:
                st.error("Please enter both matrices and expressions to evaluate.")
        
        # Help section
        with st.expander("Help: Expression Syntax"):
            st.write("""
            **Supported Operations:**
            - `A + B` - Matrix addition
            - `A - B` - Matrix subtraction  
            - `A * B` or `A × B` - Matrix multiplication
            - `A|` - Matrix transpose (A^T)
            
            **Example Expressions:**
            - `A - A|` - A minus A transpose
            - `A * C` - A times C
            - `B| * B` - B transpose times B
            - `A| * B` - A transpose times B
            - `B * C|` - B times C transpose
            
            **Matrix Input Format:**
            - Separate elements with commas
            - Separate rows with newlines
            - Example: `1, 2\n3, 4` represents [[1,2], [3,4]]
            """)