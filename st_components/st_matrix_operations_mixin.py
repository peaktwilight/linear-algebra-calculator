#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Matrix Operations Mixin
"""

import streamlit as st
import numpy as np
import pandas as pd # Keep for potential future use with matrix display
import plotly.express as px # Keep for potential future use
import plotly.graph_objects as go # Keep for potential future use
from given_reference.core import mrref # For displaying RREF steps
from .st_visualization_utils import display_matrix_heatmap
from .st_utils import StreamOutput

class MatrixOperationsMixin:
    def matrix_operations(self, operation, matrix_a_input, matrix_b_input=None, scalar_input=None):
        """Perform basic matrix operations like add, subtract, scalar multiply, transpose."""
        
        class Args:
            def __init__(self, operation, matrix_a, matrix_b=None, scalar=None):
                self.operation = operation
                self.matrix_a = matrix_a
                self.matrix_b = matrix_b
                self.scalar = scalar

        args = Args(operation, matrix_a_input, matrix_b_input, scalar_input)

        st.subheader(f"Result for Matrix {operation.replace('_', ' ').title()}")
        
        try:
            # It's crucial that self.framework is available from the main class
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None

            matrix_a = self.framework.parse_matrix(matrix_a_input)
            matrix_b = self.framework.parse_matrix(matrix_b_input) if matrix_b_input else None
            scalar = float(scalar_input) if scalar_input is not None else None

            result = None
            # The actual computation is done by the framework's matrix_operations
            # This Streamlit method is primarily for UI and result display.
            
            # We'll call the framework's method and capture its print output if necessary,
            # but for matrix_operations, the framework returns the result directly.
            # We need to replicate the display logic here.

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Calculation Summary")
                st.markdown(f"**Operation:** {operation.replace('_', ' ').title()}")
                st.markdown("**Matrix A:**")
                st.text(str(matrix_a))
                if matrix_b is not None and operation in ["add", "subtract"]:
                    st.markdown("**Matrix B:**")
                    st.text(str(matrix_b))
                if scalar is not None and operation == "multiply_scalar":
                    st.markdown(f"**Scalar:** {scalar}")

                # Perform operation using a simplified direct call for result
                # The framework.matrix_operations in linalg_cli.py handles the core logic
                # and returns the result matrix.
                if args.operation == "add":
                    if matrix_a.shape != matrix_b.shape:
                        st.error("Matrices must have the same dimensions for addition.")
                        return
                    result = matrix_a + matrix_b
                    st.markdown("**A + B =**")
                elif args.operation == "subtract":
                    if matrix_a.shape != matrix_b.shape:
                        st.error("Matrices must have the same dimensions for subtraction.")
                        return
                    result = matrix_a - matrix_b
                    st.markdown("**A - B =**")
                elif args.operation == "multiply_scalar":
                    result = scalar * matrix_a
                    st.markdown(f"**{scalar} * A =**")
                elif args.operation == "transpose":
                    result = matrix_a.T
                    st.markdown("**A<sup>T</sup> =**")
                else:
                    st.error(f"Unknown operation: {args.operation}")
                    return

                if result is not None:
                    st.text(str(result))

            with col2:
                st.markdown("### Visualization")
                tabs_needed = []
                if matrix_a is not None: tabs_needed.append("Matrix A")
                if matrix_b is not None and operation in ["add", "subtract"]: tabs_needed.append("Matrix B")
                if result is not None: tabs_needed.append("Result")

                if len(tabs_needed) > 0:
                    tabs = st.tabs(tabs_needed)
                    tab_idx = 0
                    if matrix_a is not None:
                        with tabs[tab_idx]:
                            display_matrix_heatmap(matrix_a, title="Matrix A")
                        tab_idx+=1
                    if matrix_b is not None and operation in ["add", "subtract"]:
                        with tabs[tab_idx]:
                            display_matrix_heatmap(matrix_b, title="Matrix B")
                        tab_idx+=1
                    if result is not None:
                        with tabs[tab_idx]:
                            op_display_name = operation.replace("_", " ").title()
                            display_matrix_heatmap(result, title=f"Result: {op_display_name}")
                        tab_idx+=1
                else:
                    st.info("No matrices to visualize for this operation yet.")
            
            return result

        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        return None

    def solve_gauss(self, matrix_input):
        """Solve a system of linear equations using Gaussian elimination."""
        class Args:
            def __init__(self, matrix):
                self.matrix = matrix
        
        args = Args(matrix_input)
        st.subheader("Gaussian Elimination Result")

        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None

            augmented_matrix = self.framework.parse_matrix(args.matrix)

            # Capture print output from the CLI function for detailed steps
            with StreamOutput() as captured_output:
                # The framework's solve_gauss returns the solution if one exists,
                # and prints steps. We want to display those steps.
                solution = self.framework.solve_gauss(args) 
            
            cli_steps = captured_output.get_output()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### System & Steps")
                
                # LaTeX version of the augmented matrix with clearer title
                st.markdown("### Original System of Equations")
                st.markdown("**Augmented Matrix [A|b]:**")
                
                # Generate LaTeX for augmented matrix
                matrix_latex = self._matrix_to_latex(augmented_matrix)
                # Apply scrolling if matrix is large
                if augmented_matrix.shape[0] > 5 or augmented_matrix.shape[1] > 5:
                    st.markdown(f"""
                    <div style="max-height: 200px; overflow-y: auto;">
                        ${matrix_latex}$
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"$${matrix_latex}$$")
                
                # Extract and display step-by-step elimination process
                st.markdown("### Step-by-Step Elimination")
                
                # Parse the row echelon form from CLI output
                row_echelon = None
                reduced_matrix = mrref(np.copy(augmented_matrix))  # Use a copy
                
                # Create step-by-step explanation with LaTeX
                m, n = augmented_matrix.shape
                A = augmented_matrix.copy()
                steps_markdown = []
                
                # Perform step-by-step elimination to show the process
                for i in range(min(m, n-1)):  # For each pivot position
                    # Find maximum element in column i
                    pivot_row = i
                    for k in range(i+1, m):
                        if abs(A[k, i]) > abs(A[pivot_row, i]):
                            pivot_row = k
                    
                    # If needed, swap rows
                    if pivot_row != i:
                        A[[i, pivot_row]] = A[[pivot_row, i]]
                        steps_markdown.append(f"**Step {len(steps_markdown)+1}: Swap rows {i+1} and {pivot_row+1}**")
                        steps_markdown.append(f"$${self._matrix_to_latex(A)}$$")
                    
                    # Skip if pivot is zero (column already eliminated)
                    if abs(A[i, i]) < 1e-10:
                        continue
                    
                    # Eliminate below pivot
                    for j in range(i+1, m):
                        if abs(A[j, i]) < 1e-10:  # Skip if element is already zero
                            continue
                            
                        factor = A[j, i] / A[i, i]
                        # Format factor to be cleaner
                        factor_formatted = str(factor)
                        if np.isclose(factor, round(factor), atol=1e-10):
                            factor_formatted = str(int(round(factor)))
                        elif not np.isclose(factor, 0, atol=1e-10):
                            factor_formatted = f"{factor:.4f}"
                            
                        elimination_step = f"**Step {len(steps_markdown)+1}: Replace $R_{j+1} \\leftarrow R_{j+1} - ({factor_formatted}) \\cdot R_{i+1}$**"
                        A[j] = A[j] - factor * A[i]
                        steps_markdown.append(elimination_step)
                        steps_markdown.append(f"$${self._matrix_to_latex(A)}$$")
                
                # Display back-substitution if we have triangular form
                steps_markdown.append("**Back Substitution to find solution:**")
                
                # Show each back-substitution step in LaTeX format
                is_triangular = True
                for i in range(m):
                    for j in range(i):
                        if abs(A[i,j]) > 1e-10:
                            is_triangular = False
                            break
                
                if is_triangular:
                    # Perform back-substitution
                    x = np.zeros(n-1)
                    valid_solution = True
                    
                    for i in range(m-1, -1, -1):  # Start from bottom row
                        # Find first non-zero element in row i (pivot)
                        pivot_col = -1
                        for j in range(n-1):
                            if abs(A[i, j]) > 1e-10:
                                pivot_col = j
                                break
                        
                        if pivot_col == -1:  # No pivot in row
                            if abs(A[i, -1]) > 1e-10:  # But has non-zero RHS
                                # Format the right-hand side
                                rhs_formatted = str(A[i, -1])
                                if np.isclose(A[i, -1], round(A[i, -1]), atol=1e-10):
                                    rhs_formatted = str(int(round(A[i, -1])))
                                else:
                                    rhs_formatted = f"{A[i, -1]:.4f}"
                                    
                                steps_markdown.append(f"$$0 = {rhs_formatted}$$ <span style='color:red;font-weight:bold;'>(inconsistent system)</span>")
                                valid_solution = False
                                break
                        else:  # Pivot exists
                            # Compute x_j = (b_i - sum(a_ik * x_k)) / a_ij
                            rhs_value = A[i, -1]
                            for j in range(pivot_col+1, n-1):
                                if abs(A[i, j]) > 1e-10:  # Skip if coefficient is zero
                                    rhs_value -= A[i, j] * x[j]
                            
                            x[pivot_col] = rhs_value / A[i, pivot_col]
                            eq_parts = []
                            # Check for close to zero or integer
                            rhs_formatted = str(A[i, -1])
                            if np.isclose(A[i, -1], 0, atol=1e-10):
                                rhs_formatted = "0"
                            elif np.isclose(A[i, -1], round(A[i, -1]), atol=1e-10):
                                rhs_formatted = str(int(round(A[i, -1])))
                            else:
                                rhs_formatted = f"{A[i, -1]:.4f}"
                                
                            eq_parts.append(f"x_{pivot_col+1} = \\frac{{{rhs_formatted}")
                            
                            for j in range(pivot_col+1, n-1):
                                if abs(A[i, j]) > 1e-10:  # Skip if coefficient is zero
                                    # Format coefficient
                                    coef_formatted = str(A[i, j])
                                    if np.isclose(A[i, j], 0, atol=1e-10):
                                        continue  # Skip zeros
                                    elif np.isclose(A[i, j], round(A[i, j]), atol=1e-10):
                                        coef_formatted = str(int(round(A[i, j])))
                                    else:
                                        coef_formatted = f"{A[i, j]:.4f}"
                                    
                                    eq_parts.append(f" - ({coef_formatted}) \\cdot x_{j+1}")
                            
                            # Format denominator
                            denom_formatted = str(A[i, pivot_col])
                            if np.isclose(A[i, pivot_col], round(A[i, pivot_col]), atol=1e-10):
                                denom_formatted = str(int(round(A[i, pivot_col])))
                            else:
                                denom_formatted = f"{A[i, pivot_col]:.4f}"
                                
                            # Format result
                            result_formatted = str(x[pivot_col])
                            if np.isclose(x[pivot_col], 0, atol=1e-10):
                                result_formatted = "0"
                            elif np.isclose(x[pivot_col], round(x[pivot_col]), atol=1e-10):
                                result_formatted = str(int(round(x[pivot_col])))
                            else:
                                result_formatted = f"{x[pivot_col]:.4f}"
                                
                            eq_parts.append(f"}}{{{denom_formatted}}} = {result_formatted}")
                            steps_markdown.append("$$" + "".join(eq_parts) + "$$")
                
                # Display all steps directly
                with st.container():
                    if len(steps_markdown) > 6:
                        # Create an expander for steps if there are many
                        with st.expander("View All Steps (click to expand)", expanded=True):
                            for step in steps_markdown:
                                st.markdown(step)
                    else:
                        # If only a few steps, show directly
                        for step in steps_markdown:
                            st.markdown(step)
                
                # Display CLI steps as reference (collapsed)
                with st.expander("CLI Solver Output (Reference)"):
                    st.text_area("Solver Steps", value=cli_steps, height=300, disabled=True)

                st.markdown("**Solution:**")
                if solution is None:
                    # Check cli_steps for inconsistency message as framework prints it
                    if "inconsistent" in cli_steps.lower() or "no solution" in cli_steps.lower():
                         st.warning("The system is inconsistent (no solution).")
                    else:
                         st.info("The framework did not return a solution (it might be inconsistent or an issue occurred).")
                elif isinstance(solution, dict) and "particular" in solution and "null_space" in solution:
                    st.markdown("The system has infinitely many solutions.")
                    
                    # Format particular solution for LaTeX
                    particular = solution["particular"]
                    null_space = solution["null_space"]
                    
                    # Format the particular solution part
                    st.markdown("**Particular Solution:**")
                    particular_latex = "$$\\mathbf{x}_p = " + self._vector_to_latex(particular) + "$$"
                    st.markdown(particular_latex)
                    
                    # Format the null space part
                    st.markdown("**Null Space (for general solution):**")
                    null_space_latex = "$$\mathbf{x}_n = t \cdot " + self._vector_to_latex(null_space.flatten()) + "$$"
                    st.markdown(null_space_latex)
                    
                    # Format the general solution
                    st.markdown("**General Solution:**")
                    general_solution = "$$\mathbf{x} = \mathbf{x}_p + \mathbf{x}_n = " + self._vector_to_latex(particular) + " + t \cdot " + self._vector_to_latex(null_space.flatten()) + "$$"
                    st.markdown(general_solution)
                    
                elif isinstance(solution, np.ndarray):
                    st.markdown("The system has a unique solution:")
                    # Format the solution vector for LaTeX display
                    formatted_latex = "$$\mathbf{x} = " + self._vector_to_latex(solution) + "$$"
                    st.markdown(formatted_latex)
                else: # Should not happen if framework.solve_gauss behaves as expected
                    st.text(str(solution))

            with col2:
                st.markdown("### Visualizations")
                # RREF from CLI output (if available and parsable) or re-calculate for display
                # For simplicity, we'll use the framework's mrref directly for visualization.
                reduced_matrix = mrref(np.copy(augmented_matrix)) # Use a copy

                tab1, tab2 = st.tabs(["Original Augmented Matrix", "Reduced Row Echelon Form"])
                with tab1:
                    display_matrix_heatmap(augmented_matrix, title="Original Augmented Matrix [A|b]")
                with tab2:
                    display_matrix_heatmap(reduced_matrix, title="Reduced Row Echelon Form (RREF)")
            
            return solution

        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred during Gaussian elimination: {e}")
        return None
        
    def _matrix_to_latex(self, matrix):
        """Convert a numpy matrix to LaTeX format."""
        rows, cols = matrix.shape
        latex = "\\begin{bmatrix}"
        
        for i in range(rows):
            row_str = []
            for j in range(cols):
                # Format the number with appropriate precision
                if np.isclose(matrix[i, j], 0, atol=1e-10):
                    row_str.append("0")
                elif np.isclose(matrix[i, j], round(matrix[i, j]), atol=1e-10):
                    # It's close to an integer
                    row_str.append(f"{int(round(matrix[i, j]))}")
                else:
                    # Use decimal format with 4 decimal places
                    row_str.append(f"{matrix[i, j]:.4f}")
            
            # Join elements of this row with ampersands
            latex += " & ".join(row_str)
            
            # Add line break if not the last row
            if i < rows - 1:
                latex += "\\\\"
        
        latex += "\\end{bmatrix}"
        return latex
    
    def _vector_to_latex(self, vector):
        """Convert a numpy vector to LaTeX format."""
        elements = []
        for x in vector:
            if np.isclose(x, 0, atol=1e-10):
                elements.append("0")
            elif np.isclose(x, round(x), atol=1e-10):
                # It's close to an integer
                elements.append(f"{int(round(x))}")
            else:
                # Use decimal format with 4 decimal places
                elements.append(f"{x:.4f}")
        
        return "\\begin{bmatrix} " + " \\\\ ".join(elements) + " \\end{bmatrix}" 