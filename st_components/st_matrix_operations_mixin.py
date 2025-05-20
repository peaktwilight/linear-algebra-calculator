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
import scipy.linalg as sp
from given_reference.core import mrref # For displaying RREF steps
from .st_visualization_utils import display_matrix_heatmap
from .st_utils import StreamOutput

class MatrixOperationsMixin:
    """
    Mixin class that provides matrix operation methods for the Streamlit interface.
    This class contains methods that display and process matrix operations with visualization.
    """
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
        
    def standard_form(self, matrix_input, equations_input=None):
        """
        Convert a system of linear equations to standard form and provide educational content.
        
        This method takes either an augmented matrix or a set of equations in string form,
        converts it to standard form (Ax = b), and provides educational visualization.
        """
        import re  # Import for regex pattern matching
        
        st.subheader("Standard Form of Linear System")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the matrix if provided
            if matrix_input:
                augmented_matrix = self.framework.parse_matrix(matrix_input)
                matrix_provided = True
            else:
                matrix_provided = False
            
            # If equations are provided, parse them into a matrix
            equations_parsed = None
            if equations_input:
                try:
                    equations_parsed = self._parse_equations(equations_input)
                    if not matrix_provided:
                        augmented_matrix = equations_parsed
                except Exception as e:
                    st.error(f"Error parsing equations: {e}")
                    if not matrix_provided:
                        return None
            
            # Split into coefficient matrix A and right-hand side b
            A = augmented_matrix[:, :-1]
            b = augmented_matrix[:, -1:]
            
            # Display the results in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original System")
                
                # Display the augmented matrix [A|b]
                st.markdown("**Augmented Matrix Form [A|b]:**")
                st.markdown(f"$${self._matrix_to_latex(augmented_matrix)}$$")
                
                # If equations were parsed or provided, show them too
                if equations_parsed is not None or equations_input:
                    st.markdown("**Equation Form:**")
                    
                    equations_display = []
                    for i in range(A.shape[0]):
                        eq = []
                        for j in range(A.shape[1]):
                            coef = A[i, j]
                            if np.isclose(coef, 0, atol=1e-10):
                                continue
                            
                            # Format the coefficient
                            if j > 0 and coef > 0:
                                eq.append("+")
                                
                            if np.isclose(abs(coef), 1, atol=1e-10):
                                # If coefficient is 1 or -1, don't show the number
                                if coef < 0:
                                    eq.append("-")
                            else:
                                eq.append(f"{coef:.4g}")
                                
                            # Add the variable
                            eq.append(f"x_{{{j+1}}}")
                        
                        # Add the equals sign and right-hand side
                        rhs = b[i, 0]
                        eq.append("=")
                        eq.append(f"{rhs:.4g}")
                        
                        equations_display.append("$" + " ".join(eq) + "$")
                    
                    for eq in equations_display:
                        st.markdown(eq)
                
                # Display Standard Form: Ax = b
                st.markdown("### Standard Form (Ax = b)")
                
                # Display A
                st.markdown("**Coefficient Matrix A:**")
                st.markdown(f"$$A = {self._matrix_to_latex(A)}$$")
                
                # Display x (as a vector of variables)
                var_vector = "\\begin{bmatrix} x_1 \\\\ x_2 \\\\ \\vdots \\\\ x_n \\end{bmatrix}"
                if A.shape[1] <= 10:  # If not too many variables, show them specifically
                    var_vector = "\\begin{bmatrix} " + " \\\\ ".join([f"x_{i+1}" for i in range(A.shape[1])]) + " \\end{bmatrix}"
                
                st.markdown("**Variable Vector x:**")
                st.markdown(f"$$x = {var_vector}$$")
                
                # Display b
                st.markdown("**Right-hand Side b:**")
                st.markdown(f"$$b = {self._vector_to_latex(b)}$$")
                
                # Display the complete system Ax = b
                st.markdown("**Complete System Ax = b:**")
                st.markdown(f"$${self._matrix_to_latex(A)} \\cdot {var_vector} = {self._vector_to_latex(b)}$$")
            
            with col2:
                st.markdown("### System Properties")
                
                # Number of equations and variables
                m, n = A.shape
                st.markdown(f"**Number of Equations (m):** {m}")
                st.markdown(f"**Number of Variables (n):** {n}")
                
                # System classification
                if m == n:
                    system_type = "Square System (m = n)"
                    st.success(system_type)
                    st.markdown("- A square system has exactly one solution if det(A) ≠ 0")
                    st.markdown("- If det(A) = 0, the system is either inconsistent or has infinitely many solutions")
                elif m < n:
                    system_type = "Underdetermined System (m < n)"
                    st.info(system_type)
                    st.markdown("- An underdetermined system has either no solutions or infinitely many solutions")
                    st.markdown("- If consistent, the system has infinitely many solutions with (n-r) free parameters, where r is the rank of A")
                else:  # m > n
                    system_type = "Overdetermined System (m > n)"
                    st.warning(system_type)
                    st.markdown("- An overdetermined system typically has no exact solution")
                    st.markdown("- An approximate solution can be found using methods like least squares")
                
                # Calculate and display the rank of A
                rank_A = np.linalg.matrix_rank(A)
                rank_Ab = np.linalg.matrix_rank(augmented_matrix)
                
                st.markdown(f"**Rank of A:** {rank_A}")
                st.markdown(f"**Rank of [A|b]:** {rank_Ab}")
                
                # Consistency check
                if rank_A == rank_Ab:
                    st.success("The system is consistent (has at least one solution)")
                    if rank_A < n:
                        free_params = n - rank_A
                        st.markdown(f"The system has infinitely many solutions with {free_params} free parameter(s)")
                    elif rank_A == n:
                        st.markdown("The system has a unique solution")
                else:
                    st.error("The system is inconsistent (has no solution)")
                
                # Check if the system is homogeneous
                is_homogeneous = np.allclose(b, 0, atol=1e-10)
                if is_homogeneous:
                    st.info("This is a homogeneous system (b = 0)")
                    st.markdown("- A homogeneous system always has the trivial solution (x = 0)")
                    st.markdown("- If rank(A) < n, it also has infinitely many non-trivial solutions")
                else:
                    st.info("This is a non-homogeneous system (b ≠ 0)")
                
                # Visualizations based on size
                st.markdown("### Visualizations")
                
                # Heat map of the coefficient matrix
                st.markdown("**Coefficient Matrix Heatmap:**")
                display_matrix_heatmap(A, title="Coefficient Matrix A")
            
            return {
                "A": A,
                "b": b,
                "rank_A": rank_A,
                "rank_Ab": rank_Ab,
                "is_consistent": rank_A == rank_Ab,
                "is_homogeneous": is_homogeneous,
                "system_type": system_type
            }
                
        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        return None
        
    def _parse_equations(self, equations_input):
        """
        Parse a list of linear equations into an augmented matrix.
        
        Example input:
        "2x1 - 3x2 + 5x3 = 7
         4x1 + 2x2 - x3 = -3
         x1 + x2 + x3 = 0"
        """
        import re  # Import for regex pattern matching
        
        # Split the input into individual equations
        equations = [eq.strip() for eq in equations_input.split('\n') if eq.strip()]
        
        # Process each equation into coefficients and constants
        processed_equations = []
        max_var_index = 0
        
        for eq in equations:
            # Split the equation at the equals sign
            if '=' not in eq:
                raise ValueError(f"Equation '{eq}' is missing an equals sign.")
                
            lhs, rhs = eq.split('=', 1)
            
            # Parse the right-hand side to get the constant
            try:
                constant = float(rhs.strip())
            except ValueError:
                raise ValueError(f"Right-hand side '{rhs}' must be a number.")
            
            # Parse the left-hand side to get the coefficients
            terms = re.findall(r'([+\-]?\s*\d*\.?\d*)\s*x_?(\d+)', lhs)
            
            if not terms:
                raise ValueError(f"No valid terms found in '{lhs}'. Use format like '2x1 - 3x2 = 5'")
            
            # Process the terms into (coefficient, variable_index) pairs
            coefficients = {}
            for coef_str, var_idx_str in terms:
                # Get the variable index (1-based)
                var_idx = int(var_idx_str)
                max_var_index = max(max_var_index, var_idx)
                
                # Get the coefficient
                coef_str = coef_str.strip()
                if coef_str in ('', '+'):
                    coef = 1.0
                elif coef_str == '-':
                    coef = -1.0
                else:
                    coef = float(coef_str)
                
                # Add to the dict, combining any duplicate variables
                coefficients[var_idx] = coefficients.get(var_idx, 0.0) + coef
            
            processed_equations.append((coefficients, constant))
        
        # Create the augmented matrix
        num_equations = len(processed_equations)
        num_variables = max_var_index
        augmented = np.zeros((num_equations, num_variables + 1))
        
        # Fill in the matrix
        for i, (coeffs, constant) in enumerate(processed_equations):
            for var_idx, coef in coeffs.items():
                augmented[i, var_idx - 1] = coef  # Adjust for 0-based indexing
            augmented[i, -1] = constant
        
        return augmented
        
    def row_operations_analysis(self, matrix_input):
        """
        Perform step-by-step row operations on a matrix to obtain row echelon form,
        with detailed explanations of each step.
        """
        st.subheader("Row Operations Analysis")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the matrix
            matrix = self.framework.parse_matrix(matrix_input)
            
            # Display the original matrix
            st.markdown("### Original Matrix")
            st.markdown(f"$${self._matrix_to_latex(matrix)}$$")
            
            # Perform step-by-step row reduction
            m, n = matrix.shape
            A = matrix.copy()
            steps = []
            op_count = 0
            
            # Track the row operations performed to generate the transformation matrices
            row_operations = []
            
            # Step 1: Forward elimination to get row echelon form
            for i in range(min(m, n)):
                # Find the pivot row (row with largest absolute value in the current column)
                pivot_row = i
                for k in range(i+1, m):
                    if abs(A[k, i]) > abs(A[pivot_row, i]):
                        pivot_row = k
                
                # If the pivot is zero, move to the next column
                if abs(A[pivot_row, i]) < 1e-10:
                    continue
                
                # Swap rows if necessary
                if pivot_row != i:
                    A[[i, pivot_row]] = A[[pivot_row, i]]
                    op_count += 1
                    
                    # Record the row operation
                    swap_op = {
                        "type": "swap",
                        "rows": (i+1, pivot_row+1),
                        "description": f"Swap rows {i+1} and {pivot_row+1}",
                        "matrix_before": A.copy(),
                        "matrix_after": A.copy()
                    }
                    row_operations.append(swap_op)
                    
                    # Add to step history
                    steps.append({
                        "operation": f"Swap rows {i+1} and {pivot_row+1}",
                        "latex": self._matrix_to_latex(A),
                        "explanation": f"Swapping rows {i+1} and {pivot_row+1} to bring the largest element to the pivot position."
                    })
                
                # Eliminate below the pivot
                pivot = A[i, i]
                for j in range(i+1, m):
                    if abs(A[j, i]) < 1e-10:  # Skip if already zero
                        continue
                        
                    factor = A[j, i] / pivot
                    
                    # Format factor for display
                    if np.isclose(factor, round(factor), atol=1e-10):
                        factor_display = str(int(round(factor)))
                    else:
                        factor_display = f"{factor:.4f}"
                    
                    # Record this row operation
                    op_count += 1
                    elim_op = {
                        "type": "elimination",
                        "pivot_row": i+1,
                        "target_row": j+1,
                        "factor": factor,
                        "matrix_before": A.copy()
                    }
                    
                    # Perform the elimination
                    A[j] = A[j] - factor * A[i]
                    
                    elim_op["matrix_after"] = A.copy()
                    row_operations.append(elim_op)
                    
                    # Add to step history
                    steps.append({
                        "operation": f"R{j+1} ← R{j+1} - {factor_display}·R{i+1}",
                        "latex": self._matrix_to_latex(A),
                        "explanation": f"Eliminating the element at position ({j+1}, {i+1}) by subtracting {factor_display} times row {i+1} from row {j+1}."
                    })
            
            # Step 2: Back substitution to get reduced row echelon form
            for i in range(min(m, n)-1, -1, -1):
                # Find the pivot column
                pivot_col = -1
                for j in range(n):
                    if abs(A[i, j]) > 1e-10:
                        pivot_col = j
                        break
                        
                if pivot_col == -1:
                    continue  # Skip if row is all zeros
                
                # Normalize the pivot row
                pivot = A[i, pivot_col]
                if not np.isclose(pivot, 1.0, atol=1e-10):
                    # Record this row operation
                    op_count += 1
                    norm_op = {
                        "type": "scaling",
                        "row": i+1,
                        "factor": 1.0/pivot,
                        "matrix_before": A.copy()
                    }
                    
                    A[i] = A[i] / pivot
                    
                    norm_op["matrix_after"] = A.copy()
                    row_operations.append(norm_op)
                    
                    # Format for display
                    if np.isclose(pivot, round(pivot), atol=1e-10):
                        pivot_display = str(int(round(pivot)))
                    else:
                        pivot_display = f"{pivot:.4f}"
                    
                    # Add to step history
                    steps.append({
                        "operation": f"R{i+1} ← R{i+1} / {pivot_display}",
                        "latex": self._matrix_to_latex(A),
                        "explanation": f"Normalizing row {i+1} by dividing by {pivot_display} to make the pivot element 1."
                    })
                
                # Eliminate above the pivot
                for j in range(i):
                    factor = A[j, pivot_col]
                    if abs(factor) < 1e-10:  # Skip if already zero
                        continue
                    
                    # Record this row operation
                    op_count += 1
                    back_op = {
                        "type": "back_substitution",
                        "pivot_row": i+1,
                        "target_row": j+1,
                        "factor": factor,
                        "matrix_before": A.copy()
                    }
                    
                    # Perform back substitution
                    A[j] = A[j] - factor * A[i]
                    
                    back_op["matrix_after"] = A.copy()
                    row_operations.append(back_op)
                    
                    # Format for display
                    if np.isclose(factor, round(factor), atol=1e-10):
                        factor_display = str(int(round(factor)))
                    else:
                        factor_display = f"{factor:.4f}"
                    
                    # Add to step history
                    steps.append({
                        "operation": f"R{j+1} ← R{j+1} - {factor_display}·R{i+1}",
                        "latex": self._matrix_to_latex(A),
                        "explanation": f"Eliminating the element at position ({j+1}, {pivot_col+1}) by subtracting {factor_display} times row {i+1} from row {j+1}."
                    })
            
            # Display the steps
            st.markdown("### Step-by-Step Row Operations")
            st.info(f"Total operations performed: {op_count}")
            
            # Display in tabs if there are many steps
            if len(steps) > 6:
                with st.expander("View All Row Operations (click to expand)", expanded=True):
                    for i, step in enumerate(steps):
                        with st.container():
                            st.markdown(f"**Step {i+1}: {step['operation']}**")
                            st.markdown(f"$${step['latex']}$$")
                            st.markdown(f"_{step['explanation']}_")
                            st.markdown("---")
            else:
                for i, step in enumerate(steps):
                    with st.container():
                        st.markdown(f"**Step {i+1}: {step['operation']}**")
                        st.markdown(f"$${step['latex']}$$")
                        st.markdown(f"_{step['explanation']}_")
                        st.markdown("---")
            
            # Display the final results
            st.markdown("### Result: Row Echelon Form (RREF)")
            st.markdown(f"$${self._matrix_to_latex(A)}$$")
            
            # Determine the row rank and properties of the matrix
            rank = 0
            for i in range(m):
                if np.any(np.abs(A[i]) > 1e-10):  # Row is not all zeros
                    rank += 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Matrix Properties")
                st.markdown(f"**Matrix Size:** {m} × {n}")
                st.markdown(f"**Rank:** {rank}")
                
                if m == n:
                    # Calculate determinant
                    try:
                        det = np.linalg.det(matrix)
                        st.markdown(f"**Determinant:** {det:.4f}")
                        
                        if abs(det) < 1e-10:
                            st.warning("The matrix is singular (determinant ≈ 0)")
                        else:
                            st.success("The matrix is non-singular (invertible)")
                            
                    except Exception as e:
                        st.error(f"Could not calculate determinant: {e}")
                
                # For augmented matrices [A|b]
                if n > 1:
                    # Check if this might be an augmented matrix
                    st.markdown("### Linear System Analysis")
                    st.info("If this is an augmented matrix [A|b] for a linear system:")
                    
                    A_coef = matrix[:, :-1]
                    b_rhs = matrix[:, -1:]
                    
                    rank_A = np.linalg.matrix_rank(A_coef)
                    rank_Ab = np.linalg.matrix_rank(matrix)
                    
                    st.markdown(f"**Rank of coefficient matrix A:** {rank_A}")
                    st.markdown(f"**Rank of augmented matrix [A|b]:** {rank_Ab}")
                    
                    if rank_A == rank_Ab:
                        st.success("The system is consistent (has at least one solution)")
                        if rank_A < A_coef.shape[1]:
                            free_vars = A_coef.shape[1] - rank_A
                            st.markdown(f"The system has infinitely many solutions with {free_vars} free parameters")
                        else:
                            st.markdown("The system has a unique solution")
                    else:
                        st.error("The system is inconsistent (has no solution)")
            
            with col2:
                st.markdown("### Visualizations")
                st.markdown("**Heat Map of the Original Matrix:**")
                display_matrix_heatmap(matrix, title="Original Matrix")
                
                st.markdown("**Heat Map of the Row Echelon Form:**")
                display_matrix_heatmap(A, title="Row Echelon Form")
            
            return {
                "original_matrix": matrix,
                "row_echelon_form": A,
                "steps": steps,
                "rank": rank,
                "row_operations": row_operations
            }
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
        return None
        
    def free_parameter_analysis(self, matrix_input):
        """
        Analyze a linear system to identify free parameters and express solutions in parametric form.
        This is useful for systems with infinitely many solutions.
        """
        st.subheader("Free Parameter Analysis")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the augmented matrix for the linear system
            augmented_matrix = self.framework.parse_matrix(matrix_input)
            
            # Split the matrix into coefficient matrix A and RHS vector b
            A = augmented_matrix[:, :-1]
            b = augmented_matrix[:, -1:]
            
            # Display the original system
            st.markdown("### Original Linear System")
            st.markdown(f"**Augmented Matrix [A|b]:**")
            st.markdown(f"$${self._matrix_to_latex(augmented_matrix)}$$")
            
            # Compute the reduced row echelon form
            rref = mrref(np.copy(augmented_matrix))
            
            # Identify pivot and free variable columns
            m, n = A.shape
            pivot_columns = []
            free_columns = []
            row_has_pivot = [False] * m
            
            # First, determine which columns contain pivots
            for i in range(m):
                pivot_col = -1
                for j in range(n):
                    if abs(rref[i, j]) > 1e-10:
                        pivot_col = j
                        pivot_columns.append(j)
                        row_has_pivot[i] = True
                        break
                
                # Check for inconsistency
                if pivot_col == -1 and abs(rref[i, -1]) > 1e-10:
                    st.error("The system is inconsistent (no solution exists)")
                    return {"status": "inconsistent"}
            
            # Identify free variables
            for j in range(n):
                if j not in pivot_columns:
                    free_columns.append(j)
            
            # Determine if the system is consistent
            is_consistent = True
            for i in range(m):
                if np.all(abs(rref[i, :-1]) < 1e-10) and abs(rref[i, -1]) > 1e-10:
                    is_consistent = False
                    break
            
            # Display the RREF
            st.markdown("### Reduced Row Echelon Form")
            st.markdown(f"$${self._matrix_to_latex(rref)}$$")
            
            # System analysis
            rank_A = np.linalg.matrix_rank(A)
            rank_augmented = np.linalg.matrix_rank(augmented_matrix)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### System Analysis")
                st.markdown(f"**Number of Equations:** {m}")
                st.markdown(f"**Number of Variables:** {n}")
                st.markdown(f"**Rank of Coefficient Matrix A:** {rank_A}")
                st.markdown(f"**Rank of Augmented Matrix [A|b]:** {rank_augmented}")
                
                # Determine system type
                if rank_A != rank_augmented:
                    st.error("The system is inconsistent (no solution exists)")
                    solution_type = "inconsistent"
                elif rank_A == n:
                    st.success("The system has a unique solution")
                    solution_type = "unique"
                else:
                    st.info(f"The system has infinitely many solutions with {n - rank_A} free parameter(s)")
                    solution_type = "parametric"
                
                # List pivot and free variables
                st.markdown("### Variable Classification")
                
                st.markdown("**Pivot Variables:**")
                if pivot_columns:
                    for col in pivot_columns:
                        st.markdown(f"- x_{col+1}")
                else:
                    st.markdown("- None")
                
                st.markdown("**Free Variables (Parameters):**")
                if free_columns:
                    for col in free_columns:
                        st.markdown(f"- x_{col+1}")
                else:
                    st.markdown("- None")
            
            with col2:
                # Visualize the solution structure
                st.markdown("### Solution Structure")
                
                # For a unique solution, compute and display it
                if solution_type == "unique":
                    # Extract the solution from RREF
                    solution = np.zeros(n)
                    for i, col in enumerate(pivot_columns):
                        solution[col] = rref[i, -1]
                    
                    st.markdown("**Unique Solution:**")
                    solution_latex = "$$\\mathbf{x} = " + self._vector_to_latex(solution) + "$$"
                    st.markdown(solution_latex)
                    
                    # Verification
                    verification = A @ solution.reshape(-1, 1) - b
                    st.markdown("**Verification (A·x - b should be approximately 0):**")
                    st.markdown(f"$${self._vector_to_latex(verification)}$$")
                    
                    # Display solution in equation form
                    st.markdown("**Solution Values:**")
                    for i in range(n):
                        value = solution[i]
                        if np.isclose(value, 0, atol=1e-10):
                            formatted_value = "0"
                        elif np.isclose(value, round(value), atol=1e-10):
                            formatted_value = str(int(round(value)))
                        else:
                            formatted_value = f"{value:.4f}"
                        
                        st.markdown(f"$x_{i+1} = {formatted_value}$")
                
                # For parametric solutions, express in terms of free variables
                elif solution_type == "parametric":
                    st.markdown("**Parametric Solution Form:**")
                    
                    # Create mapping from pivot columns to rows
                    pivot_to_row = {}
                    for i, col in enumerate(pivot_columns):
                        pivot_to_row[col] = i
                    
                    # Generate parametric form for each variable
                    parametric_form = {}
                    
                    # For pivot variables
                    for pivot_col in pivot_columns:
                        row = pivot_to_row[pivot_col]
                        
                        # Start with the constant term
                        constant_term = rref[row, -1]
                        terms = []
                        
                        if not np.isclose(constant_term, 0, atol=1e-10):
                            if np.isclose(constant_term, round(constant_term), atol=1e-10):
                                terms.append(str(int(round(constant_term))))
                            else:
                                terms.append(f"{constant_term:.4f}")
                        
                        # Add terms for free variables
                        for free_col in free_columns:
                            coef = -rref[row, free_col]  # Negative because we're moving to the RHS
                            if not np.isclose(coef, 0, atol=1e-10):
                                if np.isclose(coef, 1, atol=1e-10):
                                    terms.append(f"t_{free_col+1}")
                                elif np.isclose(coef, -1, atol=1e-10):
                                    terms.append(f"-t_{free_col+1}")
                                else:
                                    if np.isclose(coef, round(coef), atol=1e-10):
                                        coef_str = str(int(round(coef)))
                                    else:
                                        coef_str = f"{coef:.4f}"
                                    
                                    if coef > 0:
                                        terms.append(f"{coef_str} \\cdot t_{free_col+1}")
                                    else:
                                        terms.append(f"({coef_str}) \\cdot t_{free_col+1}")
                        
                        # Handle empty terms list (all zeros)
                        if not terms:
                            terms.append("0")
                        
                        # Construct the complete expression
                        parametric_form[pivot_col] = " + ".join(terms)
                    
                    # For free variables, they are the parameters themselves
                    for free_col in free_columns:
                        parametric_form[free_col] = f"t_{free_col+1}"
                    
                    # Display the parametric form
                    st.markdown("**General Solution:**")
                    
                    # Create LaTeX for the general solution vector
                    solution_components = []
                    for i in range(n):
                        if i in parametric_form:
                            solution_components.append(parametric_form[i])
                        else:
                            solution_components.append("0")
                    
                    solution_latex = "$$\\mathbf{x} = \\begin{bmatrix} " + " \\\\ ".join(solution_components) + " \\end{bmatrix}$$"
                    st.markdown(solution_latex)
                    
                    # Display each variable's parametric form
                    st.markdown("**Solution in Terms of Parameters:**")
                    for i in range(n):
                        if i in parametric_form:
                            st.markdown(f"$x_{i+1} = {parametric_form[i]}$")
                        else:
                            st.markdown(f"$x_{i+1} = 0$")
                    
                    # Explain the parameters
                    st.markdown("**Parameters:**")
                    for free_col in free_columns:
                        st.markdown(f"- $t_{free_col+1}$ (represents the value of $x_{free_col+1}$)")
                    
                    # Example values
                    st.markdown("**Example Solutions:**")
                    st.markdown("You can generate different solutions by choosing different values for the parameters:")
                    
                    # Case 1: All parameters = 0
                    st.markdown("*Example 1: All parameters = 0*")
                    particular_solution = np.zeros(n)
                    for pivot_col in pivot_columns:
                        row = pivot_to_row[pivot_col]
                        particular_solution[pivot_col] = rref[row, -1]
                    
                    particular_latex = "$$\\mathbf{x}_p = " + self._vector_to_latex(particular_solution) + "$$"
                    st.markdown(particular_latex)
                    
                    # Case 2: One parameter = 1, others = 0
                    if len(free_columns) > 0:
                        st.markdown(f"*Example 2: $t_{free_columns[0]+1} = 1$, other parameters = 0*")
                        param_solution = np.copy(particular_solution)
                        free_col = free_columns[0]
                        param_solution[free_col] = 1
                        
                        # Update pivot variables
                        for pivot_col in pivot_columns:
                            row = pivot_to_row[pivot_col]
                            param_solution[pivot_col] -= rref[row, free_col]
                        
                        param_latex = "$$\\mathbf{x} = " + self._vector_to_latex(param_solution) + "$$"
                        st.markdown(param_latex)
                
                elif solution_type == "inconsistent":
                    st.error("The system has no solution.")
                    st.markdown("Inconsistency detected in the augmented matrix row echelon form.")
                    
                    # Identify the inconsistent equation
                    for i in range(m):
                        if np.all(abs(rref[i, :-1]) < 1e-10) and abs(rref[i, -1]) > 1e-10:
                            st.markdown(f"**Inconsistent Equation:**")
                            st.markdown(f"$0 = {rref[i, -1]:.4f}$ (from row {i+1} of RREF)")
                
                # Heat map visualization
                st.markdown("### Visualization")
                st.markdown("**Reduced Row Echelon Form with Free and Pivot Columns:**")
                
                # Create a custom colormap for visualizing pivot and free columns
                import plotly.graph_objects as go
                
                # Create a heatmap
                fig = go.Figure()
                
                z = rref.copy()
                
                # Add annotations for pivot and free columns
                annotations = []
                for i in range(m):
                    for j in range(n+1):
                        if j < n:  # Skip the RHS column
                            if j in pivot_columns:
                                annotations.append(dict(
                                    x=j, y=i,
                                    text="Pivot" if np.isclose(abs(rref[i, j]), 1, atol=1e-10) else "",
                                    showarrow=False,
                                    font=dict(size=10, color="white")
                                ))
                            elif j in free_columns:
                                annotations.append(dict(
                                    x=j, y=i,
                                    text="Free" if i == 0 else "",
                                    showarrow=False,
                                    font=dict(size=10, color="white")
                                ))
                
                # Add the heatmap
                fig.add_trace(go.Heatmap(
                    z=z,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Value")
                ))
                
                # Set layout
                fig.update_layout(
                    title="RREF with Pivot and Free Columns",
                    xaxis=dict(title="Column", tickmode="array", tickvals=list(range(n+1)), 
                              ticktext=[f"x{i+1}" for i in range(n)] + ["b"]),
                    yaxis=dict(title="Row", tickmode="array", tickvals=list(range(m)), 
                              ticktext=[f"Eq{i+1}" for i in range(m)]),
                    annotations=annotations,
                    width=600,
                    height=400,
                    margin=dict(l=50, r=50, b=50, t=70)
                )
                
                # Display the figure
                st.plotly_chart(fig)
            
            return {
                "augmented_matrix": augmented_matrix,
                "rref": rref,
                "pivot_columns": pivot_columns,
                "free_columns": free_columns,
                "solution_type": solution_type,
                "rank_A": rank_A,
                "rank_augmented": rank_augmented
            }
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
        return None
    
    def homogeneous_inhomogeneous_solutions(self, matrix_input):
        """
        Analyze homogeneous and inhomogeneous solutions for a linear system.
        This method demonstrates the relationship between solutions of Ax=b and Ax=0.
        """
        st.subheader("Homogeneous & Inhomogeneous Solutions")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the augmented matrix for the linear system Ax=b
            augmented_matrix = self.framework.parse_matrix(matrix_input)
            
            # Split into coefficient matrix A and RHS vector b
            A = augmented_matrix[:, :-1]
            b = augmented_matrix[:, -1:]
            
            # Create the homogeneous system by setting b=0
            homogeneous_matrix = np.copy(augmented_matrix)
            homogeneous_matrix[:, -1] = 0
            
            # Calculate row echelon forms
            rref_inhomogeneous = mrref(np.copy(augmented_matrix))
            rref_homogeneous = mrref(np.copy(homogeneous_matrix))
            
            # Calculate ranks
            rank_A = np.linalg.matrix_rank(A)
            rank_augmented = np.linalg.matrix_rank(augmented_matrix)
            m, n = A.shape
            
            # Display the original systems
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original Systems")
                
                st.markdown("**Inhomogeneous System (Ax = b):**")
                st.markdown(f"$${self._matrix_to_latex(augmented_matrix)}$$")
                
                st.markdown("**Homogeneous System (Ax = 0):**")
                st.markdown(f"$${self._matrix_to_latex(homogeneous_matrix)}$$")
                
                st.markdown("### System Analysis")
                
                # Consistency check for inhomogeneous
                st.markdown("**Inhomogeneous System (Ax = b) Analysis:**")
                if rank_A == rank_augmented:
                    if rank_A == n:
                        st.success("The inhomogeneous system has a unique solution.")
                        inhomo_solution_type = "unique"
                    else:
                        free_vars = n - rank_A
                        st.info(f"The inhomogeneous system has infinitely many solutions with {free_vars} free parameter(s).")
                        inhomo_solution_type = "parametric"
                else:
                    st.error("The inhomogeneous system is inconsistent (no solution).")
                    inhomo_solution_type = "inconsistent"
                
                # Analysis for homogeneous
                st.markdown("**Homogeneous System (Ax = 0) Analysis:**")
                if rank_A == n:
                    st.info("The homogeneous system has only the trivial solution (zero vector).")
                    homo_solution_type = "trivial"
                else:
                    null_dim = n - rank_A
                    st.success(f"The homogeneous system has a {null_dim}-dimensional null space (infinitely many solutions).")
                    homo_solution_type = "parametric"
                
                # Relationship
                st.markdown("### Solution Relationship")
                
                if inhomo_solution_type == "inconsistent":
                    st.error("Since the inhomogeneous system is inconsistent, there is no relationship to establish with the homogeneous solutions.")
                elif inhomo_solution_type == "unique" and homo_solution_type == "trivial":
                    st.success("The inhomogeneous system has a unique solution, and the homogeneous system has only the zero solution.")
                    st.markdown("There is a single solution for Ax = b and only the trivial solution (0) for Ax = 0.")
                elif inhomo_solution_type == "parametric" and homo_solution_type == "parametric":
                    st.success("Both systems have infinitely many solutions!")
                    st.markdown("""
                    **Key Relationship:**
                    - If x₁ and x₂ are both solutions to Ax = b, then x₁ - x₂ is a solution to Ax = 0
                    - If x₀ is a solution to Ax = b and v is a solution to Ax = 0, then x₀ + v is also a solution to Ax = b
                    
                    In other words, the general solution to Ax = b can be written as:
                    - x = x₀ + v, where:
                      - x₀ is any particular solution to Ax = b
                      - v is the general solution to Ax = 0 (the null space of A)
                    """)
            
            with col2:
                st.markdown("### Row Echelon Forms")
                
                st.markdown("**RREF of Inhomogeneous System (Ax = b):**")
                st.markdown(f"$${self._matrix_to_latex(rref_inhomogeneous)}$$")
                
                st.markdown("**RREF of Homogeneous System (Ax = 0):**")
                st.markdown(f"$${self._matrix_to_latex(rref_homogeneous)}$$")
                
                # Compute solutions if possible
                if inhomo_solution_type != "inconsistent":
                    st.markdown("### Solution Computation")
                    
                    # Find particular solution for inhomogeneous system
                    x_particular = np.zeros((n, 1))
                    pivot_columns = []
                    
                    # Identify pivot columns
                    for i in range(min(m, n)):
                        for j in range(n):
                            if abs(rref_inhomogeneous[i, j]) > 1e-10:
                                pivot_columns.append(j)
                                x_particular[j] = rref_inhomogeneous[i, -1]
                                break
                    
                    # Free columns
                    free_columns = [j for j in range(n) if j not in pivot_columns]
                    
                    # Display the particular solution
                    st.markdown("**Particular Solution to Ax = b:**")
                    st.markdown(f"$$\\mathbf{{x}}_p = {self._vector_to_latex(x_particular)}$$")
                    
                    # Compute null space basis
                    if homo_solution_type == "parametric":
                        import scipy.linalg as la
                        null_space = la.null_space(A)
                        
                        # If the null space has a non-zero dimension
                        if null_space.shape[1] > 0:
                            st.markdown("**Null Space Basis (Solutions to Ax = 0):**")
                            
                            # Display each basis vector
                            for i in range(null_space.shape[1]):
                                basis_vector = null_space[:, i].reshape(-1, 1)
                                st.markdown(f"$$\\mathbf{{v}}_{i+1} = {self._vector_to_latex(basis_vector)}$$")
                            
                            # Display the general solution
                            st.markdown("**General Solution to Ax = b:**")
                            
                            # Create a parametric form for display
                            param_terms = [f"\\mathbf{{x}}_p"]
                            for i in range(null_space.shape[1]):
                                param_terms.append(f"c_{i+1} \\mathbf{{v}}_{i+1}")
                            
                            general_form = " + ".join(param_terms)
                            st.markdown(f"$$\\mathbf{{x}} = {general_form}$$")
                            
                            st.markdown("Where c₁, c₂, ... are arbitrary scalar parameters.")
                
                # Visualize the solution space (for 2D and 3D cases)
                if n <= 3:
                    st.markdown("### Solution Space Visualization")
                    
                    if n == 2:
                        # 2D visualization
                        import plotly.graph_objects as go
                        
                        # Create a figure
                        fig = go.Figure()
                        
                        # Grid range
                        x_range = np.linspace(-5, 5, 100)
                        y_range = np.linspace(-5, 5, 100)
                        
                        # For homogeneous system (Ax = 0)
                        if rank_A < n:
                            # This is a line through the origin in 2D
                            # Get the null space vector
                            null_vector = null_space[:, 0]
                            
                            # Generate points along the line
                            t_vals = np.linspace(-5, 5, 100)
                            x_vals = null_vector[0] * t_vals
                            y_vals = null_vector[1] * t_vals
                            
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=y_vals,
                                mode='lines',
                                name='Homogeneous Solutions (Ax = 0)',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Add the origin point
                            fig.add_trace(go.Scatter(
                                x=[0], y=[0],
                                mode='markers',
                                name='Origin (0,0)',
                                marker=dict(color='black', size=8)
                            ))
                        
                        # For inhomogeneous system (Ax = b)
                        if inhomo_solution_type != "inconsistent":
                            if inhomo_solution_type == "unique":
                                # Single point
                                fig.add_trace(go.Scatter(
                                    x=[x_particular[0, 0]], y=[x_particular[1, 0]],
                                    mode='markers',
                                    name='Unique Solution to Ax = b',
                                    marker=dict(color='red', size=8)
                                ))
                            else:
                                # Parallel line to homogeneous solution
                                # Generate points along the line
                                t_vals = np.linspace(-5, 5, 100)
                                x_vals = x_particular[0, 0] + null_vector[0] * t_vals
                                y_vals = x_particular[1, 0] + null_vector[1] * t_vals
                                
                                fig.add_trace(go.Scatter(
                                    x=x_vals, y=y_vals,
                                    mode='lines',
                                    name='Inhomogeneous Solutions (Ax = b)',
                                    line=dict(color='red', width=2)
                                ))
                                
                                # Add the particular solution point
                                fig.add_trace(go.Scatter(
                                    x=[x_particular[0, 0]], y=[x_particular[1, 0]],
                                    mode='markers',
                                    name='Particular Solution',
                                    marker=dict(color='red', size=8)
                                ))
                        
                        # Add the equation lines
                        for i in range(m):
                            if abs(A[i, 1]) > 1e-10:  # Non-vertical line
                                # Line equation: a*x + b*y = c
                                # y = (c - a*x) / b
                                line_x = x_range
                                line_y = [(b[i, 0] - A[i, 0] * x) / A[i, 1] for x in line_x]
                                
                                fig.add_trace(go.Scatter(
                                    x=line_x, y=line_y,
                                    mode='lines',
                                    name=f'Equation {i+1}',
                                    line=dict(color='green', width=1, dash='dash')
                                ))
                            else:  # Vertical line
                                if abs(A[i, 0]) > 1e-10:
                                    # x = c/a
                                    x_val = b[i, 0] / A[i, 0]
                                    fig.add_trace(go.Scatter(
                                        x=[x_val, x_val], y=[-5, 5],
                                        mode='lines',
                                        name=f'Equation {i+1}',
                                        line=dict(color='green', width=1, dash='dash')
                                    ))
                        
                        # Set layout
                        fig.update_layout(
                            title="2D Visualization of Solution Spaces",
                            xaxis=dict(title="x₁", range=[-5, 5]),
                            yaxis=dict(title="x₂", range=[-5, 5]),
                            legend=dict(x=0, y=1),
                            width=600, height=500
                        )
                        
                        # Add grid and center
                        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='gray')
                        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='gray')
                        
                        # Display the plot
                        st.plotly_chart(fig)
                        
                    elif n == 3:
                        # 3D visualization (simplified for complexity)
                        import plotly.graph_objects as go
                        
                        # Create a figure
                        fig = go.Figure()
                        
                        # For homogeneous system (Ax = 0)
                        if homo_solution_type == "parametric":
                            # Get the null space vectors
                            null_dim = null_space.shape[1]
                            
                            if null_dim == 1:
                                # Line through origin
                                t_vals = np.linspace(-3, 3, 50)
                                null_vec = null_space[:, 0]
                                
                                fig.add_trace(go.Scatter3d(
                                    x=null_vec[0] * t_vals,
                                    y=null_vec[1] * t_vals,
                                    z=null_vec[2] * t_vals,
                                    mode='lines',
                                    name='Homogeneous Solutions (Ax = 0)',
                                    line=dict(color='blue', width=4)
                                ))
                            
                            elif null_dim == 2:
                                # Plane through origin
                                # Create a grid for parameters
                                u = np.linspace(-2, 2, 10)
                                v = np.linspace(-2, 2, 10)
                                U, V = np.meshgrid(u, v)
                                
                                # Compute points
                                X = null_space[0, 0] * U + null_space[0, 1] * V
                                Y = null_space[1, 0] * U + null_space[1, 1] * V
                                Z = null_space[2, 0] * U + null_space[2, 1] * V
                                
                                fig.add_trace(go.Surface(
                                    x=X, y=Y, z=Z,
                                    colorscale='Blues',
                                    opacity=0.8,
                                    name='Homogeneous Solutions (Ax = 0)'
                                ))
                        
                        # Add the origin point
                        fig.add_trace(go.Scatter3d(
                            x=[0], y=[0], z=[0],
                            mode='markers',
                            name='Origin',
                            marker=dict(color='black', size=6)
                        ))
                        
                        # For inhomogeneous system (Ax = b)
                        if inhomo_solution_type != "inconsistent":
                            # Add the particular solution
                            fig.add_trace(go.Scatter3d(
                                x=[x_particular[0, 0]],
                                y=[x_particular[1, 0]],
                                z=[x_particular[2, 0]],
                                mode='markers',
                                name='Particular Solution',
                                marker=dict(color='red', size=6)
                            ))
                            
                            if inhomo_solution_type == "parametric" and null_dim == 1:
                                # Line parallel to homogeneous solution
                                t_vals = np.linspace(-3, 3, 50)
                                null_vec = null_space[:, 0]
                                
                                fig.add_trace(go.Scatter3d(
                                    x=x_particular[0, 0] + null_vec[0] * t_vals,
                                    y=x_particular[1, 0] + null_vec[1] * t_vals,
                                    z=x_particular[2, 0] + null_vec[2] * t_vals,
                                    mode='lines',
                                    name='Inhomogeneous Solutions (Ax = b)',
                                    line=dict(color='red', width=4)
                                ))
                            
                            elif inhomo_solution_type == "parametric" and null_dim == 2:
                                # Plane parallel to homogeneous solution
                                u = np.linspace(-2, 2, 10)
                                v = np.linspace(-2, 2, 10)
                                U, V = np.meshgrid(u, v)
                                
                                # Compute points
                                X = x_particular[0, 0] + null_space[0, 0] * U + null_space[0, 1] * V
                                Y = x_particular[1, 0] + null_space[1, 0] * U + null_space[1, 1] * V
                                Z = x_particular[2, 0] + null_space[2, 0] * U + null_space[2, 1] * V
                                
                                fig.add_trace(go.Surface(
                                    x=X, y=Y, z=Z,
                                    colorscale='Reds',
                                    opacity=0.8,
                                    name='Inhomogeneous Solutions (Ax = b)'
                                ))
                        
                        # Set layout
                        fig.update_layout(
                            title="3D Visualization of Solution Spaces",
                            scene=dict(
                                xaxis_title="x₁",
                                yaxis_title="x₂",
                                zaxis_title="x₃",
                                aspectmode='cube'
                            ),
                            width=600, height=600
                        )
                        
                        # Display the plot
                        st.plotly_chart(fig)
                
                # Educational explanation
                st.markdown("### Key Concepts")
                st.markdown("""
                **Null Space (Ker A):**
                - The set of all solutions to Ax = 0
                - Also called the kernel of A
                - Always contains the zero vector
                - Forms a vector subspace of ℝⁿ
                
                **Solution Set for Ax = b:**
                - If Ax = b has a solution, its solution set is an affine space
                - It's a translation of the null space by any particular solution
                
                **Fundamental Theorem:**
                - If Ax = b has a solution x₀, then all solutions are of the form x = x₀ + v, where v is in Null(A)
                - The solution space has dimension n - rank(A)
                """)
            
            return {
                "A": A, 
                "b": b,
                "rank_A": rank_A,
                "rank_augmented": rank_augmented,
                "homogeneous_rref": rref_homogeneous,
                "inhomogeneous_rref": rref_inhomogeneous,
                "homogeneous_solution_type": homo_solution_type,
                "inhomogeneous_solution_type": inhomo_solution_type
            }
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
        return None
    
    def geometric_interpretation(self, matrix_input):
        """
        Provide geometric interpretation of a linear system, visualizing the solution
        as the intersection of lines, planes, or higher-dimensional objects.
        """
        st.subheader("Geometric Interpretation of Linear Systems")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the augmented matrix
            augmented_matrix = self.framework.parse_matrix(matrix_input)
            
            # Split into coefficient matrix A and constant vector b
            A = augmented_matrix[:, :-1]
            b = augmented_matrix[:, -1:]
            
            # Dimensions
            m, n = A.shape
            
            # Display the linear system
            st.markdown("### Original Linear System")
            st.markdown(f"**Augmented Matrix [A|b]:**")
            st.markdown(f"$${self._matrix_to_latex(augmented_matrix)}$$")
            
            # Display in equation form
            st.markdown("**Equation Form:**")
            equations = []
            for i in range(m):
                eq_terms = []
                for j in range(n):
                    coef = A[i, j]
                    if np.isclose(coef, 0, atol=1e-10):
                        continue
                        
                    if j > 0 and coef > 0:
                        eq_terms.append("+")
                        
                    if np.isclose(abs(coef), 1, atol=1e-10):
                        if coef < 0:
                            eq_terms.append("-")
                    else:
                        if coef < 0:
                            eq_terms.append(f"{coef:.2f}")
                        else:
                            eq_terms.append(f"{coef:.2f}")
                            
                    eq_terms.append(f"x_{j+1}")
                
                if not eq_terms:
                    eq_terms.append("0")
                    
                # Add the equals sign and right-hand side
                eq_terms.append("=")
                eq_terms.append(f"{b[i, 0]:.2f}")
                
                equations.append("$" + " ".join(eq_terms) + "$")
            
            for eq in equations:
                st.markdown(eq)
            
            # Analyze the system
            rank_A = np.linalg.matrix_rank(A)
            rank_augmented = np.linalg.matrix_rank(augmented_matrix)
            
            # Check dimensions for visualization support
            if n <= 3:
                # Determine the solution type
                if rank_A != rank_augmented:
                    solution_type = "inconsistent"
                    solution_message = "The system is inconsistent (no solution)"
                elif rank_A == n:
                    solution_type = "unique"
                    solution_message = "The system has a unique solution"
                else:
                    solution_type = "infinite"
                    nullity = n - rank_A
                    solution_message = f"The system has infinitely many solutions ({nullity} parameter(s))"
                
                # Compute the solution if possible
                solution = None
                if solution_type == "unique":
                    # Use the framework to solve the system
                    # Create args for solve_gauss
                    class Args:
                        def __init__(self, matrix):
                            self.matrix = matrix
                            
                    args = Args(matrix_input)
                    solution = self.framework.solve_gauss(args)
                    
                    if isinstance(solution, np.ndarray):
                        solution_vector = solution.reshape(-1, 1)
                    else:
                        solution_vector = None
                
                # Create visualizations based on dimensions
                if n == 2:
                    st.markdown("### 2D Geometric Interpretation")
                    st.markdown("Each equation represents a line in the x₁-x₂ plane:")
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    # Create a grid of points
                    x_range = np.linspace(-10, 10, 100)
                    y_range = np.linspace(-10, 10, 100)
                    
                    # Plot each line (equation)
                    for i in range(m):
                        if abs(A[i, 1]) > 1e-10:
                            # Line: a*x + b*y = c
                            # y = (c - a*x) / b
                            line_x = x_range
                            line_y = [(b[i, 0] - A[i, 0] * x) / A[i, 1] for x in line_x]
                            
                            fig.add_trace(go.Scatter(
                                x=line_x, y=line_y,
                                mode='lines',
                                name=f'Equation {i+1}',
                                line=dict(width=2)
                            ))
                        elif abs(A[i, 0]) > 1e-10:
                            # Vertical line: x = c/a
                            x_val = b[i, 0] / A[i, 0]
                            fig.add_trace(go.Scatter(
                                x=[x_val, x_val], y=[-10, 10],
                                mode='lines',
                                name=f'Equation {i+1}',
                                line=dict(width=2)
                            ))
                    
                    # Add the solution point if it exists
                    if solution_type == "unique" and solution_vector is not None:
                        fig.add_trace(go.Scatter(
                            x=[solution_vector[0, 0]],
                            y=[solution_vector[1, 0]],
                            mode='markers',
                            name='Solution',
                            marker=dict(color='red', size=10)
                        ))
                    
                    # Set layout
                    fig.update_layout(
                        title="Geometric Interpretation: Lines in 2D Plane",
                        xaxis=dict(title="x₁", range=[-10, 10], zeroline=True),
                        yaxis=dict(title="x₂", range=[-10, 10], zeroline=True, scaleanchor="x", scaleratio=1),
                        legend=dict(x=0, y=1),
                        width=700, height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    st.markdown(f"**Solution Status:** {solution_message}")
                    
                    if solution_type == "unique":
                        st.success("The unique solution is the point where all lines intersect.")
                        if solution_vector is not None:
                            st.markdown(f"**Solution Coordinates:** ({solution_vector[0, 0]:.4f}, {solution_vector[1, 0]:.4f})")
                    elif solution_type == "infinite":
                        st.info("The system has infinitely many solutions, represented by a line (or multiple overlapping lines).")
                        st.markdown("This happens when two or more lines coincide, or when there are fewer independent equations than variables.")
                    else:  # inconsistent
                        st.error("The system is inconsistent - the lines do not intersect at a common point.")
                        st.markdown("This means there is no point that satisfies all equations simultaneously.")
                
                elif n == 3:
                    st.markdown("### 3D Geometric Interpretation")
                    st.markdown("Each equation represents a plane in 3D space:")
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    # Define a grid for plotting planes
                    grid_size = 10
                    xx, yy = np.meshgrid(np.linspace(-5, 5, grid_size), np.linspace(-5, 5, grid_size))
                    
                    # Plot each plane
                    for i in range(m):
                        if abs(A[i, 2]) > 1e-10:
                            # Plane: a*x + b*y + c*z = d
                            # z = (d - a*x - b*y) / c
                            zz = (b[i, 0] - A[i, 0] * xx - A[i, 1] * yy) / A[i, 2]
                            
                            # Add a surface for this plane
                            fig.add_trace(go.Surface(
                                x=xx, y=yy, z=zz,
                                opacity=0.7,
                                colorscale=[[0, f'rgba({30+40*i}, {100+30*i}, {200-20*i}, 0.7)'], 
                                           [1, f'rgba({60+40*i}, {120+30*i}, {220-20*i}, 0.7)']],
                                showscale=False,
                                name=f'Plane {i+1}'
                            ))
                        elif abs(A[i, 1]) > 1e-10:
                            # y = (d - a*x - c*z) / b
                            zz, xx = np.meshgrid(np.linspace(-5, 5, grid_size), np.linspace(-5, 5, grid_size))
                            yy = (b[i, 0] - A[i, 0] * xx - A[i, 2] * zz) / A[i, 1]
                            
                            fig.add_trace(go.Surface(
                                x=xx, y=yy, z=zz,
                                opacity=0.7,
                                colorscale=[[0, f'rgba({30+40*i}, {100+30*i}, {200-20*i}, 0.7)'], 
                                           [1, f'rgba({60+40*i}, {120+30*i}, {220-20*i}, 0.7)']],
                                showscale=False,
                                name=f'Plane {i+1}'
                            ))
                        elif abs(A[i, 0]) > 1e-10:
                            # x = (d - b*y - c*z) / a
                            zz, yy = np.meshgrid(np.linspace(-5, 5, grid_size), np.linspace(-5, 5, grid_size))
                            xx = (b[i, 0] - A[i, 1] * yy - A[i, 2] * zz) / A[i, 0]
                            
                            fig.add_trace(go.Surface(
                                x=xx, y=yy, z=zz,
                                opacity=0.7,
                                colorscale=[[0, f'rgba({30+40*i}, {100+30*i}, {200-20*i}, 0.7)'], 
                                           [1, f'rgba({60+40*i}, {120+30*i}, {220-20*i}, 0.7)']],
                                showscale=False,
                                name=f'Plane {i+1}'
                            ))
                    
                    # Add the solution point if unique
                    if solution_type == "unique" and solution_vector is not None:
                        fig.add_trace(go.Scatter3d(
                            x=[solution_vector[0, 0]],
                            y=[solution_vector[1, 0]],
                            z=[solution_vector[2, 0]],
                            mode='markers',
                            name='Solution',
                            marker=dict(color='red', size=8)
                        ))
                    
                    # Set layout
                    fig.update_layout(
                        title=f"Geometric Interpretation: Planes in 3D Space ({solution_message})",
                        scene=dict(
                            xaxis_title="x₁",
                            yaxis_title="x₂",
                            zaxis_title="x₃",
                            aspectmode='cube',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                        ),
                        width=700, height=600
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    st.markdown(f"**Solution Status:** {solution_message}")
                    
                    if solution_type == "unique":
                        st.success("The unique solution is the point where all planes intersect.")
                        if solution_vector is not None:
                            st.markdown(f"**Solution Coordinates:** ({solution_vector[0, 0]:.4f}, {solution_vector[1, 0]:.4f}, {solution_vector[2, 0]:.4f})")
                    elif solution_type == "infinite":
                        if n - rank_A == 1:
                            st.info("The system has infinitely many solutions, represented by a line in 3D space.")
                            st.markdown("This happens when the planes intersect along a common line.")
                        else:  # n - rank_A == 2
                            st.info("The system has infinitely many solutions, represented by a plane in 3D space.")
                            st.markdown("This happens when multiple planes coincide.")
                    else:  # inconsistent
                        st.error("The system is inconsistent - the planes do not have a common point of intersection.")
                        st.markdown("This can happen when planes are parallel or when their lines of intersection don't meet.")
            
            else:
                st.warning(f"Geometric visualization is only available for systems with 2 or 3 variables. Your system has {n} variables.")
                
                # Provide general information for higher dimensions
                st.markdown("### Higher-Dimensional Interpretation")
                st.markdown(f"**Number of Equations:** {m}")
                st.markdown(f"**Number of Variables:** {n}")
                st.markdown(f"**Rank of Coefficient Matrix:** {rank_A}")
                
                if rank_A != rank_augmented:
                    st.error("The system is inconsistent (no solution exists).")
                    st.markdown("Each equation represents a hyperplane in n-dimensional space. These hyperplanes don't have a common intersection point.")
                elif rank_A == n:
                    st.success("The system has a unique solution.")
                    st.markdown("The unique solution is the single point where all hyperplanes intersect in n-dimensional space.")
                else:
                    st.info(f"The system has infinitely many solutions with {n - rank_A} free parameters.")
                    
                    if n - rank_A == 1:
                        st.markdown("The solution forms a line in n-dimensional space.")
                    elif n - rank_A == 2:
                        st.markdown("The solution forms a plane in n-dimensional space.")
                    else:
                        st.markdown(f"The solution forms a {n - rank_A}-dimensional affine subspace in {n}-dimensional space.")
            
            # Add educational content about geometric interpretation
            with st.expander("Learn More About Geometric Interpretation"):
                st.markdown("""
                ### Geometric Intuition in Different Dimensions
                
                **In 2D (n=2):**
                - Each equation represents a line in the plane
                - Two lines can:
                  - Intersect at a single point (unique solution)
                  - Be parallel (no solution)
                  - Coincide (infinitely many solutions)
                
                **In 3D (n=3):**
                - Each equation represents a plane in 3D space
                - The intersection of planes can be:
                  - A single point (unique solution)
                  - A line (infinitely many solutions with 1 parameter)
                  - A plane (infinitely many solutions with 2 parameters)
                  - Empty (no solution)
                
                **In Higher Dimensions:**
                - Each equation represents a hyperplane
                - Intersections become more complex, but the same principles apply
                - A system with n variables can have a solution space of dimension n - rank(A)
                
                **Relationship to Linear Independence:**
                - If the rows of the coefficient matrix A are linearly independent, then the corresponding hyperplanes are in "general position"
                - Linear dependence among equations means the corresponding hyperplanes have a special geometric relationship (like being parallel or coincident)
                """)
            
            return {
                "A": A,
                "b": b,
                "rank_A": rank_A,
                "rank_augmented": rank_augmented,
                "solution_type": solution_type
            }
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
        return None
        
    def matrix_multiply(self, matrix_a_input, matrix_b_input):
        """
        Multiply two matrices and display the result with explanation.
        """
        st.subheader("Matrix Multiplication")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the matrices
            matrix_a = self.framework.parse_matrix(matrix_a_input)
            matrix_b = self.framework.parse_matrix(matrix_b_input)
            
            # Check compatibility for multiplication
            if matrix_a.shape[1] != matrix_b.shape[0]:
                st.error(f"Matrices are not compatible for multiplication. Matrix A has {matrix_a.shape[1]} columns but Matrix B has {matrix_b.shape[0]} rows.")
                return None
                
            # Perform the multiplication
            product = matrix_a @ matrix_b
            
            # Display the result
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Input Matrices")
                st.markdown("**Matrix A:**")
                st.markdown(f"$${self._matrix_to_latex(matrix_a)}$$")
                st.markdown(f"Shape: {matrix_a.shape[0]}×{matrix_a.shape[1]}")
                
                st.markdown("**Matrix B:**")
                st.markdown(f"$${self._matrix_to_latex(matrix_b)}$$")
                st.markdown(f"Shape: {matrix_b.shape[0]}×{matrix_b.shape[1]}")
                
                st.markdown("### Result")
                st.markdown("**Product A×B:**")
                st.markdown(f"$${self._matrix_to_latex(product)}$$")
                st.markdown(f"Shape: {product.shape[0]}×{product.shape[1]}")
                
                # Explanation of the calculation for a small example
                if product.size <= 9:  # Only show detailed calculation for small matrices
                    st.markdown("### Calculation Details")
                    for i in range(product.shape[0]):
                        for j in range(product.shape[1]):
                            # Calculate the dot product for element (i,j)
                            dot_product_parts = []
                            for k in range(matrix_a.shape[1]):
                                a_ik = matrix_a[i, k]
                                b_kj = matrix_b[k, j]
                                dot_product_parts.append(f"{a_ik:.2f} \\cdot {b_kj:.2f}")
                            
                            dot_product_str = " + ".join(dot_product_parts)
                            st.markdown(f"$C_{{{i+1}{j+1}}} = {dot_product_str} = {product[i, j]:.2f}$")
            
            with col2:
                st.markdown("### Visualizations")
                
                # Heatmap visualization of the matrices
                tab1, tab2, tab3 = st.tabs(["Matrix A", "Matrix B", "Product A×B"])
                
                with tab1:
                    display_matrix_heatmap(matrix_a, title="Matrix A")
                
                with tab2:
                    display_matrix_heatmap(matrix_b, title="Matrix B")
                
                with tab3:
                    display_matrix_heatmap(product, title="Product A×B")
                
                # Educational content about matrix multiplication
                st.markdown("### Matrix Multiplication Properties")
                
                # Check if multiplication is commutative for this example
                if matrix_b.shape[1] == matrix_a.shape[0]:
                    try:
                        product_ba = matrix_b @ matrix_a
                        if np.allclose(product, product_ba):
                            st.success("For this specific example, A×B = B×A (commutative)")
                        else:
                            st.warning("For this example, A×B ≠ B×A (not commutative)")
                    except:
                        st.warning("Matrix multiplication is generally not commutative: A×B ≠ B×A")
                else:
                    st.warning("Matrix multiplication is generally not commutative: A×B ≠ B×A")
                
                st.markdown("""
                **Key Properties:**
                - (AB)C = A(BC) (associative)
                - A(B+C) = AB+AC (distributive)
                - In general, AB ≠ BA (not commutative)
                - If AB = 0, it doesn't mean A = 0 or B = 0
                """)
                
            return product
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
        return None
        
    def matrix_inverse(self, matrix_input):
        """
        Calculate the inverse of a square matrix and visualize the result.
        """
        st.subheader("Matrix Inverse")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the matrix
            matrix = self.framework.parse_matrix(matrix_input)
            
            # Check if the matrix is square
            if matrix.shape[0] != matrix.shape[1]:
                st.error(f"Cannot compute inverse: matrix must be square. Got shape {matrix.shape}")
                return None
            
            # Calculate determinant to check invertibility
            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:
                st.error(f"Matrix is not invertible (determinant ≈ 0: {det:.10f})")
                return None
            
            # Calculate the inverse matrix
            inverse = np.linalg.inv(matrix)
            
            # Calculate the verification product (should be identity)
            verification = matrix @ inverse
            
            # Display the results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Input Matrix A:")
                st.markdown(f"$${self._matrix_to_latex(matrix)}$$")
                
                st.markdown("### Inverse Matrix A⁻¹:")
                st.markdown(f"$${self._matrix_to_latex(inverse)}$$")
                
                st.markdown("### Verification")
                st.markdown("Checking that A × A⁻¹ = I (identity matrix):")
                st.markdown(f"$${self._matrix_to_latex(verification)}$$")
                
                # For 2x2 matrices, show the formula explicitly
                if matrix.shape == (2, 2):
                    st.markdown("### Matrix Inverse Formula")
                    st.markdown(r"""For a 2×2 matrix $A=\begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$A^{-1}=\frac{1}{\det(A)}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}=\frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$

Where:
- $\det(A) = ad - bc$
""")
                    # Calculate the actual values
                    a, b = matrix[0, 0], matrix[0, 1]
                    c, d = matrix[1, 0], matrix[1, 1]
                    determinant = a*d - b*c
                    
                    # Format the calculation
                    st.markdown("**Calculation:**")
                    st.markdown(f"$\\det(A) = ({a})({d}) - ({b})({c}) = {determinant}$")
                    st.markdown(f"$A^{{-1}} = \\frac{{1}}{{{determinant}}}\\begin{{pmatrix}} {d} & {-b} \\\\ {-c} & {a} \\end{{pmatrix}}$")
            
            with col2:
                st.markdown("### Visualization")
                
                # Visualize the matrices
                st.markdown("**Original Matrix A**")
                display_matrix_heatmap(matrix, title="Matrix A")
                
                st.markdown("**Inverse Matrix A⁻¹**")
                display_matrix_heatmap(inverse, title="Inverse Matrix", center_scale=True)
                
                st.markdown("**Verification: A·A⁻¹**")
                display_matrix_heatmap(verification, title="A·A⁻¹ (should be identity)", center_scale=True)
                
                # Additional information about matrix inverses
                st.markdown("### Properties of Matrix Inverses")
                st.markdown("""
                **Key Properties:**
                - Only square matrices can have inverses
                - A matrix has an inverse if and only if its determinant is non-zero
                - If A is invertible, then (A⁻¹)⁻¹ = A
                - If A and B are invertible, then (AB)⁻¹ = B⁻¹A⁻¹
                - The inverse of a product reverses the order
                """)
                
                # For larger matrices, mention numerical methods
                if matrix.shape[0] > 3:
                    st.info("""
                    For matrices larger than 3×3, inverses are typically computed using numerical methods:
                    - Gaussian elimination (with pivoting)
                    - LU decomposition
                    - QR decomposition
                    - Singular Value Decomposition (SVD)
                    
                    These methods are more stable and efficient than the explicit formula approach.
                    """)
            
            return inverse
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except np.linalg.LinAlgError as e:
            st.error(f"Linear algebra error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
        return None
    
    def matrix_determinant(self, matrix_input):
        """Calculate the determinant of a matrix."""
        
        st.subheader("Matrix Determinant")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the matrix
            matrix = self.framework.parse_matrix(matrix_input)
            
            # Check if the matrix is square
            rows, cols = matrix.shape
            if rows != cols:
                st.error(f"Cannot calculate determinant of a non-square matrix. Matrix shape: {matrix.shape}")
                return None
            
            # Calculate the determinant
            determinant = np.linalg.det(matrix)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Matrix and Result")
                st.markdown("**Input Matrix:**")
                st.text(str(matrix))
                
                st.markdown("**Determinant Value:**")
                st.markdown(f"det(A) = {determinant:.6f}")
                
                # If close to an integer, show that as well
                if np.isclose(determinant, round(determinant), atol=1e-10):
                    st.markdown(f"det(A) ≈ {int(round(determinant))}")
                
                # LaTeX version of the calculation
                matrix_latex = self._matrix_to_latex(matrix)
                # Properly format the matrix for the determinant notation
                # Extract the matrix content without slicing by position, which could be unreliable
                if "\\begin{pmatrix}" in matrix_latex and "\\end{pmatrix}" in matrix_latex:
                    # Just use the entire matrix_latex directly
                    st.markdown(f"$$\\det{matrix_latex} = {determinant:.6f}$$")
                else:
                    # Fallback in case matrix_latex doesn't have the expected format
                    st.markdown(f"$$\\det(A) = {determinant:.6f}$$")
                
                # Properties of the determinant
                st.markdown("### Properties")
                if np.isclose(determinant, 0, atol=1e-10):
                    st.markdown("- Matrix is **singular** (not invertible)")
                    st.markdown("- The system of linear equations represented by this matrix has either no solution or infinitely many solutions")
                    st.markdown("- Zero is an eigenvalue of this matrix")
                else:
                    st.markdown("- Matrix is **non-singular** (invertible)")
                    st.markdown("- The system of linear equations represented by this matrix has a unique solution")
                    st.markdown(f"- The volume scaling factor of linear transformations represented by this matrix is {abs(determinant):.6f}")
                    if determinant < 0:
                        st.markdown("- The transformation changes the orientation of space (reflection)")
            
            with col2:
                st.markdown("### Visualization")
                display_matrix_heatmap(matrix, title="Matrix A")
                
                # Add a large display of the determinant value
                det_box = f"""
                <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center; margin-top: 20px;">
                    <h2 style="margin-bottom: 10px;">Determinant</h2>
                    <h1 style="font-size: 2.5em; color: {('#e74c3c' if determinant < 0 else '#2ecc71' if determinant > 0 else '#7f8c8d')};">
                        {determinant:.6f}
                    </h1>
                    <p style="font-size: 1.2em; margin-top: 10px;">
                        {('Negative (Orientation-Reversing)' if determinant < 0 else 'Positive (Orientation-Preserving)' if determinant > 0 else 'Zero (Singular Matrix)')}
                    </p>
                </div>
                """
                st.markdown(det_box, unsafe_allow_html=True)
                
                # For 2×2 matrices, show the formula
                if rows == 2:
                    st.markdown("### 2×2 Matrix Determinant Formula")
                    st.markdown(r"""
                    For a 2×2 matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$:
                    
                    $$\det(A) = ad - bc$$
                    
                    Calculation:
                    """)
                    a, b = matrix[0, 0], matrix[0, 1]
                    c, d = matrix[1, 0], matrix[1, 1]
                    st.markdown(f"$$\\det(A) = ({a})({d}) - ({b})({c}) = {a*d:.4f} - {b*c:.4f} = {determinant:.4f}$$")
                
                # For 3×3 matrices, show the formula
                elif rows == 3:
                    st.markdown("### 3×3 Matrix Determinant Formula")
                    st.markdown(r"""
                    For a 3×3 matrix, one method is using cofactor expansion along the first row:
                    
                    $$\det(A) = a_{11}M_{11} - a_{12}M_{12} + a_{13}M_{13}$$
                    
                    Where $M_{ij}$ is the minor of element $a_{ij}$.
                    """)
                    
                    # We could add the actual calculation steps here but it's quite involved
                
                elif rows > 3:
                    st.markdown("### Higher Dimension Determinant")
                    st.markdown("""
                    For larger matrices, determinants are calculated using:
                    - LU decomposition
                    - Cofactor expansion
                    - Other numerical methods
                    
                    The complexity grows factorially with dimension, making direct computation impractical for large matrices.
                    """)
            
            return {"determinant": determinant}
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except np.linalg.LinAlgError as e:
            st.error(f"Linear algebra error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        return None
        
    def eigenvalues_eigenvectors(self, matrix_input):
        """Calculate the eigenvalues and eigenvectors of a matrix."""
        
        st.subheader("Eigenvalues and Eigenvectors")
        
        try:
            if not hasattr(self, 'framework'):
                st.error("Framework not initialized in the calculator.")
                return None
            
            # Parse the matrix
            matrix = self.framework.parse_matrix(matrix_input)
            
            # Check if the matrix is square
            rows, cols = matrix.shape
            if rows != cols:
                st.error(f"Cannot calculate eigenvalues of a non-square matrix. Matrix shape: {matrix.shape}")
                return None
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            
            # Sort eigenvalues and eigenvectors by absolute value of eigenvalues (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:,idx]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Matrix and Results")
                st.markdown("**Input Matrix:**")
                st.text(str(matrix))
                
                st.markdown("### Eigenvalues")
                for i, eigenvalue in enumerate(eigenvalues):
                    # Format eigenvalues nicely
                    if eigenvalue.imag == 0:
                        eigenvalue_str = f"{eigenvalue.real:.6f}"
                    else:
                        # Complex eigenvalue
                        real_part = f"{eigenvalue.real:.6f}"
                        imag_part = f"{abs(eigenvalue.imag):.6f}"
                        sign = "+" if eigenvalue.imag > 0 else "-"
                        eigenvalue_str = f"{real_part} {sign} {imag_part}i"
                    
                    st.markdown(f"λ{i+1} = {eigenvalue_str}")
                
                # Display eigenvectors
                st.markdown("### Eigenvectors")
                for i, eigenvector in enumerate(eigenvectors.T):  # Transpose to match eigenvalues
                    # Format eigenvector nicely
                    if np.any(np.iscomplex(eigenvector)):
                        # Handle complex eigenvectors
                        eigenvector_str = ", ".join([f"{v.real:.4f} {'+' if v.imag >= 0 else '-'} {abs(v.imag):.4f}i" for v in eigenvector])
                    else:
                        # Real eigenvectors
                        eigenvector_str = ", ".join([f"{v.real:.6f}" for v in eigenvector])
                    
                    st.markdown(f"v{i+1} = [{eigenvector_str}]")
                
                # Verify Av = λv for each eigenpair (for real eigenvalues only)
                if not np.any(np.iscomplex(eigenvalues)):
                    st.markdown("### Verification")
                    st.markdown("Checking that **A·v = λ·v** for each eigenpair:")
                    
                    for i in range(len(eigenvalues)):
                        eigenvalue = eigenvalues[i]
                        eigenvector = eigenvectors[:, i]
                        
                        # Calculate A·v
                        av = matrix @ eigenvector
                        # Calculate λ·v
                        lv = eigenvalue * eigenvector
                        
                        st.markdown(f"**For eigenpair {i+1}:**")
                        st.markdown(f"A·v{i+1} = {av}")
                        st.markdown(f"λ{i+1}·v{i+1} = {lv}")
                        # Check if they're approximately equal
                        if np.allclose(av, lv, rtol=1e-5, atol=1e-5):
                            st.markdown("✓ Verification passed (A·v ≈ λ·v)")
                        else:
                            st.markdown("✗ Verification failed (numerical precision issues may be present)")
                        st.markdown("---")
                
                # Characteristic polynomial
                if rows <= 3:
                    st.markdown("### Characteristic Polynomial")
                    st.markdown(r"""
                    The eigenvalues are the roots of the characteristic polynomial:
                    
                    $$\det(A - \lambda I) = 0$$
                    """)
                    
                    if rows == 2:
                        a, b = matrix[0, 0], matrix[0, 1]
                        c, d = matrix[1, 0], matrix[1, 1]
                        # 2x2 characteristic polynomial: λ² - (a+d)λ + (ad-bc)
                        trace = a + d
                        determinant = a*d - b*c
                        
                        trace_term = f"- {trace:.4f}λ" if trace != 0 else ""
                        det_term = f"+ {determinant:.4f}" if determinant >= 0 else f"- {abs(determinant):.4f}"
                        
                        st.markdown(f"$$\\det(A - \\lambda I) = \\lambda^2 {trace_term} {det_term} = 0$$")
                    
                    elif rows == 3:
                        # Just show the general form for 3x3
                        st.markdown(r"""
                        For a 3×3 matrix, the characteristic polynomial is:
                        
                        $$\lambda^3 - \text{tr}(A)\lambda^2 + \text{(sum of principal minors)}\lambda - \det(A) = 0$$
                        """)
            
            with col2:
                st.markdown("### Visualization")
                
                # Visualize the matrix
                display_matrix_heatmap(matrix, title="Matrix A")
                
                # Create a visual representation of eigenvalues
                real_parts = eigenvalues.real
                imag_parts = eigenvalues.imag
                
                # Only create plot if we have eigenvalues
                if len(eigenvalues) > 0:
                    # Plot eigenvalues in the complex plane
                    fig = go.Figure()
                    
                    # Add eigenvalues as points
                    fig.add_trace(go.Scatter(
                        x=real_parts,
                        y=imag_parts,
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color='blue',
                            line=dict(width=2, color='DarkSlateGrey')
                        ),
                        text=[f'λ{i+1}' for i in range(len(eigenvalues))],
                        textposition="top center",
                        name='Eigenvalues'
                    ))
                    
                    # Add a line through the origin
                    fig.add_trace(go.Scatter(
                        x=[-max(1.5, abs(real_parts.max()), abs(real_parts.min())),
                           max(1.5, abs(real_parts.max()), abs(real_parts.min()))],
                        y=[0, 0],
                        mode='lines',
                        line=dict(color='black', width=1, dash='dash'),
                        name='Real axis'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[0, 0],
                        y=[-max(1.5, abs(imag_parts.max()), abs(imag_parts.min())),
                           max(1.5, abs(imag_parts.max()), abs(imag_parts.min()))],
                        mode='lines',
                        line=dict(color='black', width=1, dash='dash'),
                        name='Imaginary axis'
                    ))
                    
                    # Set the plot title and labels
                    fig.update_layout(
                        title='Eigenvalues in the Complex Plane',
                        xaxis_title='Real Part',
                        yaxis_title='Imaginary Part',
                        showlegend=False,
                        xaxis=dict(
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black',
                        ),
                        yaxis=dict(
                            scaleanchor="x",
                            scaleratio=1,
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black',
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation of eigenvalues and eigenvectors
                st.markdown("### What are Eigenvalues and Eigenvectors?")
                st.markdown("""
                - An **eigenvector** of a matrix A is a non-zero vector v such that when A is multiplied by v, the result is a scalar multiple of v.
                - The **eigenvalue** λ is the scalar that the eigenvector is multiplied by: A·v = λ·v
                - Eigenvalues and eigenvectors have many applications including:
                  - Principal Component Analysis (PCA)
                  - Solving systems of differential equations
                  - Determining the stability of systems
                  - Finding the axes of rotation in 3D transformations
                """)
                
                # Add information about the eigenvalues of this specific matrix
                st.markdown("### Properties of This Matrix")
                
                # Check if all eigenvalues are real
                all_real = np.allclose(eigenvalues.imag, 0)
                if all_real:
                    st.markdown("- All eigenvalues are real numbers")
                else:
                    st.markdown("- Some eigenvalues are complex numbers")
                
                # Check if diagonalizable (simplification: if all eigenvalues are distinct)
                distinct_eigenvalues = len(eigenvalues) == len(np.unique(eigenvalues))
                if distinct_eigenvalues:
                    st.markdown("- Matrix appears to be diagonalizable (all eigenvalues are distinct)")
                else:
                    st.markdown("- Matrix may not be diagonalizable (has repeated eigenvalues)")
                
                # Calculate determinant from eigenvalues
                det_from_eig = np.prod(eigenvalues)
                if np.isclose(det_from_eig.imag, 0, atol=1e-10):
                    det_from_eig = det_from_eig.real
                    st.markdown(f"- Determinant (product of eigenvalues): {det_from_eig:.6f}")
                
                # Calculate trace from eigenvalues
                trace_from_eig = np.sum(eigenvalues)
                if np.isclose(trace_from_eig.imag, 0, atol=1e-10):
                    trace_from_eig = trace_from_eig.real
                    st.markdown(f"- Trace (sum of eigenvalues): {trace_from_eig:.6f}")
            
            return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}
            
        except ValueError as e:
            st.error(f"Input error: {e}")
        except np.linalg.LinAlgError as e:
            st.error(f"Linear algebra error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        return None