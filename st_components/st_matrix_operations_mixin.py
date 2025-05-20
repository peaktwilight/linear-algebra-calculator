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
            
            cli_steps = captured_output.getvalue()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### System & Steps")
                st.markdown("**Original Augmented Matrix [A|b]:**")
                st.text(str(augmented_matrix))
                
                st.markdown("**Steps from Gaussian Elimination (CLI Output):**")
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
                    st.markdown("**Particular Solution:**")
                    st.text(str(solution["particular"]))
                    st.markdown("**Null Space (for general solution):**")
                    st.text(str(solution["null_space"]))
                elif isinstance(solution, np.ndarray):
                    st.markdown("The system has a unique solution:")
                    st.text(str(solution))
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