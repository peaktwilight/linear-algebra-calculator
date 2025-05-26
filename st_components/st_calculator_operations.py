#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Linear Algebra Calculator Operations
"""

import streamlit as st
import numpy as np
import pandas as pd
from given_reference.core import mrref # Keep for RREF functionality

# Import self-sufficient utilities
from .st_math_utils import MathUtils

# Import Mixins
from .st_vector_operations_mixin import VectorOperationsMixin
from .st_matrix_operations_mixin import MatrixOperationsMixin
from .st_plane_operations_mixin import PlaneOperationsMixin

class LinAlgCalculator(VectorOperationsMixin, MatrixOperationsMixin, PlaneOperationsMixin):
    def __init__(self):
        super().__init__() # Initialize mixins if they have their own __init__ (optional)
        self.math_utils = MathUtils()
    
    def _format_number_latex(self, num):
        """Helper to format numbers for LaTeX - delegate to MathUtils"""
        return MathUtils.format_number_latex(num)
    
    def _format_vector_to_latex_string(self, vector):
        """Formats a NumPy vector into a LaTeX column vector string - delegate to MathUtils"""
        return MathUtils.format_vector_latex(vector)
    
    def _format_matrix_to_latex_string(self, matrix_val):
        """Formats a NumPy matrix into a LaTeX pmatrix string - delegate to MathUtils"""
        return MathUtils.format_matrix_latex(matrix_val)
    
    def calculate_null_space_basis(self, matrix_a_str):
        """
        Calculates and displays a basis for the null space (kernel) of matrix A.
        The null space is the set of all vectors x such that Ax = 0.
        """
        st.write("### Null Space Calculation")
        try:
            matrix_a = MathUtils.parse_matrix(matrix_a_str)
            if matrix_a is None or matrix_a.size == 0:
                st.error("Invalid or empty matrix format for Matrix A. Please check your input.")
                return

            st.write("#### Input Matrix A:")
            st.latex(self._format_matrix_to_latex_string(matrix_a))

            rref_matrix_full = mrref(matrix_a)

            st.write("#### Reduced Row Echelon Form (RREF) of A (full):")
            st.latex(self._format_matrix_to_latex_string(rref_matrix_full))

            if rref_matrix_full.ndim == 2 and rref_matrix_full.shape[0] > 0:
                active_rref = rref_matrix_full[np.any(rref_matrix_full, axis=1)]
            elif rref_matrix_full.ndim == 1 and np.any(rref_matrix_full):
                active_rref = rref_matrix_full.reshape(1, -1)
            else:
                active_rref = np.array([[]]).reshape(0, matrix_a.shape[1] if matrix_a.ndim == 2 else 0)
            
            st.write("#### Active RREF (non-zero rows for pivot identification):")
            if active_rref.size > 0:
                st.latex(self._format_matrix_to_latex_string(active_rref))
            else:
                st.write("(No active non-zero rows or matrix is zero/empty)")

            pivot_cols = []
            if active_rref.ndim == 2 and active_rref.shape[0] > 0:
                active_rows, active_cols_count = active_rref.shape
                for r_idx in range(active_rows):
                    leading_entry_col = -1
                    for c_idx in range(active_cols_count):
                        if not np.isclose(active_rref[r_idx, c_idx], 0):
                            if np.isclose(active_rref[r_idx, c_idx], 1):
                                leading_entry_col = c_idx
                            break
                    if leading_entry_col != -1 and leading_entry_col not in pivot_cols:
                        pivot_cols.append(leading_entry_col)
            pivot_cols.sort()

            st.write(f"Pivot columns identified (0-based): {pivot_cols}")

            num_vars = matrix_a.shape[1]
            rank = len(pivot_cols)
            num_free_vars = num_vars - rank

            st.write(f"Number of variables (columns in A): {num_vars}")
            st.write(f"Rank of A (number of pivot columns): {rank}")
            st.write(f"Dimension of null space (num_free_vars): {num_free_vars}")

            if num_free_vars == 0:
                st.success("The null space contains only the zero vector (trivial solution).")
                st.latex(r"\mathcal{N}(A) = \{\mathbf{0}\}")
                return

            st.write("#### Basis for the Null Space:")
            
            all_cols = list(range(num_vars))
            free_var_cols = [col for col in all_cols if col not in pivot_cols]
            
            basis_vectors = []
            for free_col_idx in free_var_cols:
                b_vector = np.zeros(num_vars)
                b_vector[free_col_idx] = 1
                for i, current_pivot_col in enumerate(pivot_cols):
                    if current_pivot_col < num_vars and i < active_rref.shape[0] and free_col_idx < active_rref.shape[1]:
                        b_vector[current_pivot_col] = -active_rref[i, free_col_idx]
                basis_vectors.append(b_vector)

            if basis_vectors:
                formatted_vectors = [self._format_vector_to_latex_string(vec) for vec in basis_vectors]
                basis_content = ", ".join(formatted_vectors)
                latex_null_space_str = r"\mathcal{N}(A) = \text{span}\left\{ " + basis_content + r" \right\}"
                st.latex(latex_null_space_str)

                st.write("The basis vectors are (shown as columns in a matrix B):")
                basis_matrix = np.array(basis_vectors).T
                if basis_matrix.size > 0:
                    st.latex(rf"B = {self._format_matrix_to_latex_string(basis_matrix)}")
                else:
                    st.write("(No basis vectors to display as a matrix)")

                explanation = (
                    "#### What is the Null Space (and why find its basis)?\n\n"
                    "Imagine a matrix A as a transformation that takes an input vector **x** and produces an output vector A**x**. "
                    "The **null space** of A (written as N(A)) is a special collection of all input vectors **x** that get transformed into the **zero vector** (i.e., A**x** = **0**).\n\n"
                    "Think of it like this: which vectors does matrix A \"squash\" completely down to nothing?\n\n"
                    "This collection of vectors isn't just random; it forms a *vector space*. A **basis** for this null space is the smallest set of fundamental \"direction\" vectors you need to describe every single vector in that entire null space. Any vector **x** that solves A**x** = **0** can be built by stretching, shrinking, and adding these basis vectors together (a linear combination).\n\n"
                    "--- \n"
                    "#### Exam Recipe: Finding the Basis of the Null Space N(A)\n\n"
                    "Here's a step-by-step guide to find the basis vectors for N(A):\n\n"
                    "1.  **Goal:** We want to find all vectors **x** that satisfy the equation A**x** = **0**.\n\n"
                    "2.  **Get the RREF:** Take your matrix A and transform it into its **Reduced Row Echelon Form (RREF)**. This is a simplified version of A where:\n"
                    "    *   Each leading non-zero entry in a row (called a 'pivot') is 1.\n"
                    "    *   Each pivot is the only non-zero entry in its column.\n"
                    "    *   All-zero rows are at the bottom.\n\n"
                    "3.  **Identify Pivot Columns and Free Columns (from RREF):**\n"
                    "    *   **Pivot Columns:** These are the columns in the RREF that contain a leading 1 (a pivot). The variables corresponding to these columns are called **pivot variables**.\n"
                    "    *   **Free Columns:** These are all the other columns in the RREF that *do not* contain a pivot. The variables for these columns are called **free variables**. You can choose their values!\n\n"
                    "4.  **Write Down the System A**x** = 0 using the RREF:** Each row of the RREF gives you an equation. For example, if a row in RREF is `[1 2 0 3 | 0]` (assuming it came from an augmented matrix for Ax=0, or just consider the coefficients for Ax=0), this means `1*x₁ + 2*x₂ + 0*x₃ + 3*x₄ = 0`.\n\n"
                    "5.  **Express Pivot Variables in Terms of Free Variables:** Rearrange the equations from Step 4 so that each pivot variable is on one side, and only free variables are on the other side.\n\n"
                    "6.  **Construct Basis Vectors (one for each free variable):** This is the core part!\n"
                    "    *   Count how many free variables you have. This number is the dimension of the null space, and it's how many basis vectors you'll find.\n"
                    "    *   For **each** free variable:\n"
                    "        a.  Set **that specific** free variable to **1**.\n"
                    "        b.  Set **all other** free variables to **0**.\n"
                    "        c.  Use the expressions from Step 5 to calculate the values of all the **pivot variables** based on these choices for the free variables.\n"
                    "        d.  Assemble all these values (your chosen free variable values and the calculated pivot variable values) into a vector. This vector is one basis vector for the null space.\n\n"
                    "7.  **Collect Your Basis:** The set of all vectors you created in Step 6 is the basis for the null space N(A). Any solution to A**x** = **0** can be written as a weighted sum (linear combination) of these basis vectors."
                )
                st.info(explanation)
            else:
                st.info("Could not determine basis vectors, or the null space is trivial (already handled).")

        except ValueError as e:
            st.error(f"Input Error: {e} Check matrix format: numbers separated by commas, rows by semicolons or newlines.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # import traceback
            # st.error(f"Traceback: {traceback.format_exc()}")

    def check_vector_solutions(self, matrix_input, vectors_input):
        """Check if given vectors are particular solutions, homogeneous solutions, or neither."""
        try:
            # Parse the augmented matrix [A|b]
            augmented = self.framework.parse_matrix(matrix_input)
            
            # Split into coefficient matrix and right-hand side
            A = augmented[:, :-1]
            b = augmented[:, -1]
            
            # Parse the vectors to check
            vector_lines = vectors_input.strip().split('\n')
            vectors = []
            for line in vector_lines:
                if line.strip():
                    vectors.append(self.framework.parse_vector(line.strip()))
            
            if not vectors:
                st.error("No vectors provided to check.")
                return
            
            st.subheader("Vector Solution Analysis")
            
            # Display the system
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Coefficient Matrix A:**")
                st.latex(self._format_matrix_to_latex_string(A))
                
            with col2:
                st.write("**Right-hand side b:**")
                st.latex(self._format_vector_to_latex_string(b))
            
            # Check each vector
            results = []
            for i, v in enumerate(vectors):
                st.write(f"### Vector {i+1}: {v.tolist()}")
                
                # Compute A·v
                Av = A @ v
                
                # Check if A·v = b (particular solution)
                diff_particular = Av - b
                residual_norm_particular = np.linalg.norm(diff_particular)
                is_particular = residual_norm_particular < 1e-10
                
                # Check if A·v = 0 (homogeneous solution)
                residual_norm_homogeneous = np.linalg.norm(Av)
                is_homogeneous = residual_norm_homogeneous < 1e-10
                
                # Display results
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.write("**Computing A·v:**")
                    st.latex(f"A \\vec{{v}} = {self._format_vector_to_latex_string(Av)}")
                
                with col2:
                    st.write("**Particular Solution Check:**")
                    st.write(f"A·v - b = {diff_particular.tolist()}")
                    st.write(f"||A·v - b|| = {residual_norm_particular:.2e}")
                    if is_particular:
                        st.success("✅ **IS** a particular solution (A·v = b)")
                    else:
                        st.error("❌ **NOT** a particular solution")
                
                with col3:
                    st.write("**Homogeneous Solution Check:**")
                    st.write(f"||A·v|| = {residual_norm_homogeneous:.2e}")
                    if is_homogeneous:
                        st.success("✅ **IS** a homogeneous solution (A·v = 0)")
                    else:
                        st.error("❌ **NOT** a homogeneous solution")
                
                # Summary for this vector
                if is_particular and not is_homogeneous:
                    status = "Particular Solution Only"
                    color = "success"
                elif is_homogeneous and not is_particular:
                    status = "Homogeneous Solution Only"
                    color = "info"
                elif is_particular and is_homogeneous:
                    # This would only happen if b = 0
                    status = "Both Particular and Homogeneous (b = 0)"
                    color = "warning"
                else:
                    status = "Not a Solution"
                    color = "error"
                
                if color == "success":
                    st.success(f"**Final Classification: {status}**")
                elif color == "info":
                    st.info(f"**Final Classification: {status}**")
                elif color == "warning":
                    st.warning(f"**Final Classification: {status}**")
                else:
                    st.error(f"**Final Classification: {status}**")
                
                results.append({
                    "vector": v,
                    "is_particular": is_particular,
                    "is_homogeneous": is_homogeneous,
                    "status": status
                })
                
                st.write("---")
            
            # Summary table
            st.subheader("Summary Table")
            summary_data = []
            for i, result in enumerate(results):
                summary_data.append({
                    "Vector": f"v{i+1} = {result['vector'].tolist()}",
                    "Particular Solution": "✅ Yes" if result['is_particular'] else "❌ No",
                    "Homogeneous Solution": "✅ Yes" if result['is_homogeneous'] else "❌ No",
                    "Classification": result['status']
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Educational note
            st.info("""
            **Remember**: 
            - A **particular solution** satisfies Ax = b (the original system)
            - A **homogeneous solution** satisfies Ax = 0 (the associated homogeneous system)
            - If the system is **inconsistent**, no particular solutions exist
            - The **general solution** has the form: x = particular_solution + homogeneous_solutions
            """)
            
        except ValueError as e:
            st.error(f"Input Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # All vector and matrix methods are now inherited from mixins.
    # The main calculator class can be kept lean, or include 
    # additional high-level orchestration logic if needed in the future.