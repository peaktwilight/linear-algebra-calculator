#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Linear Mapping Operations
"""

import streamlit as st
import numpy as np
import sympy as sp
from sympy import symbols, sympify
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

class LinearityOperations:
    """
    Component for checking linearity of mappings and generating matrix representations.
    Handles various types of mappings including polynomial, trigonometric, dot product, and quadratic forms.
    """
    
    def __init__(self):
        self.test_vectors = None
        self.mapping_function = None
        self.mapping_type = None
        
    def _format_vector_latex(self, vector):
        """Format a vector for LaTeX display."""
        if isinstance(vector, (list, tuple)):
            vector = np.array(vector)
        if vector.size == 0:
            return ""
        elements = [f"{x:.4f}".rstrip('0').rstrip('.') if not np.isclose(x, round(x)) else f"{int(round(x))}" for x in vector]
        return r"\begin{pmatrix} " + r" \\ ".join(elements) + r" \end{pmatrix}"
    
    def _format_matrix_latex(self, matrix):
        """Format a matrix for LaTeX display."""
        if matrix.size == 0:
            return ""
        rows = []
        for i in range(matrix.shape[0]):
            row = []
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isclose(val, round(val)):
                    row.append(f"{int(round(val))}")
                else:
                    row.append(f"{val:.4f}".rstrip('0').rstrip('.'))
            rows.append(" & ".join(row))
        return r"\begin{pmatrix} " + r" \\\\ ".join(rows) + r" \end{pmatrix}"
    
    def generate_test_vectors(self, dimension: int, num_vectors: int = 3) -> List[np.ndarray]:
        """Generate test vectors for linearity checking."""
        # Use a mix of standard basis vectors and random vectors
        vectors = []
        
        # Add standard basis vectors
        for i in range(min(dimension, num_vectors)):
            basis_vec = np.zeros(dimension)
            basis_vec[i] = 1
            vectors.append(basis_vec)
        
        # Add some simple integer vectors
        if len(vectors) < num_vectors:
            simple_vecs = [
                np.ones(dimension),
                np.array([1 if i % 2 == 0 else -1 for i in range(dimension)]),
                np.array([i + 1 for i in range(dimension)])
            ]
            for vec in simple_vecs:
                if len(vectors) < num_vectors:
                    vectors.append(vec)
        
        return vectors[:num_vectors]
    
    def evaluate_mapping(self, mapping_type: str, formula: str, vector: np.ndarray, 
                        params: Dict = None) -> np.ndarray:
        """Evaluate a mapping function on a given vector."""
        if params is None:
            params = {}
            
        try:
            if mapping_type == "polynomial":
                return self._evaluate_polynomial_mapping(formula, vector)
            elif mapping_type == "trigonometric":
                return self._evaluate_trigonometric_mapping(formula, vector, params.get('phi', 1))
            elif mapping_type == "dot_product":
                return self._evaluate_dot_product_mapping(vector, params.get('c', np.array([1, 1])))
            elif mapping_type == "quadratic":
                return self._evaluate_quadratic_mapping(vector)
            elif mapping_type == "custom":
                return self._evaluate_custom_mapping(formula, vector, params)
            else:
                raise ValueError(f"Unknown mapping type: {mapping_type}")
        except Exception as e:
            st.error(f"Error evaluating mapping: {str(e)}")
            return np.zeros(len(vector))
    
    def _evaluate_polynomial_mapping(self, formula: str, vector: np.ndarray) -> np.ndarray:
        """Evaluate polynomial mapping like L(x,y,z) = 5x - y."""
        # Replace variable names with vector components
        vars_map = {
            'x': vector[0] if len(vector) > 0 else 0,
            'y': vector[1] if len(vector) > 1 else 0,
            'z': vector[2] if len(vector) > 2 else 0,
            'w': vector[3] if len(vector) > 3 else 0
        }
        
        # Split formula by commas or semicolons for multiple outputs
        if ',' in formula or ';' in formula:
            outputs = formula.replace(';', ',').split(',')
        else:
            outputs = [formula]
        
        result = []
        for output in outputs:
            output = output.strip()
            # Create sympy expression and substitute
            x, y, z, w = symbols('x y z w')
            expr = sympify(output)
            value = float(expr.subs(vars_map))
            result.append(value)
        
        return np.array(result)
    
    
    def _evaluate_trigonometric_mapping(self, formula: str, vector: np.ndarray, phi: float) -> np.ndarray:
        """Evaluate trigonometric mapping like L(x,y) = [cos(x)*phi, sin(y)*phi]."""
        x = vector[0] if len(vector) > 0 else 0
        y = vector[1] if len(vector) > 1 else 0
        
        # Parse the formula - expecting format like "cos(x)*phi, sin(y)*phi"
        if ',' in formula or ';' in formula:
            outputs = formula.replace(';', ',').split(',')
        else:
            outputs = [formula]
        
        result = []
        for output in outputs:
            output = output.strip()
            # Replace phi with actual value
            output = output.replace('phi', str(phi))
            # Create expression with trigonometric functions
            expr_str = output.replace('cos', 'sp.cos').replace('sin', 'sp.sin').replace('tan', 'sp.tan')
            expr_str = expr_str.replace('x', str(x)).replace('y', str(y))
            
            # Evaluate using sympy
            result_val = float(sympify(expr_str))
            result.append(result_val)
        
        return np.array(result)
    
    def _evaluate_dot_product_mapping(self, vector: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Evaluate dot product mapping L(v) = [v·c, 0] or similar."""
        dot_product = np.dot(vector, c)
        # Return [v·c, 0] as in the exercise
        return np.array([dot_product, 0])
    
    def _evaluate_quadratic_mapping(self, vector: np.ndarray) -> np.ndarray:
        """Evaluate quadratic mapping L(v) = v·v."""
        dot_product = np.dot(vector, vector)
        return np.array([dot_product])
    
    def _evaluate_custom_mapping(self, formula: str, vector: np.ndarray, params: Dict) -> np.ndarray:
        """Evaluate custom mapping using sympy."""
        # This is a more general evaluator for custom formulas
        vars_map = {f'x{i}': vector[i] if i < len(vector) else 0 for i in range(len(vector))}
        vars_map.update({'x': vector[0] if len(vector) > 0 else 0,
                        'y': vector[1] if len(vector) > 1 else 0,
                        'z': vector[2] if len(vector) > 2 else 0})
        vars_map.update(params)
        
        if ',' in formula or ';' in formula:
            outputs = formula.replace(';', ',').split(',')
        else:
            outputs = [formula]
        
        result = []
        for output in outputs:
            expr = sympify(output.strip())
            value = float(expr.subs(vars_map))
            result.append(value)
        
        return np.array(result)
    
    def check_linearity(self, mapping_type: str, formula: str, dimension: int, 
                       params: Dict = None) -> Dict[str, Any]:
        """
        Check if a mapping is linear by testing additivity and homogeneity.
        Returns detailed results of the linearity test.
        """
        if params is None:
            params = {}
            
        # Generate test vectors
        test_vectors = self.generate_test_vectors(dimension, 4)
        scalars = [2, -1, 0.5, 3]
        
        results = {
            'is_linear': True,
            'additivity_tests': [],
            'homogeneity_tests': [],
            'counterexamples': [],
            'matrix_representation': None
        }
        
        # Test additivity: L(u + v) = L(u) + L(v)
        for i in range(len(test_vectors)):
            for j in range(i + 1, len(test_vectors)):
                u = test_vectors[i]
                v = test_vectors[j]
                
                # Compute L(u + v)
                L_u_plus_v = self.evaluate_mapping(mapping_type, formula, u + v, params)
                
                # Compute L(u) + L(v)
                L_u = self.evaluate_mapping(mapping_type, formula, u, params)
                L_v = self.evaluate_mapping(mapping_type, formula, v, params)
                L_u_plus_L_v = L_u + L_v
                
                # Check if they're equal (within tolerance)
                is_additive = np.allclose(L_u_plus_v, L_u_plus_L_v, rtol=1e-10, atol=1e-10)
                
                test_result = {
                    'vectors': (u, v),
                    'L_u_plus_v': L_u_plus_v,
                    'L_u_plus_L_v': L_u_plus_L_v,
                    'is_additive': is_additive,
                    'difference': L_u_plus_v - L_u_plus_L_v
                }
                
                results['additivity_tests'].append(test_result)
                
                if not is_additive:
                    results['is_linear'] = False
                    results['counterexamples'].append({
                        'type': 'additivity',
                        'vectors': (u, v),
                        'expected': L_u_plus_L_v,
                        'actual': L_u_plus_v,
                        'difference': L_u_plus_v - L_u_plus_L_v
                    })
        
        # Test homogeneity: L(cv) = cL(v)
        for i, scalar in enumerate(scalars):
            for j, vector in enumerate(test_vectors):
                # Compute L(cv)
                L_cv = self.evaluate_mapping(mapping_type, formula, scalar * vector, params)
                
                # Compute cL(v)
                L_v = self.evaluate_mapping(mapping_type, formula, vector, params)
                c_L_v = scalar * L_v
                
                # Check if they're equal
                is_homogeneous = np.allclose(L_cv, c_L_v, rtol=1e-10, atol=1e-10)
                
                test_result = {
                    'scalar': scalar,
                    'vector': vector,
                    'L_cv': L_cv,
                    'c_L_v': c_L_v,
                    'is_homogeneous': is_homogeneous,
                    'difference': L_cv - c_L_v
                }
                
                results['homogeneity_tests'].append(test_result)
                
                if not is_homogeneous:
                    results['is_linear'] = False
                    results['counterexamples'].append({
                        'type': 'homogeneity',
                        'scalar': scalar,
                        'vector': vector,
                        'expected': c_L_v,
                        'actual': L_cv,
                        'difference': L_cv - c_L_v
                    })
        
        # If linear, try to find matrix representation
        if results['is_linear'] and mapping_type in ['polynomial', 'dot_product']:
            results['matrix_representation'] = self.find_matrix_representation(
                mapping_type, formula, dimension, params)
        
        return results
    
    def find_matrix_representation(self, mapping_type: str, formula: str, 
                                 dimension: int, params: Dict = None) -> Optional[np.ndarray]:
        """Find the matrix representation of a linear mapping."""
        if params is None:
            params = {}
            
        try:
            # Apply mapping to standard basis vectors
            basis_vectors = [np.zeros(dimension) for _ in range(dimension)]
            for i in range(dimension):
                basis_vectors[i][i] = 1
            
            # Get the mapping of each basis vector
            mapped_basis = []
            for basis_vec in basis_vectors:
                mapped = self.evaluate_mapping(mapping_type, formula, basis_vec, params)
                mapped_basis.append(mapped)
            
            # The matrix columns are the mapped basis vectors
            if mapped_basis:
                matrix = np.column_stack(mapped_basis)
                return matrix
            
        except Exception as e:
            st.error(f"Error finding matrix representation: {str(e)}")
        
        return None
    
    def render_linearity_checker(self):
        """Render the main linearity checker interface."""
        st.header("🔍 Linear Mapping Analysis")
        st.write("Check if mappings are linear and find their matrix representations.")
        
        # Mapping type selection
        mapping_type = st.selectbox(
            "Select mapping type:",
            ["polynomial", "trigonometric", "dot_product", "quadratic", "custom"],
            help="Choose the type of mapping to analyze"
        )
        
        # Input dimension
        dimension = st.number_input(
            "Input dimension (domain):",
            min_value=1, max_value=10, value=3,
            help="Dimension of the input space (e.g., 3 for R³)"
        )
        
        # Mapping-specific inputs
        formula, params = self._render_mapping_inputs(mapping_type, dimension)
        
        if st.button("🔍 Check Linearity", type="primary"):
            if formula:
                self._perform_linearity_analysis(mapping_type, formula, dimension, params)
            else:
                st.error("Please provide a mapping formula.")
    
    def _render_mapping_inputs(self, mapping_type: str, dimension: int) -> Tuple[str, Dict]:
        """Render input fields specific to the mapping type."""
        formula = ""
        params = {}
        
        if mapping_type == "polynomial":
            st.subheader("Polynomial Mapping")
            st.write("Examples: `5*x - y`, `x + 2*y, -x + z` (comma-separated for multiple outputs)")
            formula = st.text_input(
                "Enter polynomial formula:",
                placeholder="5*x - y",
                help="Use x, y, z, w for variables. Separate multiple outputs with commas."
            )
            
        elif mapping_type == "trigonometric":
            st.subheader("Trigonometric Mapping")
            st.write("Example: `cos(x)*phi, sin(y)*phi`")
            formula = st.text_input(
                "Enter trigonometric formula:",
                placeholder="cos(x)*phi, sin(y)*phi",
                help="Use cos(x), sin(y), etc. Use 'phi' for the parameter."
            )
            params['phi'] = st.number_input("Parameter φ:", value=1.0)
            
        elif mapping_type == "dot_product":
            st.subheader("Dot Product Mapping")
            st.write("Mapping of the form L(v) = [v·c, 0] where c is a fixed vector")
            c_input = st.text_input(
                "Enter vector c (comma-separated):",
                placeholder="5, 2",
                help="Fixed vector for dot product"
            )
            if c_input:
                try:
                    c = np.array([float(x.strip()) for x in c_input.split(',')])
                    params['c'] = c
                    formula = "dot_product_with_c"
                except:
                    st.error("Invalid vector format")
                    
        elif mapping_type == "quadratic":
            st.subheader("Quadratic Form")
            st.write("Mapping L(v) = v·v (dot product of vector with itself)")
            formula = "quadratic_form"
            
        elif mapping_type == "custom":
            st.subheader("Custom Mapping")
            formula = st.text_area(
                "Enter custom formula:",
                placeholder="x**2 + y, x*y + z",
                help="Use mathematical expressions. Separate outputs with commas."
            )
            
            # Additional parameters for custom mappings
            with st.expander("Additional Parameters"):
                param_input = st.text_area(
                    "Parameters (one per line, format: name=value):",
                    placeholder="a=2\nb=3.14"
                )
                if param_input:
                    for line in param_input.strip().split('\n'):
                        if '=' in line:
                            name, value = line.split('=', 1)
                            try:
                                params[name.strip()] = float(value.strip())
                            except:
                                st.warning(f"Could not parse parameter: {line}")
        
        return formula, params
    
    def _perform_linearity_analysis(self, mapping_type: str, formula: str, 
                                   dimension: int, params: Dict):
        """Perform and display linearity analysis."""
        
        with st.spinner("Analyzing linearity..."):
            results = self.check_linearity(mapping_type, formula, dimension, params)
        
        # Display main result
        if results['is_linear']:
            st.success("✅ **The mapping is LINEAR!**")
        else:
            st.error("❌ **The mapping is NOT LINEAR!**")
        
        # Create tabs for detailed results
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "➕ Additivity Tests", "✖️ Homogeneity Tests", "📊 Matrix Representation"])
        
        with tab1:
            self._render_summary_tab(results, mapping_type, formula, params)
        
        with tab2:
            self._render_additivity_tab(results)
        
        with tab3:
            self._render_homogeneity_tab(results)
        
        with tab4:
            self._render_matrix_tab(results, mapping_type, formula)
    
    def _render_summary_tab(self, results: Dict, mapping_type: str, formula: str, params: Dict):
        """Render the summary tab."""
        st.subheader("Analysis Summary")
        
        # Display the mapping
        st.write("**Mapping Definition:**")
        if mapping_type == "polynomial":
            st.latex(f"L(x,y,z) = {formula}")
        elif mapping_type == "trigonometric":
            phi = params.get('phi', 1)
            st.latex(f"L(x,y) = {formula.replace('phi', str(phi))}")
        elif mapping_type == "dot_product":
            c = params.get('c', np.array([1, 1]))
            st.latex(f"L(\\vec{{v}}) = \\begin{{pmatrix}} \\vec{{v}} \\cdot {c.tolist()} \\\\ 0 \\end{{pmatrix}}")
        elif mapping_type == "quadratic":
            st.latex("L(\\vec{v}) = \\vec{v} \\cdot \\vec{v}")
        else:
            st.code(formula)
        
        # Test results summary
        col1, col2 = st.columns(2)
        
        with col1:
            additivity_passed = all(test['is_additive'] for test in results['additivity_tests'])
            if additivity_passed:
                st.success(f"✅ Additivity: Passed ({len(results['additivity_tests'])} tests)")
            else:
                failed_count = sum(1 for test in results['additivity_tests'] if not test['is_additive'])
                st.error(f"❌ Additivity: Failed ({failed_count}/{len(results['additivity_tests'])} tests)")
        
        with col2:
            homogeneity_passed = all(test['is_homogeneous'] for test in results['homogeneity_tests'])
            if homogeneity_passed:
                st.success(f"✅ Homogeneity: Passed ({len(results['homogeneity_tests'])} tests)")
            else:
                failed_count = sum(1 for test in results['homogeneity_tests'] if not test['is_homogeneous'])
                st.error(f"❌ Homogeneity: Failed ({failed_count}/{len(results['homogeneity_tests'])} tests)")
        
        # Show counterexamples if any
        if results['counterexamples']:
            st.subheader("🚫 Counterexamples")
            for i, counterexample in enumerate(results['counterexamples'][:3]):  # Show first 3
                with st.expander(f"Counterexample {i+1}: {counterexample['type'].title()} failure"):
                    if counterexample['type'] == 'additivity':
                        u, v = counterexample['vectors']
                        st.write(f"**Vectors:** u = {u.tolist()}, v = {v.tolist()}")
                        st.write(f"**L(u + v)** = {counterexample['actual'].tolist()}")
                        st.write(f"**L(u) + L(v)** = {counterexample['expected'].tolist()}")
                        st.write(f"**Difference:** {counterexample['difference'].tolist()}")
                    else:  # homogeneity
                        st.write(f"**Scalar:** c = {counterexample['scalar']}")
                        st.write(f"**Vector:** v = {counterexample['vector'].tolist()}")
                        st.write(f"**L(cv)** = {counterexample['actual'].tolist()}")
                        st.write(f"**cL(v)** = {counterexample['expected'].tolist()}")
                        st.write(f"**Difference:** {counterexample['difference'].tolist()}")
    
    def _render_additivity_tab(self, results: Dict):
        """Render the additivity tests tab."""
        st.subheader("Additivity Tests: L(u + v) = L(u) + L(v)")
        
        for i, test in enumerate(results['additivity_tests'][:6]):  # Show first 6 tests
            u, v = test['vectors']
            
            with st.expander(f"Test {i+1}: u = {u.tolist()}, v = {v.tolist()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Left side: L(u + v)**")
                    st.write(f"u + v = {(u + v).tolist()}")
                    st.write(f"L(u + v) = {test['L_u_plus_v'].tolist()}")
                
                with col2:
                    st.write("**Right side: L(u) + L(v)**")
                    st.write(f"L(u) = {(test['L_u_plus_L_v'] - self.evaluate_mapping('temp', '', v, {})).tolist()}")
                    st.write(f"L(v) = calculation needed")
                    st.write(f"L(u) + L(v) = {test['L_u_plus_L_v'].tolist()}")
                
                if test['is_additive']:
                    st.success("✅ **PASSED** - Additivity holds")
                else:
                    st.error("❌ **FAILED** - Additivity violation")
                    st.write(f"**Difference:** {test['difference'].tolist()}")
    
    def _render_homogeneity_tab(self, results: Dict):
        """Render the homogeneity tests tab."""
        st.subheader("Homogeneity Tests: L(cv) = cL(v)")
        
        for i, test in enumerate(results['homogeneity_tests'][:8]):  # Show first 8 tests
            with st.expander(f"Test {i+1}: c = {test['scalar']}, v = {test['vector'].tolist()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Left side: L(cv)**")
                    st.write(f"cv = {(test['scalar'] * test['vector']).tolist()}")
                    st.write(f"L(cv) = {test['L_cv'].tolist()}")
                
                with col2:
                    st.write("**Right side: cL(v)**")
                    st.write(f"L(v) = {(test['c_L_v'] / test['scalar'] if test['scalar'] != 0 else test['c_L_v']).tolist()}")
                    st.write(f"cL(v) = {test['c_L_v'].tolist()}")
                
                if test['is_homogeneous']:
                    st.success("✅ **PASSED** - Homogeneity holds")
                else:
                    st.error("❌ **FAILED** - Homogeneity violation")
                    st.write(f"**Difference:** {test['difference'].tolist()}")
    
    def _render_matrix_tab(self, results: Dict, mapping_type: str, formula: str):
        """Render the matrix representation tab."""
        st.subheader("Matrix Representation")
        
        if results['is_linear']:
            if results['matrix_representation'] is not None:
                matrix = results['matrix_representation']
                st.success("✅ **Matrix representation found!**")
                st.write("**Standard basis representation:**")
                st.latex(f"[L] = {self._format_matrix_latex(matrix)}")
                
                # Show verification
                st.write("**Verification:** This matrix represents the linear transformation in the standard basis.")
                st.write("Each column shows where the corresponding standard basis vector is mapped.")
                
                # Display as dataframe for easier reading
                df = pd.DataFrame(matrix)
                df.columns = [f"e{i+1}" for i in range(matrix.shape[1])]
                df.index = [f"Output {i+1}" for i in range(matrix.shape[0])]
                st.dataframe(df, use_container_width=True)
                
            else:
                st.warning("⚠️ The mapping is linear, but matrix representation could not be determined automatically.")
                st.write("This might happen for complex mappings or infinite-dimensional cases.")
        else:
            st.error("❌ No matrix representation exists because the mapping is not linear.")
            st.write("Only linear mappings can be represented as matrices.")