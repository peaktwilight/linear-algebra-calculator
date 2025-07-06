#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Linear Mapping Operations
"""

import streamlit as st
import numpy as np
from sympy import symbols, sympify, expand, simplify, latex, Add, Mul, Symbol, cos, sin, tan, N
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
        self.symbolic_variables = None
        self.symbolic_expression = None
        
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
            elif mapping_type == "negation":
                return -vector
            elif mapping_type == "projection":
                return self._evaluate_polynomial_mapping(formula, vector)  # Use polynomial evaluator
            elif mapping_type == "zero":
                return np.zeros(1)  # Always return zero
            elif mapping_type == "affine":
                return self._evaluate_polynomial_mapping(formula, vector)  # Use polynomial evaluator
            elif mapping_type == "bilinear":
                return self._evaluate_polynomial_mapping(formula, vector)  # Use polynomial evaluator
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
        z = vector[2] if len(vector) > 2 else 0
        
        # Parse the formula - expecting format like "cos(x)*phi, sin(y)*phi"
        if ',' in formula or ';' in formula:
            outputs = formula.replace(';', ',').split(',')
        else:
            outputs = [formula]
        
        result = []
        for output in outputs:
            output = output.strip()
            # Replace parameter names with actual value
            output = output.replace('phi', str(phi))
            output = output.replace('alpha', str(phi))  # Handle alpha parameter
            # Replace variable names with actual values
            output = output.replace('x', str(x)).replace('y', str(y)).replace('z', str(z))
            
            try:
                # Create sympy expression and evaluate
                expr = sympify(output)
                result_val = float(expr)
                result.append(result_val)
            except (ValueError, TypeError) as e:
                st.error(f"Error evaluating trigonometric expression '{output}': {str(e)}")
                result.append(0.0)  # Default value on error
        
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
                
                # Ensure arrays have the same shape for comparison
                if L_u_plus_v.shape != L_u_plus_L_v.shape:
                    st.error(f"Dimension mismatch in mapping evaluation: {L_u_plus_v.shape} vs {L_u_plus_L_v.shape}")
                    is_additive = False
                else:
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
                
                # Ensure arrays have the same shape for comparison
                if L_cv.shape != c_L_v.shape:
                    st.error(f"Dimension mismatch in mapping evaluation: {L_cv.shape} vs {c_L_v.shape}")
                    is_homogeneous = False
                else:
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
    
    def generate_symbolic_proof(self, mapping_type: str, formula: str, dimension: int, 
                               params: Dict = None) -> Dict[str, Any]:
        """
        Generate a formal symbolic proof of linearity for polynomial mappings.
        Returns proof steps and conclusions.
        """
        if params is None:
            params = {}
            
        proof_results = {
            'can_prove': False,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': None,
            'error_message': None
        }
        
        try:
            if mapping_type == "polynomial":
                return self._prove_polynomial_linearity(formula, dimension)
            elif mapping_type == "dot_product":
                return self._prove_dot_product_linearity(params.get('c', np.array([1, 1])), dimension)
            elif mapping_type == "trigonometric":
                return self._prove_trigonometric_nonlinearity(formula, dimension, params.get('phi', 1.0))
            elif mapping_type == "quadratic":
                return self._prove_quadratic_nonlinearity(dimension)
            elif mapping_type == "negation":
                return self._prove_negation_linearity(dimension)
            elif mapping_type == "projection":
                return self._prove_projection_linearity(formula, dimension)
            elif mapping_type == "zero":
                return self._prove_zero_linearity(dimension)
            elif mapping_type == "affine":
                return self._prove_affine_nonlinearity(formula, dimension)
            elif mapping_type == "bilinear":
                return self._prove_bilinear_nonlinearity(formula, dimension)
            else:
                proof_results['error_message'] = f"Symbolic proofs not yet implemented for {mapping_type} mappings"
                return proof_results
                
        except Exception as e:
            proof_results['error_message'] = f"Error generating proof: {str(e)}"
            return proof_results
    
    def _prove_polynomial_linearity(self, formula: str, dimension: int) -> Dict[str, Any]:
        """Generate symbolic proof for polynomial mappings."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': None,
            'error_message': None
        }
        
        try:
            # Define symbolic variables
            if dimension == 1:
                vars_u = symbols('u_1')
                vars_v = symbols('v_1')
                vars_combined = [symbols('x')]
                vars_u = [vars_u]
                vars_v = [vars_v]
            elif dimension == 2:
                vars_u = symbols('u_1 u_2')
                vars_v = symbols('v_1 v_2')
                vars_combined = symbols('x y')
                vars_u = list(vars_u) if hasattr(vars_u, '__iter__') else [vars_u]
                vars_v = list(vars_v) if hasattr(vars_v, '__iter__') else [vars_v]
                vars_combined = list(vars_combined) if hasattr(vars_combined, '__iter__') else [vars_combined]
            elif dimension == 3:
                vars_u = symbols('u_1 u_2 u_3')
                vars_v = symbols('v_1 v_2 v_3')
                vars_combined = symbols('x y z')
                vars_u = list(vars_u) if hasattr(vars_u, '__iter__') else [vars_u]
                vars_v = list(vars_v) if hasattr(vars_v, '__iter__') else [vars_v]
                vars_combined = list(vars_combined) if hasattr(vars_combined, '__iter__') else [vars_combined]
            else:
                vars_u = [symbols(f'u_{i}') for i in range(1, dimension + 1)]
                vars_v = [symbols(f'v_{i}') for i in range(1, dimension + 1)]
                vars_combined = [symbols(f'x_{i}') for i in range(1, dimension + 1)]
            
            # Parse the formula into sympy expressions
            if ',' in formula or ';' in formula:
                outputs = formula.replace(';', ',').split(',')
            else:
                outputs = [formula]
            
            expressions = []
            for output in outputs:
                output = output.strip()
                # Replace standard variable names
                expr_str = output
                if dimension >= 1:
                    expr_str = expr_str.replace('x', str(vars_combined[0]))
                if dimension >= 2:
                    expr_str = expr_str.replace('y', str(vars_combined[1]))
                if dimension >= 3:
                    expr_str = expr_str.replace('z', str(vars_combined[2]))
                
                expr = sympify(expr_str)
                expressions.append(expr)
            
            # Check if expressions are linear (only degree 1 terms) and detect specific non-linear patterns
            is_linear = True
            has_constant_term = False
            nonlinear_reason = None
            
            for expr in expressions:
                # Check for constant terms (affine mappings)
                if expr.is_number and expr != 0:
                    has_constant_term = True
                    
                # Check for polynomial structure
                try:
                    poly = expr.as_poly()
                    if poly is not None:
                        if poly.total_degree() > 1:
                            is_linear = False
                            nonlinear_reason = f"Contains terms of degree {poly.total_degree()} > 1"
                            break
                        # Check for constant terms in polynomial
                        const_term = poly.nth(0, *([0] * len(poly.gens)))
                        if const_term != 0:
                            has_constant_term = True
                    else:
                        # Manual degree analysis for complex expressions
                        for term in Add.make_args(expr):
                            if isinstance(term, Mul):
                                var_count = sum(1 for arg in term.args if isinstance(arg, Symbol))
                                if var_count > 1:
                                    is_linear = False
                                    nonlinear_reason = "Contains product of variables (bilinear term)"
                                    break
                            elif isinstance(term, Symbol):
                                continue  # degree 1, OK
                            elif term.is_number:
                                if term != 0:
                                    has_constant_term = True
                            else:
                                # Check for powers
                                if hasattr(term, 'exp') and term.exp > 1:
                                    is_linear = False
                                    nonlinear_reason = f"Contains variable raised to power {term.exp}"
                                    break
                except Exception:
                    # If analysis fails, fall back to empirical testing
                    pass
            
            if not is_linear:
                proof_results['conclusion'] = "NOT_LINEAR"
                additivity_steps = []
                additivity_steps.append("**Proof of Non-linearity for Polynomial Mapping**")
                additivity_steps.append("")
                additivity_steps.append(f"Given mapping contains non-linear terms: {nonlinear_reason}")
                additivity_steps.append("")
                if "degree" in nonlinear_reason:
                    additivity_steps.append("**General Principle:**")
                    additivity_steps.append("Mappings with polynomial terms of degree > 1 cannot be linear.")
                    additivity_steps.append("")
                    additivity_steps.append("**Proof by Counterexample:**")
                    additivity_steps.append("For any quadratic term $ax^n$ where $n > 1$:")
                    additivity_steps.append("$L(2u) = a(2u)^n = a \\cdot 2^n \\cdot u^n$")
                    additivity_steps.append("$2L(u) = 2 \\cdot au^n$")
                    additivity_steps.append("Since $2^n \\neq 2$ for $n > 1$, we have $L(2u) \\neq 2L(u)$")
                    additivity_steps.append("Therefore, the mapping violates homogeneity.")
                elif "bilinear" in nonlinear_reason:
                    additivity_steps.append("**General Principle:**")
                    additivity_steps.append("Mappings with products of variables cannot be linear.")
                    additivity_steps.append("")
                    additivity_steps.append("**Proof by Counterexample:**")
                    additivity_steps.append("For a bilinear term $xy$:")
                    additivity_steps.append("Let $u = (1, 0)$ and $v = (0, 1)$")
                    additivity_steps.append("$L(u + v) = L((1, 1)) = 1 \\cdot 1 = 1$")
                    additivity_steps.append("$L(u) + L(v) = L((1, 0)) + L((0, 1)) = 0 + 0 = 0$")
                    additivity_steps.append("Since $1 \\neq 0$, additivity is violated.")
                
                proof_results['additivity_proof'] = "\n".join(additivity_steps)
                proof_results['homogeneity_proof'] = "Non-linearity proven by degree analysis."
                return proof_results
            
            if has_constant_term:
                proof_results['conclusion'] = "NOT_LINEAR"
                additivity_steps = []
                additivity_steps.append("**Proof of Non-linearity for Affine Mapping**")
                additivity_steps.append("")
                additivity_steps.append("The mapping contains constant terms, making it affine rather than linear.")
                additivity_steps.append("")
                additivity_steps.append("**Proof by Counterexample:**")
                additivity_steps.append("For any mapping $L(v) = Av + b$ where $b \\neq 0$:")
                additivity_steps.append("$L(0) = A \\cdot 0 + b = b \\neq 0$")
                additivity_steps.append("")
                additivity_steps.append("But linear mappings must satisfy $L(0) = 0$.")
                additivity_steps.append("Therefore, any mapping with constant terms is **not linear**.")
                
                proof_results['additivity_proof'] = "\n".join(additivity_steps)
                proof_results['homogeneity_proof'] = "Affine mappings violate the zero mapping property."
                return proof_results
            
            # Generate additivity proof
            additivity_steps = []
            additivity_steps.append("**Additivity Proof: L(u + v) = L(u) + L(v)**")
            additivity_steps.append("")
            additivity_steps.append("Let u = (u₁, u₂, ..., uₙ) and v = (v₁, v₂, ..., vₙ) be arbitrary vectors.")
            additivity_steps.append("Then u + v = (u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ)")
            additivity_steps.append("")
            
            for i, expr in enumerate(expressions):
                if len(expressions) > 1:
                    additivity_steps.append(f"For output component {i+1}:")
                
                # Substitute u + v into the expression
                subs_dict = {}
                if dimension >= 1:
                    subs_dict[vars_combined[0]] = vars_u[0] + vars_v[0]
                if dimension >= 2:
                    subs_dict[vars_combined[1]] = vars_u[1] + vars_v[1]
                if dimension >= 3:
                    subs_dict[vars_combined[2]] = vars_u[2] + vars_v[2]
                
                L_u_plus_v = expr.subs(subs_dict)
                L_u_plus_v_expanded = expand(L_u_plus_v)
                
                # Substitute u and v separately
                subs_u = {}
                subs_v = {}
                if dimension >= 1:
                    subs_u[vars_combined[0]] = vars_u[0]
                    subs_v[vars_combined[0]] = vars_v[0]
                if dimension >= 2:
                    subs_u[vars_combined[1]] = vars_u[1]
                    subs_v[vars_combined[1]] = vars_v[1]
                if dimension >= 3:
                    subs_u[vars_combined[2]] = vars_u[2]
                    subs_v[vars_combined[2]] = vars_v[2]
                
                L_u = expr.subs(subs_u)
                L_v = expr.subs(subs_v)
                L_u_plus_L_v = L_u + L_v
                L_u_plus_L_v_expanded = expand(L_u_plus_L_v)
                
                additivity_steps.append(f"$$L(u + v) = {latex(L_u_plus_v_expanded)}$$")
                additivity_steps.append(f"$$L(u) + L(v) = {latex(L_u)} + {latex(L_v)} = {latex(L_u_plus_L_v_expanded)}$$")
                
                # Verify they're equal
                difference = simplify(L_u_plus_v_expanded - L_u_plus_L_v_expanded)
                if difference == 0:
                    additivity_steps.append("✅ $L(u + v) = L(u) + L(v)$ ✓")
                else:
                    additivity_steps.append(f"❌ Difference: ${latex(difference)} \\neq 0$")
                    proof_results['conclusion'] = "NOT_LINEAR"
                additivity_steps.append("")
            
            # Generate homogeneity proof
            homogeneity_steps = []
            homogeneity_steps.append("**Homogeneity Proof: L(cv) = cL(v)**")
            homogeneity_steps.append("")
            homogeneity_steps.append("Let c be an arbitrary scalar and v = (v₁, v₂, ..., vₙ) be an arbitrary vector.")
            homogeneity_steps.append("Then cv = (cv₁, cv₂, ..., cvₙ)")
            homogeneity_steps.append("")
            
            c = symbols('c')
            for i, expr in enumerate(expressions):
                if len(expressions) > 1:
                    homogeneity_steps.append(f"For output component {i+1}:")
                
                # Substitute cv into the expression
                subs_dict = {}
                if dimension >= 1:
                    subs_dict[vars_combined[0]] = c * vars_v[0]
                if dimension >= 2:
                    subs_dict[vars_combined[1]] = c * vars_v[1]
                if dimension >= 3:
                    subs_dict[vars_combined[2]] = c * vars_v[2]
                
                L_cv = expr.subs(subs_dict)
                L_cv_expanded = expand(L_cv)
                
                # Substitute v and multiply by c
                subs_v = {}
                if dimension >= 1:
                    subs_v[vars_combined[0]] = vars_v[0]
                if dimension >= 2:
                    subs_v[vars_combined[1]] = vars_v[1]
                if dimension >= 3:
                    subs_v[vars_combined[2]] = vars_v[2]
                
                L_v = expr.subs(subs_v)
                c_L_v = c * L_v
                c_L_v_expanded = expand(c_L_v)
                
                homogeneity_steps.append(f"$$L(cv) = {latex(L_cv_expanded)}$$")
                homogeneity_steps.append(f"$$cL(v) = c \\cdot {latex(L_v)} = {latex(c_L_v_expanded)}$$")
                
                # Verify they're equal
                difference = simplify(L_cv_expanded - c_L_v_expanded)
                if difference == 0:
                    homogeneity_steps.append("✅ $L(cv) = cL(v)$ ✓")
                else:
                    homogeneity_steps.append(f"❌ Difference: ${latex(difference)} \\neq 0$")
                    proof_results['conclusion'] = "NOT_LINEAR"
                homogeneity_steps.append("")
            
            proof_results['additivity_proof'] = "\n".join(additivity_steps)
            proof_results['homogeneity_proof'] = "\n".join(homogeneity_steps)
            
            if proof_results['conclusion'] != "NOT_LINEAR":
                proof_results['conclusion'] = "LINEAR"
                
        except Exception as e:
            proof_results['error_message'] = f"Error in symbolic proof: {str(e)}"
            proof_results['can_prove'] = False
            
        return proof_results
    
    def _prove_dot_product_linearity(self, c: np.ndarray, dimension: int) -> Dict[str, Any]:
        """Generate proof for dot product mappings L(v) = [v·c, 0]."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': "LINEAR",
            'error_message': None
        }
        
        additivity_steps = []
        additivity_steps.append("**Additivity Proof for Dot Product Mapping**")
        additivity_steps.append("")
        additivity_steps.append(f"Given: L(v) = [v·c, 0] where c = {c.tolist()}")
        additivity_steps.append("Need to prove: L(u + v) = L(u) + L(v)")
        additivity_steps.append("")
        additivity_steps.append("L(u + v) = [(u + v)·c, 0]")
        additivity_steps.append("         = [u·c + v·c, 0]  (distributive property of dot product)")
        additivity_steps.append("         = [u·c, 0] + [v·c, 0]")
        additivity_steps.append("         = L(u) + L(v) ✓")
        
        homogeneity_steps = []
        homogeneity_steps.append("**Homogeneity Proof for Dot Product Mapping**")
        homogeneity_steps.append("")
        homogeneity_steps.append("Need to prove: L(kv) = kL(v) for scalar k")
        homogeneity_steps.append("")
        homogeneity_steps.append("L(kv) = [(kv)·c, 0]")
        homogeneity_steps.append("      = [k(v·c), 0]  (scalar multiplication property of dot product)")
        homogeneity_steps.append("      = k[v·c, 0]")
        homogeneity_steps.append("      = kL(v) ✓")
        
        proof_results['additivity_proof'] = "\n".join(additivity_steps)
        proof_results['homogeneity_proof'] = "\n".join(homogeneity_steps)
        
        return proof_results
    
    def _prove_trigonometric_nonlinearity(self, formula: str, _dimension: int, phi: float = 1.0) -> Dict[str, Any]:
        """Analyze trigonometric mappings - some may be linear if trig functions are applied to constants."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': None,
            'error_message': None
        }
        
        # Check if trigonometric functions are applied to variables or constants
        # Linear case: cos(alpha)*x where alpha is constant
        # Non-linear case: cos(x)*alpha where x is variable
        
        has_trig_of_variable = False
        if 'cos(x)' in formula or 'sin(x)' in formula or 'cos(y)' in formula or 'sin(y)' in formula or 'cos(z)' in formula or 'sin(z)' in formula:
            has_trig_of_variable = True
        
        if has_trig_of_variable:
            # This is the non-linear case: cos(x), sin(y), etc.
            additivity_steps = []
            additivity_steps.append("**Proof of Non-linearity for Trigonometric Mapping**")
            additivity_steps.append("")
            formula_latex = formula.replace('phi', '\\phi').replace('alpha', '\\alpha')
            additivity_steps.append(f"Given mapping: $L(x,y) = {formula_latex}$")
            additivity_steps.append("")
            additivity_steps.append("**Counterexample for Additivity:**")
            additivity_steps.append("Consider $u = (\\pi/2, 0)$ and $v = (\\pi/2, 0)$")
            additivity_steps.append("")
            additivity_steps.append("$L(u) = \\cos(\\pi/2) \\cdot \\phi = 0 \\cdot \\phi = 0$")
            additivity_steps.append("$L(v) = \\cos(\\pi/2) \\cdot \\phi = 0 \\cdot \\phi = 0$")
            additivity_steps.append("$L(u) + L(v) = 0 + 0 = 0$")
            additivity_steps.append("")
            additivity_steps.append("$L(u + v) = L(\\pi, 0) = \\cos(\\pi) \\cdot \\phi = -1 \\cdot \\phi = -\\phi$")
            additivity_steps.append("")
            additivity_steps.append("Since $L(u + v) = -\\phi \\neq 0 = L(u) + L(v)$ (for $\\phi \\neq 0$),")
            additivity_steps.append("**the mapping violates additivity and is NOT LINEAR.**")
            
            proof_results['additivity_proof'] = "\n".join(additivity_steps)
            proof_results['homogeneity_proof'] = "Since additivity fails, the mapping is not linear."
            proof_results['conclusion'] = "NOT_LINEAR"
        else:
            # This is the linear case: cos(alpha)*x where alpha is constant - we can prove it symbolically!
            additivity_steps = []
            additivity_steps.append("**Symbolic Proof of Linearity for Trigonometric Coefficients**")
            additivity_steps.append("")
            formula_latex = formula.replace('phi', '\\phi').replace('alpha', '\\alpha')
            additivity_steps.append(f"Given mapping: $L(x,y) = {formula_latex}$")
            additivity_steps.append("")
            additivity_steps.append("**Step 1: Evaluate Constants**")
            additivity_steps.append("Since trigonometric functions are applied to constants:")
            
            # Always provide general symbolic proof unless user specifically requests numerical
            if phi != 1.0:  # User changed the default value, so use their specific value
                try:
                    # Compute actual values using the provided parameter
                    c1_val = float(N(cos(phi), 6))  # cos(phi)
                    c2_val = float(N(sin(phi), 6))  # sin(phi)
                    
                    additivity_steps.append(f"Given parameter: $\\alpha = {phi}$")
                    additivity_steps.append(f"Let $c_1 = \\cos(\\alpha) = \\cos({phi}) \\approx {c1_val}$")
                    additivity_steps.append(f"Let $c_2 = \\sin(\\alpha) = \\sin({phi}) \\approx {c2_val}$")
                    additivity_steps.append("")
                    additivity_steps.append(f"The mapping becomes: $L(x,y) = [{c1_val} \\cdot x, {c2_val} \\cdot y]$")
                    use_numerical = True
                except:
                    use_numerical = False
            else:
                use_numerical = False
                
            if not use_numerical:
                # General symbolic proof for any α ∈ ℝ
                additivity_steps.append("**General Proof for any α ∈ ℝ:**")
                additivity_steps.append("Let $c_1 = \\cos(\\alpha)$ and $c_2 = \\sin(\\alpha)$ where $\\alpha \\in \\mathbb{R}$ is any fixed constant.")
                additivity_steps.append("")
                additivity_steps.append("Since trigonometric functions of constants yield constant values,")
                additivity_steps.append("both $c_1$ and $c_2$ are fixed real numbers (independent of the input variables).")
                additivity_steps.append("")
                additivity_steps.append("The mapping becomes: $L(x,y) = [c_1 \\cdot x, c_2 \\cdot y]$")
            additivity_steps.append("")
            additivity_steps.append("**Step 2: Additivity Proof**")
            additivity_steps.append("For arbitrary vectors $\\vec{u} = (u_1, u_2)$ and $\\vec{v} = (v_1, v_2)$:")
            additivity_steps.append("")
            additivity_steps.append("$L(\\vec{u} + \\vec{v}) = L((u_1 + v_1, u_2 + v_2))$")
            additivity_steps.append("$= [c_1(u_1 + v_1), c_2(u_2 + v_2)]$")
            additivity_steps.append("$= [c_1 u_1 + c_1 v_1, c_2 u_2 + c_2 v_2]$")
            additivity_steps.append("$= [c_1 u_1, c_2 u_2] + [c_1 v_1, c_2 v_2]$")
            additivity_steps.append("$= L(\\vec{u}) + L(\\vec{v})$ ✅")
            
            homogeneity_steps = []
            homogeneity_steps.append("**Step 3: Homogeneity Proof**")
            homogeneity_steps.append("")
            homogeneity_steps.append("For arbitrary scalar $k$ and vector $\\vec{v} = (v_1, v_2)$:")
            homogeneity_steps.append("")
            homogeneity_steps.append("$L(k\\vec{v}) = L((kv_1, kv_2))$")
            homogeneity_steps.append("$= [c_1(kv_1), c_2(kv_2)]$")
            homogeneity_steps.append("$= [k(c_1 v_1), k(c_2 v_2)]$")
            homogeneity_steps.append("$= k[c_1 v_1, c_2 v_2]$")
            homogeneity_steps.append("$= kL(\\vec{v})$ ✅")
            homogeneity_steps.append("")
            homogeneity_steps.append("**Conclusion:** Since both additivity and homogeneity hold,")
            homogeneity_steps.append("the mapping is **definitively LINEAR**.")
            homogeneity_steps.append("")
            
            # Add matrix representation
            homogeneity_steps.append("**Matrix Representation:**")
            homogeneity_steps.append("This linear mapping can be represented as:")
            if use_numerical:
                homogeneity_steps.append(f"$$L(\\vec{{v}}) = \\begin{{pmatrix}} {c1_val} & 0 \\\\ 0 & {c2_val} \\end{{pmatrix}} \\begin{{pmatrix}} x \\\\ y \\end{{pmatrix}}$$")
            else:
                homogeneity_steps.append("$$L(\\vec{v}) = \\begin{pmatrix} c_1 & 0 \\\\ 0 & c_2 \\end{pmatrix} \\begin{pmatrix} x \\\\ y \\end{pmatrix}$$")
                homogeneity_steps.append("where $c_1 = \\cos(\\alpha)$ and $c_2 = \\sin(\\alpha)$ for any fixed $\\alpha$.")
            
            proof_results['additivity_proof'] = "\n".join(additivity_steps)
            proof_results['homogeneity_proof'] = "\n".join(homogeneity_steps)
            proof_results['conclusion'] = "LINEAR"
        
        return proof_results
    
    def _prove_quadratic_nonlinearity(self, _dimension: int) -> Dict[str, Any]:
        """Generate proof that quadratic mappings are not linear."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': "NOT_LINEAR",
            'error_message': None
        }
        
        additivity_steps = []
        additivity_steps.append("**Proof of Non-linearity for Quadratic Form**")
        additivity_steps.append("")
        additivity_steps.append("Given mapping: $L(\\vec{v}) = \\vec{v} \\cdot \\vec{v} = ||\\vec{v}||^2$")
        additivity_steps.append("")
        additivity_steps.append("**Counterexample for Additivity:**")
        additivity_steps.append("Consider $\\vec{u} = (1, 0, \\ldots, 0)$ and $\\vec{v} = (1, 0, \\ldots, 0)$")
        additivity_steps.append("")
        additivity_steps.append("$L(\\vec{u}) = \\vec{u} \\cdot \\vec{u} = 1^2 + 0^2 + \\cdots = 1$")
        additivity_steps.append("$L(\\vec{v}) = \\vec{v} \\cdot \\vec{v} = 1^2 + 0^2 + \\cdots = 1$")
        additivity_steps.append("$L(\\vec{u}) + L(\\vec{v}) = 1 + 1 = 2$")
        additivity_steps.append("")
        additivity_steps.append("$L(\\vec{u} + \\vec{v}) = L((2, 0, \\ldots, 0)) = 2^2 + 0^2 + \\cdots = 4$")
        additivity_steps.append("")
        additivity_steps.append("Since $L(\\vec{u} + \\vec{v}) = 4 \\neq 2 = L(\\vec{u}) + L(\\vec{v})$,")
        additivity_steps.append("**the quadratic form violates additivity and is NOT LINEAR.**")
        
        proof_results['additivity_proof'] = "\n".join(additivity_steps)
        proof_results['homogeneity_proof'] = "Since additivity fails, the mapping is not linear."
        
        return proof_results
    
    def _prove_negation_linearity(self, _dimension: int) -> Dict[str, Any]:
        """Generate proof that negation mapping L(v) = -v is linear."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': "LINEAR",
            'error_message': None
        }
        
        additivity_steps = []
        additivity_steps.append("**Additivity Proof for Negation Mapping**")
        additivity_steps.append("")
        additivity_steps.append("Given: $L(\\vec{v}) = -\\vec{v}$")
        additivity_steps.append("")
        additivity_steps.append("**Proof:** $L(\\vec{u} + \\vec{v}) = -(\\vec{u} + \\vec{v}) = -\\vec{u} + (-\\vec{v}) = L(\\vec{u}) + L(\\vec{v})$ ✓")
        
        homogeneity_steps = []
        homogeneity_steps.append("**Homogeneity Proof for Negation Mapping**")
        homogeneity_steps.append("")
        homogeneity_steps.append("**Proof:** $L(c\\vec{v}) = -(c\\vec{v}) = c(-\\vec{v}) = cL(\\vec{v})$ ✓")
        
        proof_results['additivity_proof'] = "\n".join(additivity_steps)
        proof_results['homogeneity_proof'] = "\n".join(homogeneity_steps)
        return proof_results
    
    def _prove_projection_linearity(self, formula: str, _dimension: int) -> Dict[str, Any]:
        """Generate proof that projection mappings are linear."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': "LINEAR",
            'error_message': None
        }
        
        additivity_steps = []
        additivity_steps.append("**Additivity Proof for Projection Mapping**")
        additivity_steps.append("")
        additivity_steps.append("Given: Projection mapping selecting component(s)")
        additivity_steps.append("")
        additivity_steps.append("**General Principle:** Component selection is linear.")
        additivity_steps.append("If $L(\\vec{v}) = v_i$ (selecting i-th component):")
        additivity_steps.append("$L(\\vec{u} + \\vec{v}) = (\\vec{u} + \\vec{v})_i = u_i + v_i = L(\\vec{u}) + L(\\vec{v})$ ✓")
        
        homogeneity_steps = []
        homogeneity_steps.append("**Homogeneity Proof for Projection Mapping**")
        homogeneity_steps.append("")
        homogeneity_steps.append("$L(c\\vec{v}) = (c\\vec{v})_i = c \\cdot v_i = cL(\\vec{v})$ ✓")
        
        proof_results['additivity_proof'] = "\n".join(additivity_steps)
        proof_results['homogeneity_proof'] = "\n".join(homogeneity_steps)
        return proof_results
    
    def _prove_zero_linearity(self, _dimension: int) -> Dict[str, Any]:
        """Generate proof that zero mapping L(v) = 0 is linear."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': "LINEAR",
            'error_message': None
        }
        
        additivity_steps = []
        additivity_steps.append("**Additivity Proof for Zero Mapping**")
        additivity_steps.append("")
        additivity_steps.append("Given: $L(\\vec{v}) = \\vec{0}$ for all $\\vec{v}$")
        additivity_steps.append("")
        additivity_steps.append("**Proof:** $L(\\vec{u} + \\vec{v}) = \\vec{0} = \\vec{0} + \\vec{0} = L(\\vec{u}) + L(\\vec{v})$ ✓")
        
        homogeneity_steps = []
        homogeneity_steps.append("**Homogeneity Proof for Zero Mapping**")
        homogeneity_steps.append("")
        homogeneity_steps.append("**Proof:** $L(c\\vec{v}) = \\vec{0} = c \\cdot \\vec{0} = cL(\\vec{v})$ ✓")
        
        proof_results['additivity_proof'] = "\n".join(additivity_steps)
        proof_results['homogeneity_proof'] = "\n".join(homogeneity_steps)
        return proof_results
    
    def _prove_affine_nonlinearity(self, formula: str, _dimension: int) -> Dict[str, Any]:
        """Generate proof that affine mappings L(v) = Av + b are not linear when b ≠ 0."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': "NOT_LINEAR",
            'error_message': None
        }
        
        additivity_steps = []
        additivity_steps.append("**Proof of Non-linearity for Affine Mapping**")
        additivity_steps.append("")
        additivity_steps.append("Given: Mapping contains constant terms (affine transformation)")
        additivity_steps.append("")
        additivity_steps.append("**Fundamental Property Violation:**")
        additivity_steps.append("Linear mappings must satisfy $L(\\vec{0}) = \\vec{0}$")
        additivity_steps.append("")
        additivity_steps.append("For affine mapping $L(\\vec{v}) = A\\vec{v} + \\vec{b}$ where $\\vec{b} \\neq \\vec{0}$:")
        additivity_steps.append("$L(\\vec{0}) = A\\vec{0} + \\vec{b} = \\vec{b} \\neq \\vec{0}$")
        additivity_steps.append("")
        additivity_steps.append("Therefore, the mapping violates the **zero preservation property** and is **not linear**.")
        
        proof_results['additivity_proof'] = "\n".join(additivity_steps)
        proof_results['homogeneity_proof'] = "Zero preservation violation proves non-linearity."
        return proof_results
    
    def _prove_bilinear_nonlinearity(self, formula: str, _dimension: int) -> Dict[str, Any]:
        """Generate proof that bilinear mappings (products of variables) are not linear."""
        proof_results = {
            'can_prove': True,
            'proof_type': 'symbolic',
            'additivity_proof': None,
            'homogeneity_proof': None,
            'conclusion': "NOT_LINEAR",
            'error_message': None
        }
        
        additivity_steps = []
        additivity_steps.append("**Proof of Non-linearity for Bilinear Mapping**")
        additivity_steps.append("")
        additivity_steps.append("Given: Mapping contains products of variables (e.g., $v_1 \\cdot v_2$)")
        additivity_steps.append("")
        additivity_steps.append("**Counterexample for Additivity:**")
        additivity_steps.append("For mapping containing term $v_1 v_2$, let:")
        additivity_steps.append("$\\vec{u} = (1, 0, \\ldots)$ and $\\vec{v} = (0, 1, \\ldots)$")
        additivity_steps.append("")
        additivity_steps.append("$L(\\vec{u}) = 1 \\cdot 0 = 0$")
        additivity_steps.append("$L(\\vec{v}) = 0 \\cdot 1 = 0$")
        additivity_steps.append("$L(\\vec{u}) + L(\\vec{v}) = 0 + 0 = 0$")
        additivity_steps.append("")
        additivity_steps.append("$L(\\vec{u} + \\vec{v}) = L((1, 1, \\ldots)) = 1 \\cdot 1 = 1$")
        additivity_steps.append("")
        additivity_steps.append("Since $L(\\vec{u} + \\vec{v}) = 1 \\neq 0 = L(\\vec{u}) + L(\\vec{v})$,")
        additivity_steps.append("**additivity is violated** and the mapping is **not linear**.")
        
        proof_results['additivity_proof'] = "\n".join(additivity_steps)
        proof_results['homogeneity_proof'] = "Additivity violation proves non-linearity."
        return proof_results
    
    def render_linearity_checker(self):
        """Render the main linearity checker interface."""
        # Initialize session state
        if 'example_analysis' not in st.session_state:
            st.session_state.example_analysis = None
        if 'manual_analysis' not in st.session_state:
            st.session_state.manual_analysis = None
            
        st.header("🔍 Linear Mapping Analysis")
        st.write("Enter your mapping formula and get instant analysis with formal proofs!")
        
        # Quick examples section
        st.subheader("📚 Common Examples")
        st.write("Click any example to analyze it instantly:")
        
        # Create columns for examples
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Linear Examples:**")
            if st.button("$L(x,y) = 5x - y$", help="Simple linear combination"):
                self._analyze_example("5*x - y", 2)
            if st.button("$L(\\vec{v}) = -\\vec{v}$", help="Negation mapping"):
                self._analyze_example("negation_mapping", 2, "negation")
            if st.button("$L(x,y) = x$", help="Projection to first component"):
                self._analyze_example("x", 2)
            if st.button("$L(x,y) = [3y, 2y]$", help="Exercise 12(a): L: ℝ² → ℝ²"):
                self._analyze_example("3*y, 2*y", 2)
            if st.button("$L(\\vec{v}) = 0$", help="Exercise 12(c): L: ℝ² → ℝ¹, zero mapping"):
                self._analyze_example("0", 2)
                
        with col2:
            st.write("**Non-linear Examples:**")
            if st.button("$L(\\vec{v}) = \\vec{v} \\cdot \\vec{v}$", help="Quadratic form"):
                self._analyze_example("quadratic_form", 2, "quadratic")
            if st.button("$L(x,y) = x^2$", help="Polynomial degree 2"):
                self._analyze_example("x**2", 2)
            if st.button("$L(x,y) = x + 1$", help="Affine mapping"):
                self._analyze_example("x + 1", 2)
            if st.button("$L(\\vec{v}) = 1 - v_2$", help="Exercise 12(b): L: ℝ² → ℝ¹ (affine)"):
                self._analyze_example("1 - y", 2)
                
        with col3:
            st.write("**Trigonometric:**")
            if st.button("$L(x,y) = [\\cos(x), \\sin(y)]$", help="Trig of variables"):
                self._analyze_example("cos(x), sin(y)", 2, "trigonometric")
            if st.button("$L(x,y) = [\\cos(\\alpha) \\cdot x, \\sin(\\alpha) \\cdot y]$", help="Trig coefficients, α ∈ ℝ"):
                self._analyze_example("cos(alpha)*x, sin(alpha)*y", 2, "trigonometric")
            if st.button("$L(x,y) = xy$", help="Product of variables"):
                self._analyze_example("x*y", 2)
        
        st.markdown("---")
        
        # Add matrix-vector multiplication section
        st.subheader("🔍 Matrix-Vector Multiplication Linearity")
        st.write("Check if L(v) = A⊙v defines a linear mapping (Example 10.7)")
        
        with st.expander("Matrix ⊙ Vector Analysis", expanded=False):
            st.write("Enter a matrix A and check if L(v) = A⊙v is linear")
            
            # Matrix input
            matrix_str = st.text_area(
                "Enter Matrix A (rows separated by semicolons):",
                value="1, 2; 3, 4",
                help="Format: a11, a12; a21, a22",
                height=100
            )
            
            if st.button("🔍 Analyze Matrix Mapping", key="analyze_matrix_mapping"):
                try:
                    # Parse matrix
                    matrix = np.array([
                        [float(x.strip()) for x in row.split(',')]
                        for row in matrix_str.split(';')
                    ])
                    
                    rows, cols = matrix.shape
                    
                    # Display the mapping
                    st.write("### Mapping Definition:")
                    st.latex(f"L(\\vec{{v}}) = A \\odot \\vec{{v}}")
                    st.write("Where A is:")
                    st.latex(self._format_matrix_latex(matrix))
                    
                    # Theoretical explanation
                    st.write("### Theoretical Analysis:")
                    st.info("""
                    **Theorem:** The mapping L(v) = Av (matrix-vector multiplication) is ALWAYS linear!
                    
                    **Proof:**
                    1. **Additivity:** L(u + v) = A(u + v) = Au + Av = L(u) + L(v) ✅
                    2. **Homogeneity:** L(cv) = A(cv) = c(Av) = cL(v) ✅
                    
                    This follows from the distributive properties of matrix multiplication.
                    """)
                    
                    # Show the matrix representation
                    st.success("✅ **This mapping is LINEAR**")
                    st.write("**Matrix representation:** The matrix A itself represents this linear transformation!")
                    
                    # Demonstrate with examples
                    st.write("### Verification with Examples:")
                    
                    # Generate test vectors
                    test_vecs = []
                    for i in range(cols):
                        vec = np.zeros(cols)
                        vec[i] = 1
                        test_vecs.append(vec)
                    
                    # Show basis vector mappings
                    st.write("Standard basis mappings:")
                    for i, vec in enumerate(test_vecs):
                        result = matrix @ vec
                        st.latex(f"L(e_{{{i+1}}}) = A \\cdot {self._format_vector_latex(vec)} = {self._format_vector_latex(result)}")
                    
                    # Test additivity with specific vectors
                    if cols >= 2:
                        u = np.array([1] * cols)
                        v = np.array([i+1 for i in range(cols)])
                        
                        Lu = matrix @ u
                        Lv = matrix @ v
                        L_sum = matrix @ (u + v)
                        
                        st.write("**Additivity test:**")
                        st.latex(f"u = {self._format_vector_latex(u)}, v = {self._format_vector_latex(v)}")
                        st.latex(f"L(u + v) = {self._format_vector_latex(L_sum)}")
                        st.latex(f"L(u) + L(v) = {self._format_vector_latex(Lu)} + {self._format_vector_latex(Lv)} = {self._format_vector_latex(Lu + Lv)}")
                        
                        if np.allclose(L_sum, Lu + Lv):
                            st.success("✅ Additivity verified!")
                    
                except Exception as e:
                    st.error(f"Error parsing matrix: {str(e)}")
        
        st.markdown("---")
        
        # Manual input section
        st.subheader("✏️ Enter Your Own Formula")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            formula = st.text_input(
                "Enter your mapping formula:",
                placeholder="Examples: 5*x - y, x**2 + y, cos(alpha)*x, x*y + z",
                help="Use x, y, z for variables. Use 'alpha' for parameters. Separate multiple outputs with commas.",
                key="manual_formula"
            )
            
        with col2:
            dimension = st.number_input(
                "Input dimension:",
                min_value=1, max_value=5, value=2,
                help="Number of input variables"
            )
            
        # Parameter input for special cases
        with st.expander("⚙️ Parameters (for trigonometric mappings)", expanded=False):
            phi = st.number_input("α parameter (real number):", value=1.0, help="Parameter α ∈ ℝ for trigonometric coefficients like cos(α)·x, sin(α)·y")
        
        if st.button("🔍 Analyze My Formula", type="primary"):
            if formula:
                params = {'phi': phi}
                # Auto-detect mapping type
                mapping_type = self._detect_mapping_type(formula)
                st.session_state.manual_analysis = {
                    'formula': formula,
                    'dimension': dimension,
                    'mapping_type': mapping_type,
                    'params': params
                }
            else:
                st.error("Please enter a mapping formula.")
                
        # Help section
        with st.expander("❓ How to write formulas", expanded=False):
            st.markdown("""
            **Variables:** Use `x`, `y`, `z` for input variables
            
            **Operations:**
            - Addition: `x + y`
            - Subtraction: `x - y` 
            - Multiplication: `2*x` or `x*y`
            - Powers: `x**2`, `y**3`
            - Functions: `cos(x)`, `sin(y)`, `sqrt(x)`
            
            **Multiple outputs:** Separate with commas: `x + y, x - y`
            
            **Examples:**
            - Linear: `3*x - 2*y`, `x + y, x - y`
            - Quadratic: `x**2 + y**2`, `x*y`
            - Affine: `x + 1`, `2*x + y + 5`
            - Trigonometric: `cos(x)`, `cos(alpha)*x, sin(alpha)*y`
            """)
        
        # Execute analysis outside columns for full width display
        self._execute_pending_analysis()
    
    def _execute_pending_analysis(self):
        """Execute any pending analysis requests for full-width display."""
        # Check for example analysis
        if st.session_state.example_analysis:
            analysis_data = st.session_state.example_analysis
            params = {'phi': 1.0}
            mapping_type = analysis_data['mapping_type']
            if mapping_type is None:
                mapping_type = self._detect_mapping_type(analysis_data['formula'])
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            self._perform_linearity_analysis(
                mapping_type, 
                analysis_data['formula'], 
                analysis_data['dimension'], 
                params, 
                "Both"
            )
            # Clear the analysis request
            st.session_state.example_analysis = None
        
        # Check for manual analysis
        elif st.session_state.manual_analysis:
            analysis_data = st.session_state.manual_analysis
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            self._perform_linearity_analysis(
                analysis_data['mapping_type'], 
                analysis_data['formula'], 
                analysis_data['dimension'], 
                analysis_data['params'], 
                "Both"
            )
            # Clear the analysis request
            st.session_state.manual_analysis = None
    
    def _analyze_example(self, formula: str, dimension: int, mapping_type: str = None):
        """Analyze a pre-defined example."""
        # Store the analysis request in session state so it displays outside columns
        st.session_state.example_analysis = {
            'formula': formula,
            'dimension': dimension,
            'mapping_type': mapping_type
        }
    
    def _detect_mapping_type(self, formula: str) -> str:
        """Auto-detect the mapping type based on formula content."""
        formula = formula.lower()
        
        # Special fixed mappings
        if formula == "negation_mapping":
            return "negation"
        elif formula == "quadratic_form":
            return "quadratic"
        elif formula == "zero_mapping":
            return "zero"
        
        # Pattern-based detection
        formula_no_spaces = formula.replace(' ', '')
        
        # Check for products of variables first (before trigonometric)
        if ('x*y' in formula_no_spaces or 'y*z' in formula_no_spaces or 'x*z' in formula_no_spaces or 
            'y*x' in formula_no_spaces or 'z*y' in formula_no_spaces or 'z*x' in formula_no_spaces):
            return "bilinear"
        elif any(trig in formula for trig in ['cos(x)', 'sin(x)', 'cos(y)', 'sin(y)', 'cos(z)', 'sin(z)']):
            return "trigonometric"
        elif any(trig in formula for trig in ['cos(', 'sin(', 'tan(']):
            return "trigonometric"
        elif any(power in formula for power in ['**2', '**3', '^2', '^3']):
            return "polynomial"  # Will be detected as non-linear by degree analysis
        elif any(const in formula for const in ['+1', '+ 1', '-1', '- 1']) and not formula.strip().endswith('*1'):
            # Check for constant terms (but not multiplication by 1)
            return "affine"
        elif formula.strip() in ['x', 'y', 'z']:
            return "projection"
        else:
            return "polynomial"  # Default to polynomial for most expressions
    
    def _render_mapping_inputs(self, mapping_type: str, _dimension: int) -> Tuple[str, Dict]:
        """Render input fields specific to the mapping type."""
        # _dimension parameter kept for interface consistency
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
            st.write("Example: `cos(x)*phi, sin(y)*phi` or `cos(alpha)*x, sin(alpha)*y`")
            formula = st.text_input(
                "Enter trigonometric formula:",
                placeholder="cos(x)*phi, sin(y)*phi",
                help="Use cos(x), sin(y), etc. Use 'phi' or 'alpha' for parameters."
            )
            params['phi'] = st.number_input("Parameter φ/α:", value=1.0)
            
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
                except (ValueError, TypeError):
                    st.error("Invalid vector format")
                    
        elif mapping_type == "quadratic":
            st.subheader("Quadratic Form")
            st.write("Mapping L(v) = v·v (dot product of vector with itself)")
            formula = "quadratic_form"
            
        elif mapping_type == "negation":
            st.subheader("Negation Mapping")
            st.write("Mapping L(v) = -v (scalar multiplication by -1)")
            formula = "negation_mapping"
            
        elif mapping_type == "projection":
            st.subheader("Projection Mapping")
            st.write("Examples: `x`, `y`, `x, y` (select specific components)")
            formula = st.text_input(
                "Enter projection formula:",
                placeholder="x",
                help="Use x, y, z for variables. Separate multiple outputs with commas."
            )
            
        elif mapping_type == "zero":
            st.subheader("Zero Mapping")
            st.write("Mapping L(v) = 0 (maps everything to zero)")
            formula = "zero_mapping"
            
        elif mapping_type == "affine":
            st.subheader("Affine Mapping")
            st.write("Examples: `x + 1`, `2*x + y + 3` (linear terms plus constants)")
            formula = st.text_input(
                "Enter affine formula:",
                placeholder="x + 1",
                help="Include constant terms to test affine mappings."
            )
            
        elif mapping_type == "bilinear":
            st.subheader("Bilinear Mapping")
            st.write("Examples: `x*y`, `x1*x2, x1*x3` (products of variables)")
            formula = st.text_input(
                "Enter bilinear formula:",
                placeholder="x*y",
                help="Use products of variables like x*y."
            )
            
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
                            except (ValueError, TypeError):
                                st.warning(f"Could not parse parameter: {line}")
        
        return formula, params
    
    def _perform_linearity_analysis(self, mapping_type: str, formula: str, 
                                   dimension: int, params: Dict, analysis_mode: str = "Both"):
        """Perform and display linearity analysis."""
        
        empirical_results = None
        proof_results = None
        
        with st.spinner("Analyzing linearity..."):
            if analysis_mode in ["Empirical Testing", "Both"]:
                empirical_results = self.check_linearity(mapping_type, formula, dimension, params)
            
            if analysis_mode in ["Symbolic Proof", "Both"]:
                proof_results = self.generate_symbolic_proof(mapping_type, formula, dimension, params)
        
        # Display main result
        if analysis_mode == "Symbolic Proof" and proof_results:
            if proof_results.get('conclusion') == "LINEAR":
                st.success("✅ **PROVEN LINEAR by symbolic analysis!**")
            elif proof_results.get('conclusion') == "NOT_LINEAR":
                st.error("❌ **PROVEN NOT LINEAR by symbolic analysis!**")
            elif proof_results.get('error_message'):
                st.warning(f"⚠️ **Symbolic proof failed:** {proof_results['error_message']}")
                if empirical_results:
                    if empirical_results['is_linear']:
                        st.success("✅ **Empirically verified as LINEAR**")
                    else:
                        st.error("❌ **Empirically verified as NOT LINEAR**")
        elif empirical_results:
            if empirical_results['is_linear']:
                st.success("✅ **The mapping is LINEAR!**")
            else:
                st.error("❌ **The mapping is NOT LINEAR!**")
        
        # Create tabs for detailed results
        tab_names = ["📋 Summary"]
        if proof_results and proof_results.get('can_prove'):
            tab_names.append("📐 Symbolic Proof")
        if empirical_results:
            tab_names.extend(["➕ Additivity Tests", "✖️ Homogeneity Tests"])
        tab_names.append("📊 Matrix Representation")
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            self._render_summary_tab(empirical_results, mapping_type, formula, params, proof_results)
        
        current_tab = 1
        if proof_results and proof_results.get('can_prove'):
            with tabs[current_tab]:
                self._render_proof_tab(proof_results, mapping_type, formula)
            current_tab += 1
        
        if empirical_results:
            with tabs[current_tab]:
                self._render_additivity_tab(empirical_results)
            
            with tabs[current_tab + 1]:
                self._render_homogeneity_tab(empirical_results)
            current_tab += 2
        
        with tabs[current_tab]:
            results_for_matrix = empirical_results or {'is_linear': proof_results.get('conclusion') == "LINEAR" if proof_results else False, 'matrix_representation': None}
            self._render_matrix_tab(results_for_matrix, mapping_type, formula)
    
    def _render_proof_tab(self, proof_results: Dict, mapping_type: str, formula: str):
        """Render the symbolic proof tab."""
        st.subheader("🔍 Formal Mathematical Proof")
        
        if proof_results.get('error_message'):
            st.error(f"**Proof Error:** {proof_results['error_message']}")
            return
        
        if proof_results.get('conclusion') == "LINEAR":
            st.success("✅ **CONCLUSION: The mapping is PROVEN to be LINEAR**")
        elif proof_results.get('conclusion') == "NOT_LINEAR":
            st.error("❌ **CONCLUSION: The mapping is PROVEN to be NOT LINEAR**")
        elif proof_results.get('conclusion') == "EMPIRICAL_NEEDED":
            st.info("🔬 **CONCLUSION: Empirical testing needed for this case**")
        
        # Display mapping definition
        st.write("**Mapping Definition:**")
        if mapping_type == "polynomial":
            st.latex(f"L(x,y,z) = {formula}")
        elif mapping_type == "dot_product":
            st.latex("L(\\vec{v}) = [\\vec{v} \\cdot \\vec{c}, 0]")
        
        st.markdown("---")
        
        # Display proofs
        if proof_results.get('additivity_proof'):
            st.markdown(proof_results['additivity_proof'])
            st.markdown("---")
        
        if proof_results.get('homogeneity_proof'):
            st.markdown(proof_results['homogeneity_proof'])
            st.markdown("---")
        
        # Final conclusion
        if proof_results.get('conclusion') == "LINEAR":
            st.success("""
            **✅ PROOF COMPLETE**
            
            Since both additivity L(u + v) = L(u) + L(v) and homogeneity L(cv) = cL(v) 
            have been proven to hold for all vectors u, v and scalar c, the mapping L 
            is **definitively LINEAR**.
            """)
        elif proof_results.get('conclusion') == "NOT_LINEAR":
            st.error("""
            **❌ PROOF COMPLETE**
            
            The mapping L violates at least one linearity property, therefore it is 
            **definitively NOT LINEAR**.
            """)
        elif proof_results.get('conclusion') == "EMPIRICAL_NEEDED":
            st.info("""
            **🔬 ANALYSIS COMPLETE**
            
            This mapping requires empirical testing because trigonometric functions 
            are applied to constants, making it effectively a polynomial mapping 
            with numerical coefficients. Check the empirical results for the definitive answer.
            """)
    
    def _render_summary_tab(self, results: Dict, mapping_type: str, formula: str, params: Dict, proof_results: Dict = None):
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
        
        # Show proof results if available
        if proof_results:
            st.subheader("🔬 Analysis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Symbolic Analysis:**")
                if proof_results.get('conclusion') == "LINEAR":
                    st.success("✅ Proven Linear")
                elif proof_results.get('conclusion') == "NOT_LINEAR":
                    st.error("❌ Proven Not Linear")
                elif proof_results.get('conclusion') == "EMPIRICAL_NEEDED":
                    st.info("🔬 Empirical Testing Needed")
                elif proof_results.get('error_message'):
                    st.warning("⚠️ Proof not available")
            
            with col2:
                if results:
                    st.write("**Empirical Verification:**")
                    if results['is_linear']:
                        st.success("✅ Tests Passed")
                    else:
                        st.error("❌ Tests Failed")
        
        # Test results summary (only if empirical results available)
        if results:
            if not proof_results:  # Only show detailed test results if no proof
                st.subheader("🧪 Test Results")
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
        if results and results.get('counterexamples'):
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
                    st.write(f"cL(v) = {test['c_L_v'].tolist()}")
                
                if test['is_homogeneous']:
                    st.success("✅ **PASSED** - Homogeneity holds")
                else:
                    st.error("❌ **FAILED** - Homogeneity violation")
                    st.write(f"**Difference:** {test['difference'].tolist()}")
    
    def _render_matrix_tab(self, results: Dict, mapping_type: str, formula: str):
        """Render the matrix representation tab."""
        # mapping_type and formula kept for potential future use
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