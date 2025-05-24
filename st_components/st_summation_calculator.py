#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Series and Summation Calculator
Handles various types of series calculations and pattern recognition
"""

import streamlit as st
import sympy as sym
import numpy as np
import re
from typing import Optional, Tuple, List

class SummationCalculator:
    def __init__(self):
        self.i = sym.Symbol('i')
        self.a = sym.Symbol('a')
        
    def _format_number_latex(self, num):
        """Helper to format numbers for LaTeX"""
        if isinstance(num, (int, float)):
            if np.isclose(num, round(num)):
                return f"{int(round(num))}"
            else:
                return f"{num:.6f}".rstrip('0').rstrip('.')
        return str(num)
    
    def detect_geometric_series(self, terms: List[float]) -> Optional[Tuple[float, float]]:
        """Detect if a sequence is geometric and return first term and ratio"""
        if len(terms) < 2:
            return None
        
        # Check if consecutive ratios are consistent
        ratios = []
        for i in range(1, len(terms)):
            if abs(terms[i-1]) > 1e-10:  # Avoid division by zero
                ratios.append(terms[i] / terms[i-1])
        
        if len(ratios) < 1:
            return None
            
        # Check if all ratios are approximately equal
        first_ratio = ratios[0]
        if all(abs(r - first_ratio) < 1e-10 for r in ratios):
            return terms[0], first_ratio
        
        return None
    
    def detect_arithmetic_series(self, terms: List[float]) -> Optional[Tuple[float, float]]:
        """Detect if a sequence is arithmetic and return first term and common difference"""
        if len(terms) < 2:
            return None
        
        # Check if consecutive differences are consistent
        diffs = []
        for i in range(1, len(terms)):
            diffs.append(terms[i] - terms[i-1])
        
        if len(diffs) < 1:
            return None
            
        # Check if all differences are approximately equal
        first_diff = diffs[0]
        if all(abs(d - first_diff) < 1e-10 for d in diffs):
            return terms[0], first_diff
        
        return None
    
    def parse_sequence_from_text(self, text: str) -> List[float]:
        """Parse a sequence of numbers from text"""
        # Extract numbers from text using regex
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        try:
            return [float(n) for n in numbers]
        except ValueError:
            return []
    
    def calculate_summation(self, expression: str, start: int, end: int) -> Tuple[Optional[float], str, str]:
        """Calculate summation using SymPy and return result, LaTeX explanation, and text explanation"""
        try:
            # Parse the expression
            expr = sym.sympify(expression.replace('^', '**'))
            
            # Calculate the sum
            result = sym.Sum(expr, (self.i, start, end)).doit()
            
            # Create LaTeX explanation
            latex_expr = sym.latex(expr)
            latex_result = sym.latex(result)
            latex_explanation = f"\\sum_{{i={start}}}^{{{end}}} {latex_expr} = {latex_result}"
            
            # Create text explanation
            text_explanation = f"Sum from i={start} to {end} of ({expression}) = {result}"
            
            return float(result) if result.is_number else result, latex_explanation, text_explanation
        except Exception as e:
            return None, "", f"Error calculating summation: {str(e)}"
    
    def solve_geometric_series_sum(self, first_term: float, ratio: float, n_terms: int) -> Tuple[float, str, str]:
        """Calculate sum of geometric series with LaTeX formatting"""
        if abs(ratio - 1) < 1e-10:
            # Special case: ratio = 1
            result = first_term * n_terms
            latex_explanation = f"S_n = a \\cdot n = {self._format_number_latex(first_term)} \\cdot {n_terms} = {self._format_number_latex(result)}"
            text_explanation = f"S_n = a × n = {first_term} × {n_terms} = {result}"
        else:
            result = first_term * (1 - ratio**n_terms) / (1 - ratio)
            latex_explanation = f"S_n = a \\cdot \\frac{{1-r^n}}{{1-r}} = {self._format_number_latex(first_term)} \\cdot \\frac{{1-({self._format_number_latex(ratio)})^{{{n_terms}}}}}{{1-{self._format_number_latex(ratio)}}} = {self._format_number_latex(result)}"
            text_explanation = f"S_n = a(1-r^n)/(1-r) = {first_term}(1-{ratio}^{n_terms})/(1-{ratio}) = {result}"
        
        return result, latex_explanation, text_explanation
    
    def solve_arithmetic_series_sum(self, first_term: float, diff: float, n_terms: int) -> Tuple[float, str, str]:
        """Calculate sum of arithmetic series with LaTeX formatting"""
        last_term = first_term + (n_terms - 1) * diff
        result = n_terms * (first_term + last_term) / 2
        
        latex_explanation = f"S_n = \\frac{{n(a_1 + a_n)}}{{2}} = \\frac{{{n_terms}({self._format_number_latex(first_term)} + {self._format_number_latex(last_term)})}}{{2}} = {self._format_number_latex(result)}"
        text_explanation = f"S_n = n(a_1 + a_n)/2 = {n_terms}({first_term} + {last_term})/2 = {result}"
        
        return result, latex_explanation, text_explanation
    
    def find_missing_limits_geometric(self, terms_str: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """Find missing start/end indices for geometric series given terms"""
        terms = self.parse_sequence_from_text(terms_str)
        if len(terms) < 2:
            return None, None, None
        
        geo_result = self.detect_geometric_series(terms)
        if not geo_result:
            return None, None, None
        
        first_term, ratio = geo_result
        
        # For geometric series with first term a and ratio r:
        # term_n = a * r^(n-1)
        # If we know the first term in our sequence, find what index it corresponds to
        
        if abs(first_term) > 1e-10:
            # Find the starting index
            start_power = np.log(terms[0] / first_term) / np.log(ratio) if ratio > 0 and ratio != 1 else 0
            start_index = int(round(start_power))
            
            # Find the ending index
            end_power = np.log(terms[-1] / first_term) / np.log(ratio) if ratio > 0 and ratio != 1 else len(terms) - 1
            end_index = int(round(end_power))
            
            return start_index, end_index, ratio
        
        return None, None, None
    
    def render_summation_calculator(self):
        """Render the summation calculator interface"""
        st.markdown('<h2 class="animate-subheader">Series & Summation Calculator</h2>', unsafe_allow_html=True)
        
        st.write("Calculate various types of series, summations, and identify patterns in sequences.")
        
        # Create tabs for different types of calculations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "General Summation", 
            "Geometric Series", 
            "Arithmetic Series", 
            "Pattern Recognition",
            "Exercise Helper"
        ])
        
        with tab1:
            st.subheader("General Summation Calculator")
            st.write("Calculate ∑(i=start to end) expression")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                start_val = st.number_input("Start value (i=)", value=1, step=1, key="sum_start")
            with col2:
                end_val = st.number_input("End value", value=10, step=1, key="sum_end")
            with col3:
                expression = st.text_input("Expression (use 'i' as variable)", value="i**2", key="sum_expr")
            
            if st.button("Calculate Summation", key="calc_sum"):
                if expression:
                    result, latex_explanation, text_explanation = self.calculate_summation(expression, int(start_val), int(end_val))
                    if result is not None:
                        st.success(f"**Result:** {result}")
                        st.write("**Mathematical Expression:**")
                        st.latex(latex_explanation)
                        with st.expander("Show detailed calculation"):
                            st.write(text_explanation)
                    else:
                        st.error(text_explanation)
        
        with tab2:
            st.subheader("Geometric Series Calculator")
            st.write("For series of the form: a + ar + ar² + ar³ + ...")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                geo_first = st.number_input("First term (a)", value=1.0, key="geo_first")
            with col2:
                geo_ratio = st.number_input("Common ratio (r)", value=0.5, key="geo_ratio")
            with col3:
                geo_terms = st.number_input("Number of terms (n)", value=5, min_value=1, step=1, key="geo_terms")
            
            if st.button("Calculate Geometric Series", key="calc_geo"):
                result, latex_explanation, text_explanation = self.solve_geometric_series_sum(geo_first, geo_ratio, int(geo_terms))
                st.success(f"**Sum:** {result}")
                
                st.write("**Mathematical Formula:**")
                st.latex(latex_explanation)
                
                # Show the series terms
                terms = [geo_first * (geo_ratio ** i) for i in range(int(geo_terms))]
                st.write("**Series expansion:**")
                term_strs = [f'{t:.6f}'.rstrip('0').rstrip('.') for t in terms]
                st.write(f"{' + '.join(term_strs)}")
                
                with st.expander("Show calculation details"):
                    st.write(text_explanation)
                    st.write(f"**Individual terms:** {terms}")
        
        with tab3:
            st.subheader("Arithmetic Series Calculator")
            st.write("For series of the form: a + (a+d) + (a+2d) + (a+3d) + ...")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                arith_first = st.number_input("First term (a)", value=1.0, key="arith_first")
            with col2:
                arith_diff = st.number_input("Common difference (d)", value=2.0, key="arith_diff")
            with col3:
                arith_terms = st.number_input("Number of terms (n)", value=5, min_value=1, step=1, key="arith_terms")
            
            if st.button("Calculate Arithmetic Series", key="calc_arith"):
                result, latex_explanation, text_explanation = self.solve_arithmetic_series_sum(arith_first, arith_diff, int(arith_terms))
                st.success(f"**Sum:** {result}")
                
                st.write("**Mathematical Formula:**")
                st.latex(latex_explanation)
                
                # Show the series terms
                terms = [arith_first + i * arith_diff for i in range(int(arith_terms))]
                st.write("**Series expansion:**")
                term_strs = [f'{t:.6f}'.rstrip('0').rstrip('.') for t in terms]
                st.write(f"{' + '.join(term_strs)}")
                
                with st.expander("Show calculation details"):
                    st.write(text_explanation)
                    st.write(f"**Individual terms:** {terms}")
        
        with tab4:
            st.subheader("Pattern Recognition")
            st.write("Enter a sequence of numbers and we'll try to identify the pattern")
            
            sequence_input = st.text_input(
                "Enter sequence (comma-separated or space-separated)",
                value="4, 19, 44, 79, 124, 179, 244",
                key="pattern_seq"
            )
            
            if st.button("Analyze Pattern", key="analyze_pattern"):
                if sequence_input:
                    terms = self.parse_sequence_from_text(sequence_input)
                    if len(terms) >= 2:
                        st.write(f"**Detected sequence:** {terms}")
                        
                        # Check for arithmetic progression
                        arith_result = self.detect_arithmetic_series(terms)
                        if arith_result:
                            first, diff = arith_result
                            st.success("✅ **Arithmetic Series Detected**")
                            st.write(f"**First term (a₁):** {first}")
                            st.write(f"**Common difference (d):** {diff}")
                            st.write("**General term formula:**")
                            st.latex(f"a_n = {self._format_number_latex(first)} + (n-1) \\cdot {self._format_number_latex(diff)}")
                            
                            # Calculate sum formula
                            st.write("**Sum formula:**")
                            st.latex(f"S_n = \\frac{{n \\cdot (2a_1 + (n-1)d)}}{{2}} = \\frac{{n \\cdot (2 \\cdot {self._format_number_latex(first)} + (n-1) \\cdot {self._format_number_latex(diff)})}}{{2}}")
                        
                        # Check for geometric progression
                        geo_result = self.detect_geometric_series(terms)
                        if geo_result:
                            first, ratio = geo_result
                            st.success("✅ **Geometric Series Detected**")
                            st.write(f"**First term (a):** {first}")
                            st.write(f"**Common ratio (r):** {ratio}")
                            st.write("**General term formula:**")
                            st.latex(f"a_n = {self._format_number_latex(first)} \\cdot ({self._format_number_latex(ratio)})^{{n-1}}")
                            
                            # Calculate sum formula
                            st.write("**Sum formula:**")
                            if abs(ratio - 1) < 1e-10:
                                st.latex(f"S_n = a \\cdot n = {self._format_number_latex(first)} \\cdot n")
                            else:
                                st.latex(f"S_n = a \\cdot \\frac{{1-r^n}}{{1-r}} = {self._format_number_latex(first)} \\cdot \\frac{{1-({self._format_number_latex(ratio)})^n}}{{1-{self._format_number_latex(ratio)}}}")
                        
                        if not arith_result and not geo_result:
                            st.warning("🔍 **No simple arithmetic or geometric pattern detected.**")
                            st.write("**Let's analyze differences between consecutive terms:**")
                            
                            diffs = [terms[i+1] - terms[i] for i in range(len(terms)-1)]
                            st.write(f"**First differences:** {diffs}")
                            
                            if len(diffs) > 1:
                                second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
                                st.write(f"**Second differences:** {second_diffs}")
                                
                                # Check if second differences are constant (quadratic sequence)
                                if len(set(second_diffs)) == 1:
                                    st.success("✅ **Quadratic Sequence Detected!**")
                                    st.write("**This is a quadratic sequence** since the second differences are constant.")
                                    st.write("**General form:**")
                                    st.latex("a_n = An^2 + Bn + C")
                                    st.write(f"**Constant second difference:** {second_diffs[0]}")
                                    st.write("**Note:** The coefficient A = (constant second difference)/2")
                                    A = second_diffs[0]/2
                                    st.latex(f"A = \\frac{{{second_diffs[0]}}}{{2}} = {A}")
                                    
                                    # Calculate the complete quadratic formula
                                    # Using first few terms to solve for B and C
                                    # a₁ = A(1)² + B(1) + C = A + B + C
                                    # a₂ = A(2)² + B(2) + C = 4A + 2B + C
                                    # a₃ = A(3)² + B(3) + C = 9A + 3B + C
                                    
                                    if len(terms) >= 3:
                                        a1, a2, a3 = terms[0], terms[1], terms[2]
                                        # From the differences: a₂ - a₁ = 3A + B, a₃ - a₂ = 5A + B
                                        # So: (a₃ - a₂) - (a₂ - a₁) = 2A = second difference
                                        # We know A, so: B = (a₂ - a₁) - 3A
                                        B = (a2 - a1) - 3*A
                                        # And: C = a₁ - A - B
                                        C = a1 - A - B
                                        
                                        st.write("**Complete quadratic formula:**")
                                        st.latex(f"a_n = {self._format_number_latex(A)}n^2 + {self._format_number_latex(B)}n + {self._format_number_latex(C)}")
                                        
                                        # Verify with first few terms
                                        st.write("**Verification:**")
                                        for i in range(min(3, len(terms))):
                                            n = i + 1
                                            calculated = A * n**2 + B * n + C
                                            st.write(f"a₍{n}₎ = {A}({n})² + {B}({n}) + {C} = {calculated} ✓")
                                        
                                        # Predict next terms
                                        st.write("**Next terms in the sequence:**")
                                        next_terms = []
                                        for n in range(len(terms) + 1, len(terms) + 4):
                                            next_term = A * n**2 + B * n + C
                                            next_terms.append(next_term)
                                            st.write(f"a₍{n}₎ = {A}({n})² + {B}({n}) + {C} = {next_term}")
                                        
                                        current_sequence = ', '.join([str(int(t)) if t.is_integer() else str(t) for t in terms])
                                        next_sequence = ', '.join([str(int(t)) if t.is_integer() else str(t) for t in next_terms])
                                        st.info(f"**Extended sequence:** {current_sequence}, {next_sequence}, ...")
                                
                            if len(second_diffs) > 1:
                                third_diffs = [second_diffs[i+1] - second_diffs[i] for i in range(len(second_diffs)-1)]
                                st.write(f"**Third differences:** {third_diffs}")
                                
                                if len(set(third_diffs)) == 1 and third_diffs[0] != 0:
                                    st.success("✅ **Cubic Sequence Detected!**")
                                    st.write("**This is a cubic sequence** since the third differences are constant.")
                                    st.write("**General form:**")
                                    st.latex("a_n = An^3 + Bn^2 + Cn + D")
                    else:
                        st.error("Please enter at least 2 numbers")
        
        with tab5:
            st.subheader("Exercise Helper")
            st.write("Tools to help solve specific summation exercises")
            
            exercise_type = st.selectbox(
                "Select exercise type:",
                ["Geometric series with fractions", "Sum with quadratic terms", "Find missing limits", "Pattern in sequence", "Your Specific Exercise"]
            )
            
            if exercise_type == "Geometric series with fractions":
                st.write("**Example: ∑(1/5)^i = 1 + 1/5 + 1/25 + ... + 1/625**")
                
                fraction_base = st.number_input("Fraction denominator (for 1/n series)", value=5, min_value=2, step=1)
                
                if st.button("Analyze Fraction Series"):
                    # Calculate series 1 + 1/n + 1/n² + ... until we reach a small term
                    terms = []
                    term = 1.0
                    i = 0
                    while term >= 1e-6 and i < 20:  # Stop when term becomes very small
                        terms.append(term)
                        i += 1
                        term = 1.0 / (fraction_base ** i)
                    
                    st.write(f"**Series terms:** {' + '.join([f'1/{fraction_base**i}' if i > 0 else '1' for i in range(len(terms))])}")
                    st.write(f"**Decimal values:** {' + '.join([f'{t:.6f}' for t in terms])}")
                    
                    # This is a geometric series with a=1, r=1/fraction_base
                    a = 1.0
                    r = 1.0 / fraction_base
                    n = len(terms)
                    
                    sum_result = a * (1 - r**n) / (1 - r)
                    st.success(f"**Sum of {n} terms:** {sum_result}")
                    st.write(f"**Formula:** S_n = 1 × (1-(1/{fraction_base})^{n})/(1-1/{fraction_base}) = {sum_result}")
            
            elif exercise_type == "Sum with quadratic terms":
                st.write("**Example: ∑(i+5)² - 350**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    quad_start = st.number_input("Start index", value=1, step=1)
                with col2:
                    quad_end = st.number_input("End index", value=24, step=1)
                with col3:
                    quad_offset = st.number_input("Offset (for i+k)", value=5, step=1)
                
                subtract_val = st.number_input("Value to subtract", value=350.0)
                
                if st.button("Calculate Quadratic Sum"):
                    # Calculate ∑(i+offset)²
                    expr_str = f"(i+{quad_offset})**2"
                    result, latex_explanation, text_explanation = self.calculate_summation(expr_str, int(quad_start), int(quad_end))
                    
                    if result is not None:
                        final_result = result - subtract_val
                        st.success(f"**∑(i+{quad_offset})² = {result}**")
                        st.success(f"**Final result after subtracting {subtract_val}: {final_result}**")
                        
                        st.write("**Mathematical Expression:**")
                        st.latex(latex_explanation)
                        
                        st.write("**Final Calculation:**")
                        st.latex(f"\\text{{Result}} = {result} - {subtract_val} = {final_result}")
                        
                        with st.expander("Show detailed steps"):
                            st.write(text_explanation)
            
            elif exercise_type == "Your Specific Exercise":
                st.write("**Solve the Linear Algebra Test Exercise:**")
                st.markdown("""
                **Given Exercise:**
                
                (a) ∑(1/5)ⁱ = 1 + 1/5 + ... + 1/625 (find missing limits)
                
                (b) ∑₂₄ᵢ₌₁(i+5)² - 350
                
                (c) ∑₂₆ᵢ₌₁₅(i+5)³ = ∑ⱼ₌₀(?)(?) (transform)
                
                (d) ∑₁₂ᵢ₌₁(a+i)²
                
                (e) 4+19+44+79+124+179+244+...+49999 (find pattern)
                """)
                
                problem = st.selectbox("Select problem to solve:", ["(a) Geometric series limits", "(b) Quadratic sum", "(c) Index transformation", "(d) Symbolic sum", "(e) Pattern sequence"])
                
                if problem == "(a) Geometric series limits":
                    st.write("**Problem:** Find ? values for ∑?ᵢ₌?(1/5)ⁱ = 1 + 1/5 + 1/25 + ... + 1/625")
                    
                    if st.button("Solve Problem (a)"):
                        # Identify the pattern: 1, 1/5, 1/25, 1/125, 1/625
                        # This is (1/5)⁰, (1/5)¹, (1/5)², (1/5)³, (1/5)⁴
                        terms = [1, 1/5, 1/25, 1/125, 1/625]
                        powers = [0, 1, 2, 3, 4]
                        
                        st.success("**Solution:**")
                        st.write("**Given terms:** 1 + 1/5 + 1/25 + 1/125 + 1/625")
                        st.write("**Pattern identification:**")
                        for i, (term, power) in enumerate(zip(terms, powers)):
                            st.write(f"Term {i+1}: {term} = (1/5)^{power}")
                        
                        st.write("**Therefore:** The series is ∑ᵢ₌₀⁴(1/5)ⁱ")
                        st.latex("\\sum_{i=0}^{4} \\left(\\frac{1}{5}\\right)^i")
                        
                        # Calculate the sum
                        a, r, n = 1, 1/5, 5
                        result = a * (1 - r**n) / (1 - r)
                        st.write("**Sum calculation:**")
                        st.latex(f"S = \\frac{{1 - (1/5)^5}}{{1 - 1/5}} = \\frac{{1 - 1/3125}}{{4/5}} = \\frac{{3124/3125}}{{4/5}} = {result}")
                        
                elif problem == "(b) Quadratic sum":
                    st.write("**Problem:** Calculate ∑₂₄ᵢ₌₁(i+5)² - 350")
                    
                    if st.button("Solve Problem (b)"):
                        result, latex_explanation, text_explanation = self.calculate_summation("(i+5)**2", 1, 24)
                        final_result = result - 350
                        
                        st.success("**Solution:**")
                        st.latex("\\sum_{i=1}^{24} (i+5)^2 - 350")
                        st.latex(latex_explanation)
                        st.latex(f"\\text{{Final result}} = {result} - 350 = {final_result}")
                        
                elif problem == "(e) Pattern sequence":
                    st.write("**Problem:** Find pattern in 4+19+44+79+124+179+244+...+49999")
                    
                    if st.button("Solve Problem (e)"):
                        sequence = [4, 19, 44, 79, 124, 179, 244]
                        st.write("**Given sequence:** 4, 19, 44, 79, 124, 179, 244, ..., 49999")
                        
                        # Calculate differences
                        diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
                        second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
                        
                        st.write(f"**First differences:** {diffs}")
                        st.write(f"**Second differences:** {second_diffs}")
                        st.success("**Pattern identified:** Quadratic sequence with constant second difference = 10")
                        
                        # Find the quadratic formula
                        A = 10/2  # A = 5
                        a1, a2 = sequence[0], sequence[1]
                        B = (a2 - a1) - 3*A  # B = 15 - 15 = 0
                        C = a1 - A - B  # C = 4 - 5 - 0 = -1
                        
                        st.write("**Quadratic formula:**")
                        st.latex(f"a_n = {A}n^2 + {B}n + {C} = 5n^2 - 1")
                        
                        # Find which term equals 49999
                        st.write("**Find n where aₙ = 49999:**")
                        st.latex("5n^2 - 1 = 49999")
                        st.latex("5n^2 = 50000")
                        st.latex("n^2 = 10000")
                        st.latex("n = 100")
                        
                        # Calculate the sum
                        st.write("**Sum from a₁ to a₁₀₀:**")
                        result, latex_explanation, text_explanation = self.calculate_summation("5*i**2 - 1", 1, 100)
                        st.latex(latex_explanation)
                        st.success(f"**Final Answer:** {result}")
            
            # Help section
            with st.expander("Help: Common Series Formulas"):
                st.write("""
                **Arithmetic Series:** S_n = n(a₁ + aₙ)/2 = n(2a + (n-1)d)/2
                
                **Geometric Series:** S_n = a(1-rⁿ)/(1-r) for r ≠ 1, S_n = na for r = 1
                
                **Common Summations:**
                - ∑i = n(n+1)/2
                - ∑i² = n(n+1)(2n+1)/6  
                - ∑i³ = [n(n+1)/2]²
                
                **Power Series:**
                - ∑xⁱ = (1-xⁿ⁺¹)/(1-x) for |x| < 1
                - ∑i×xⁱ = x(1-(n+1)xⁿ+nxⁿ⁺¹)/(1-x)²
                """)