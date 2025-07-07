#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Complex Number Operations
Implements operations from Week 20: Complex Numbers (Komplexe Zahlen)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Optional, List, Dict, Union
import cmath
from fractions import Fraction

class ComplexOperations:
    """
    Component for complex number operations and visualizations.
    Handles arithmetic operations, polar/rectangular conversions, 
    and Gaussian plane visualizations.
    """
    
    def __init__(self):
        self.complex_number = None
        
    def parse_complex_input(self, input_str: str) -> Optional[complex]:
        """Parse complex number from various input formats."""
        try:
            # Remove spaces
            input_str = input_str.replace(' ', '')
            
            # Handle special cases first
            if input_str.lower() in ['i', '+i']:
                return complex(0, 1)
            elif input_str.lower() in ['-i']:
                return complex(0, -1)
            
            # Handle complex fractions like (1+2i)/(3+4i)
            if '/' in input_str and ('i' in input_str.lower() or 'j' in input_str.lower()):
                try:
                    # Find the division point, accounting for parentheses
                    paren_level = 0
                    division_index = -1
                    
                    for i, char in enumerate(input_str):
                        if char == '(':
                            paren_level += 1
                        elif char == ')':
                            paren_level -= 1
                        elif char == '/' and paren_level == 0:
                            division_index = i
                            break
                    
                    if division_index > 0:
                        numerator_str = input_str[:division_index]
                        denominator_str = input_str[division_index+1:]
                        
                        # Remove parentheses if they wrap the entire expression
                        if numerator_str.startswith('(') and numerator_str.endswith(')'):
                            numerator_str = numerator_str[1:-1]
                        if denominator_str.startswith('(') and denominator_str.endswith(')'):
                            denominator_str = denominator_str[1:-1]
                        
                        # Recursively parse numerator and denominator
                        numerator = self.parse_complex_input(numerator_str)
                        denominator = self.parse_complex_input(denominator_str)
                        
                        if numerator is not None and denominator is not None:
                            if abs(denominator) > 1e-10:
                                return numerator / denominator
                            else:
                                st.error("Division by zero")
                                return None
                except:
                    pass
            
            # Handle different formats
            if 'j' in input_str.lower():
                # Python format: 3+4j or 3+4J
                return complex(input_str.replace('J', 'j'))
            elif 'i' in input_str.lower():
                # Math format: 3+4i
                # First handle standalone i
                processed = input_str.replace('I', 'i')
                
                # Replace lone 'i' with '1j' (but not when it's part of a number)
                import re
                # Replace 'i' at the end or when preceded by +/- or at start
                processed = re.sub(r'(?<![0-9])i(?![0-9])', '1j', processed)
                processed = re.sub(r'([+-])i(?![0-9])', r'\g<1>1j', processed)
                # Replace other 'i' with 'j'
                processed = processed.replace('i', 'j')
                
                return complex(processed)
            else:
                # Check if it's just a real number
                return complex(float(input_str), 0)
        except:
            # Try parsing as (a,b) format
            try:
                if ',' in input_str:
                    parts = input_str.strip('()[]').split(',')
                    if len(parts) == 2:
                        return complex(float(parts[0]), float(parts[1]))
            except:
                pass
            
            st.error(f"Invalid complex number format: {input_str}")
            return None
    
    def format_complex_latex(self, z: complex) -> str:
        """Format complex number for LaTeX display."""
        if z.imag == 0:
            return f"{z.real:.4g}"
        elif z.real == 0:
            if z.imag == 1:
                return "i"
            elif z.imag == -1:
                return "-i"
            else:
                return f"{z.imag:.4g}i"
        else:
            if z.imag > 0:
                if z.imag == 1:
                    return f"{z.real:.4g} + i"
                else:
                    return f"{z.real:.4g} + {z.imag:.4g}i"
            else:
                if z.imag == -1:
                    return f"{z.real:.4g} - i"
                else:
                    return f"{z.real:.4g} - {abs(z.imag):.4g}i"
    
    def format_complex_fraction(self, z: complex, tolerance: float = 1e-10) -> str:
        """Format complex number as fraction when possible."""
        def to_fraction_str(value: float) -> str:
            """Convert float to fraction string if it's a simple fraction."""
            if abs(value) < tolerance:
                return "0"
            
            # Try to convert to fraction
            try:
                frac = Fraction(value).limit_denominator(1000)
                if abs(float(frac) - value) < tolerance:
                    if frac.denominator == 1:
                        return str(frac.numerator)
                    else:
                        return f"\\frac{{{frac.numerator}}}{{{frac.denominator}}}"
                else:
                    return f"{value:.4g}"
            except:
                return f"{value:.4g}"
        
        real_frac = to_fraction_str(z.real)
        imag_frac = to_fraction_str(z.imag)
        
        if abs(z.imag) < tolerance:
            return real_frac
        elif abs(z.real) < tolerance:
            if imag_frac == "1":
                return "i"
            elif imag_frac == "-1":
                return "-i"
            else:
                return f"{imag_frac}i"
        else:
            if abs(z.imag - 1) < tolerance:
                return f"{real_frac} + i"
            elif abs(z.imag + 1) < tolerance:
                return f"{real_frac} - i"
            elif z.imag > 0:
                return f"{real_frac} + {imag_frac}i"
            else:
                return f"{real_frac} - {imag_frac.replace('-', '')}i"
    
    def complex_to_polar(self, z: complex) -> Tuple[float, float]:
        """Convert complex number to polar form (r, θ)."""
        r = abs(z)
        theta = cmath.phase(z)  # Returns angle in radians
        return r, theta
    
    def polar_to_complex(self, r: float, theta: float) -> complex:
        """Convert polar form to complex number."""
        return cmath.rect(r, theta)
    
    def create_gaussian_plane_plot(self, numbers: List[Tuple[complex, str]], 
                                  title: str = "Gaussian Plane (Complex Plane)",
                                  show_operations: bool = False) -> go.Figure:
        """Create an interactive plot of complex numbers on the Gaussian plane."""
        fig = go.Figure()
        
        # Calculate max value for axis range
        max_val = 1.0  # Default minimum
        for z, _ in numbers:
            max_val = max(max_val, abs(z.real), abs(z.imag))
        
        max_val = max_val * 1.2  # Add 20% padding
        
        # Add coordinate axes with dark theme styling
        fig.add_trace(go.Scatter(
            x=[-max_val, max_val], y=[0, 0],
            mode='lines',
            line=dict(color='white', width=2),
            showlegend=False,
            hoverinfo='skip',
            name='Real axis'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[-max_val, max_val],
            mode='lines',
            line=dict(color='white', width=2),
            showlegend=False,
            hoverinfo='skip',
            name='Imaginary axis'
        ))
        
        # Plot each complex number with vibrant colors for dark theme
        colors = ['#00D9FF', '#FF0080', '#00FF88', '#FFD700', '#FF6B6B', '#4ECDC4']
        for i, (z, label) in enumerate(numbers):
            color = colors[i % len(colors)]
            
            # Add vector from origin to point
            fig.add_trace(go.Scatter(
                x=[0, z.real], y=[0, z.imag],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=[0, 10], symbol='circle', 
                          line=dict(width=2, color='white'))
            ))
            
            # Add annotation with dark theme styling
            fig.add_annotation(
                x=z.real, y=z.imag,
                text=f"{label}<br>{self.format_complex_latex(z)}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                ax=30, ay=-40,
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor=color,
                borderwidth=2,
                font=dict(color="white", size=12)
            )
        
        # Update layout with dark theme
        fig.update_layout(
            title=dict(text=title, font=dict(color="white", size=20)),
            xaxis_title=dict(text="Real axis", font=dict(color="white")),
            yaxis_title=dict(text="Imaginary axis", font=dict(color="white")),
            width=600,
            height=600,
            showlegend=True,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1
            )
        )
        
        # Configure axes with dark theme
        fig.update_xaxes(
            scaleanchor="y",
            scaleratio=1,
            range=[-max_val, max_val],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='white',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.2)',
            color='white'
        )
        fig.update_yaxes(
            range=[-max_val, max_val],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='white',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.2)',
            color='white'
        )
        
        return fig
    
    def complex_addition(self, z1: complex, z2: complex) -> Dict:
        """Perform complex number addition with step-by-step explanation."""
        result = z1 + z2
        
        explanation = {
            'operation': 'addition',
            'step1': f"({z1.real:.4g} + {z1.imag:.4g}i) + ({z2.real:.4g} + {z2.imag:.4g}i)",
            'step2': f"= ({z1.real:.4g} + {z2.real:.4g}) + ({z1.imag:.4g} + {z2.imag:.4g})i",
            'step3': f"= {result.real:.4g} + {result.imag:.4g}i",
            'result': result,
            'geometric': "Addition corresponds to vector addition in the Gaussian plane"
        }
        
        return explanation
    
    def complex_multiplication(self, z1: complex, z2: complex) -> Dict:
        """Perform complex number multiplication with step-by-step explanation."""
        result = z1 * z2
        
        # Using FOIL method
        ac = z1.real * z2.real
        ad = z1.real * z2.imag
        bc = z1.imag * z2.real
        bd = z1.imag * z2.imag
        
        explanation = {
            'operation': 'multiplication',
            'step1': f"({z1.real:.4g} + {z1.imag:.4g}i) × ({z2.real:.4g} + {z2.imag:.4g}i)",
            'step2': f"= {z1.real:.4g}×{z2.real:.4g} + {z1.real:.4g}×{z2.imag:.4g}i + {z1.imag:.4g}i×{z2.real:.4g} + {z1.imag:.4g}i×{z2.imag:.4g}i",
            'step3': f"= {ac:.4g} + {ad:.4g}i + {bc:.4g}i + {bd:.4g}i²",
            'step4': f"= {ac:.4g} + {ad:.4g}i + {bc:.4g}i + {bd:.4g}×(-1)",
            'step5': f"= {ac:.4g} + {ad:.4g}i + {bc:.4g}i - {abs(bd):.4g}",
            'step6': f"= ({ac:.4g} - {abs(bd):.4g}) + ({ad:.4g} + {bc:.4g})i",
            'step7': f"= {result.real:.4g} + {result.imag:.4g}i",
            'result': result,
            'geometric': "Multiplication rotates and scales in the Gaussian plane"
        }
        
        # Add polar form explanation
        r1, theta1 = self.complex_to_polar(z1)
        r2, theta2 = self.complex_to_polar(z2)
        r_result, theta_result = self.complex_to_polar(result)
        
        explanation['polar'] = {
            'z1_polar': f"|z₁| = {r1:.4g}, arg(z₁) = {theta1:.4g} rad",
            'z2_polar': f"|z₂| = {r2:.4g}, arg(z₂) = {theta2:.4g} rad", 
            'multiplication': f"|z₁×z₂| = |z₁|×|z₂| = {r1:.4g}×{r2:.4g} = {r_result:.4g}",
            'angle_addition': f"arg(z₁×z₂) = arg(z₁) + arg(z₂) = {theta1:.4g} + {theta2:.4g} = {theta_result:.4g} rad"
        }
        
        return explanation
    
    def complex_conjugate(self, z: complex) -> complex:
        """Calculate the complex conjugate."""
        return z.conjugate()
    
    def complex_division(self, z1: complex, z2: complex) -> Dict:
        """Perform complex division with explanation."""
        if z2 == 0:
            return {'error': 'Division by zero is undefined'}
        
        result = z1 / z2
        z2_conj = z2.conjugate()
        
        explanation = {
            'operation': 'division',
            'step1': f"({self.format_complex_latex(z1)}) / ({self.format_complex_latex(z2)})",
            'step2': f"Multiply numerator and denominator by conjugate of denominator",
            'step3': f"= ({self.format_complex_latex(z1)}) × ({self.format_complex_latex(z2_conj)}) / (({self.format_complex_latex(z2)}) × ({self.format_complex_latex(z2_conj)}))",
            'numerator': z1 * z2_conj,
            'denominator': (z2 * z2_conj).real,  # This is always real
            'result': result
        }
        
        return explanation
    
    def render_complex_calculator(self):
        """Render the main complex number calculator interface."""
        st.header("🔢 Complex Number Operations")
        st.write("Perform arithmetic operations and visualize complex numbers on the Gaussian plane.")
        
        # Operation selector
        operation = st.selectbox(
            "Select Operation",
            ["Addition", "Subtraction", "Multiplication", "Division", "Conjugate", "Real & Imaginary Parts", 
             "Magnitude & Argument", "Polar Form", "Polar Arithmetic", "Powers & Roots", "Polynomial Equations",
             "Gaussian Plane Visualization"]
        )
        
        if operation == "Addition":
            self._render_addition()
        elif operation == "Subtraction":
            self._render_subtraction()
        elif operation == "Multiplication":
            self._render_multiplication()
        elif operation == "Division":
            self._render_division()
        elif operation == "Conjugate":
            self._render_conjugate()
        elif operation == "Real & Imaginary Parts":
            self._render_real_imaginary_parts()
        elif operation == "Magnitude & Argument":
            self._render_magnitude_argument()
        elif operation == "Polar Form":
            self._render_polar_form()
        elif operation == "Polar Arithmetic":
            self._render_polar_arithmetic()
        elif operation == "Powers & Roots":
            self._render_powers_roots()
        elif operation == "Polynomial Equations":
            self._render_polynomial_equations()
        elif operation == "Gaussian Plane Visualization":
            self._render_gaussian_plane()
    
    def _render_addition(self):
        """Render addition interface."""
        st.subheader("Complex Number Addition")
        
        col1, col2 = st.columns(2)
        with col1:
            z1_input = st.text_input("First complex number z₁:", value="3+4i", 
                                   help="Format: a+bi or a+bj or (a,b)")
        with col2:
            z2_input = st.text_input("Second complex number z₂:", value="2-i",
                                   help="Format: a+bi or a+bj or (a,b)")
        
        if st.button("Calculate Addition"):
            z1 = self.parse_complex_input(z1_input)
            z2 = self.parse_complex_input(z2_input)
            
            if z1 is not None and z2 is not None:
                result = self.complex_addition(z1, z2)
                
                st.write("### Step-by-Step Solution:")
                st.latex(f"z_1 = {self.format_complex_latex(z1)}")
                st.latex(f"z_2 = {self.format_complex_latex(z2)}")
                st.latex("z_1 + z_2 = " + result['step1'].replace('i', 'i'))
                st.latex("= " + result['step2'].replace('i', 'i'))
                st.latex("= " + self.format_complex_latex(result['result']))
                
                st.info(f"💡 {result['geometric']}")
                
                # Visualization
                st.write("### Visualization:")
                fig = self.create_gaussian_plane_plot([
                    (z1, "z₁"),
                    (z2, "z₂"),
                    (result['result'], "z₁ + z₂")
                ], title="Complex Addition on Gaussian Plane")
                st.plotly_chart(fig)
    
    def _render_multiplication(self):
        """Render multiplication interface."""
        st.subheader("Complex Number Multiplication")
        
        col1, col2 = st.columns(2)
        with col1:
            z1_input = st.text_input("First complex number z₁:", value="2+3i")
        with col2:
            z2_input = st.text_input("Second complex number z₂:", value="1-i")
        
        if st.button("Calculate Multiplication"):
            z1 = self.parse_complex_input(z1_input)
            z2 = self.parse_complex_input(z2_input)
            
            if z1 is not None and z2 is not None:
                result = self.complex_multiplication(z1, z2)
                
                st.write("### Step-by-Step Solution (FOIL Method):")
                for key in ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7']:
                    if key in result:
                        st.latex(result[key].replace('×', '\\times'))
                
                st.success(f"Result: {self.format_complex_latex(result['result'])}")
                
                # Polar form explanation
                with st.expander("Polar Form Interpretation"):
                    st.write("In polar form, multiplication is even simpler:")
                    polar = result['polar']
                    for key, value in polar.items():
                        st.write(f"- {value}")
                
                st.info(f"💡 {result['geometric']}")
                
                # Visualization
                st.write("### Visualization:")
                fig = self.create_gaussian_plane_plot([
                    (z1, "z₁"),
                    (z2, "z₂"),
                    (result['result'], "z₁ × z₂")
                ], title="Complex Multiplication on Gaussian Plane")
                st.plotly_chart(fig)
    
    def _render_division(self):
        """Render division interface."""
        st.subheader("Complex Number Division")
        
        col1, col2 = st.columns(2)
        with col1:
            z1_input = st.text_input("Numerator z₁:", value="5+3i")
        with col2:
            z2_input = st.text_input("Denominator z₂:", value="2-i")
        
        if st.button("Calculate Division"):
            z1 = self.parse_complex_input(z1_input)
            z2 = self.parse_complex_input(z2_input)
            
            if z1 is not None and z2 is not None:
                result = self.complex_division(z1, z2)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.write("### Solution:")
                    st.latex(f"\\frac{{{self.format_complex_latex(z1)}}}{{{self.format_complex_latex(z2)}}}")
                    st.write(result['step2'])
                    st.latex(f"= \\frac{{{self.format_complex_latex(result['numerator'])}}}{{{result['denominator']:.4g}}}")
                    
                    # Show both fraction and decimal form
                    fraction_form = self.format_complex_fraction(result['result'])
                    decimal_form = self.format_complex_latex(result['result'])
                    
                    if fraction_form != decimal_form:
                        st.latex(f"= {fraction_form} = {decimal_form}")
                    else:
                        st.latex(f"= {decimal_form}")
                    
                    # Visualization
                    st.write("### Visualization:")
                    fig = self.create_gaussian_plane_plot([
                        (z1, "z₁ (numerator)"),
                        (z2, "z₂ (denominator)"),
                        (result['result'], "z₁ / z₂")
                    ], title="Complex Division on Gaussian Plane")
                    st.plotly_chart(fig)
    
    def _render_conjugate(self):
        """Render conjugate interface."""
        st.subheader("Complex Conjugate")
        
        z_input = st.text_input("Enter complex number z:", value="3+4i")
        
        if st.button("Calculate Conjugate"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                z_conj = self.complex_conjugate(z)
                
                st.write("### Result:")
                st.latex(f"z = {self.format_complex_latex(z)}")
                st.latex(f"\\overline{{z}} = {self.format_complex_latex(z_conj)}")
                
                st.info("💡 The conjugate reflects the complex number across the real axis")
                
                # Properties
                st.write("### Properties:")
                st.latex(f"|z|^2 = z \\cdot \\overline{{z}} = {abs(z)**2:.4g}")
                
                # Visualization
                fig = self.create_gaussian_plane_plot([
                    (z, "z"),
                    (z_conj, "z̄ (conjugate)")
                ], title="Complex Conjugate on Gaussian Plane")
                st.plotly_chart(fig)
    
    def _render_polar_form(self):
        """Render polar form conversion interface."""
        st.subheader("Polar Form Conversion")
        
        conversion_type = st.radio("Select conversion:", ["Rectangular to Polar", "Polar to Rectangular"])
        
        if conversion_type == "Rectangular to Polar":
            z_input = st.text_input("Enter complex number z:", value="1+i")
            
            if st.button("Convert to Polar"):
                z = self.parse_complex_input(z_input)
                
                if z is not None:
                    r, theta = self.complex_to_polar(z)
                    theta_deg = np.degrees(theta)
                    
                    st.write("### Polar Form:")
                    st.latex(f"z = {self.format_complex_latex(z)}")
                    st.latex(f"|z| = r = \\sqrt{{({z.real:.4g})^2 + ({z.imag:.4g})^2}} = {r:.4g}")
                    st.latex(f"\\arg(z) = \\theta = \\arctan\\left(\\frac{{{z.imag:.4g}}}{{{z.real:.4g}}}\\right) = {theta:.4g} \\text{{ rad}} = {theta_deg:.1f}°")
                    st.latex(f"z = {r:.4g}(\\cos({theta:.4g}) + i\\sin({theta:.4g}))")
                    st.latex(f"z = {r:.4g}e^{{i({theta:.4g})}}")
        
        else:  # Polar to Rectangular
            col1, col2 = st.columns(2)
            with col1:
                r = st.number_input("Magnitude r:", value=2.0, min_value=0.0)
            with col2:
                theta_deg = st.number_input("Angle θ (degrees):", value=45.0)
            
            if st.button("Convert to Rectangular"):
                theta = np.radians(theta_deg)
                z = self.polar_to_complex(r, theta)
                
                st.write("### Rectangular Form:")
                st.latex(f"r = {r:.4g}, \\theta = {theta_deg}° = {theta:.4g} \\text{{ rad}}")
                st.latex(f"z = r(\\cos(\\theta) + i\\sin(\\theta))")
                st.latex(f"z = {r:.4g}(\\cos({theta:.4g}) + i\\sin({theta:.4g}))")
                st.latex(f"z = {r:.4g} \\times {np.cos(theta):.4g} + i \\times {r:.4g} \\times {np.sin(theta):.4g}")
                st.latex(f"z = {self.format_complex_latex(z)}")
    
    def _render_gaussian_plane(self):
        """Render Gaussian plane visualization interface."""
        st.subheader("Gaussian Plane Visualization")
        st.write("Visualize multiple complex numbers on the Gaussian plane")
        
        num_points = st.number_input("Number of complex numbers:", min_value=1, max_value=6, value=3)
        
        numbers = []
        for i in range(num_points):
            col1, col2 = st.columns([3, 1])
            with col1:
                z_input = st.text_input(f"Complex number {i+1}:", value=f"{i+1}+{i}i", key=f"gauss_{i}")
            with col2:
                label = st.text_input("Label:", value=f"z_{i+1}", key=f"label_{i}")
            
            z = self.parse_complex_input(z_input)
            if z is not None:
                numbers.append((z, label))
        
        if st.button("Plot on Gaussian Plane"):
            if numbers:
                fig = self.create_gaussian_plane_plot(numbers, "Complex Numbers on Gaussian Plane")
                st.plotly_chart(fig)
                
                # Show table of values
                st.write("### Summary Table:")
                data = []
                for z, label in numbers:
                    r, theta = self.complex_to_polar(z)
                    data.append({
                        "Label": label,
                        "Rectangular": self.format_complex_latex(z),
                        "Real Part": f"{z.real:.4g}",
                        "Imaginary Part": f"{z.imag:.4g}",
                        "Magnitude |z|": f"{r:.4g}",
                        "Angle (rad)": f"{theta:.4g}",
                        "Angle (deg)": f"{np.degrees(theta):.1f}°"
                    })
                df = pd.DataFrame(data)
                st.dataframe(df)
    
    def _render_subtraction(self):
        """Render subtraction interface."""
        st.subheader("Complex Number Subtraction")
        
        col1, col2 = st.columns(2)
        with col1:
            z1_input = st.text_input("First complex number z₁:", value="5+3i", key="sub_z1")
        with col2:
            z2_input = st.text_input("Second complex number z₂:", value="2+i", key="sub_z2")
        
        if st.button("Calculate Subtraction"):
            z1 = self.parse_complex_input(z1_input)
            z2 = self.parse_complex_input(z2_input)
            
            if z1 is not None and z2 is not None:
                result = z1 - z2
                
                st.write("### Step-by-Step Solution:")
                st.latex(f"z_1 = {self.format_complex_latex(z1)}")
                st.latex(f"z_2 = {self.format_complex_latex(z2)}")
                st.latex(f"z_1 - z_2 = ({z1.real:.4g} + {z1.imag:.4g}i) - ({z2.real:.4g} + {z2.imag:.4g}i)")
                st.latex(f"= ({z1.real:.4g} - {z2.real:.4g}) + ({z1.imag:.4g} - {z2.imag:.4g})i")
                st.latex(f"= {self.format_complex_latex(result)}")
                
                st.success(f"Result: {self.format_complex_latex(result)}")
                
                # Visualization
                st.write("### Visualization:")
                fig = self.create_gaussian_plane_plot([
                    (z1, "z₁"),
                    (z2, "z₂"),
                    (result, "z₁ - z₂")
                ], title="Complex Subtraction on Gaussian Plane")
                st.plotly_chart(fig)
    
    def _render_real_imaginary_parts(self):
        """Render real and imaginary parts extraction interface."""
        st.subheader("Real & Imaginary Parts")
        
        z_input = st.text_input("Enter complex number z:", value="3+4i")
        
        if st.button("Extract Parts"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                real_part = z.real
                imag_part = z.imag
                
                st.write("### Results:")
                
                # Show both decimal and fraction forms
                fraction_form = self.format_complex_fraction(z)
                decimal_form = self.format_complex_latex(z)
                
                if fraction_form != decimal_form:
                    st.latex(f"z = {fraction_form} = {decimal_form}")
                else:
                    st.latex(f"z = {decimal_form}")
                
                # Real part
                real_fraction = self.format_complex_fraction(complex(real_part, 0))
                if real_fraction != f"{real_part:.4g}":
                    st.latex(f"\\text{{Re}}(z) = {real_fraction} = {real_part:.4g}")
                else:
                    st.latex(f"\\text{{Re}}(z) = {real_part:.4g}")
                
                # Imaginary part
                imag_fraction = self.format_complex_fraction(complex(0, imag_part)).replace('i', '')
                if imag_fraction != f"{imag_part:.4g}":
                    st.latex(f"\\text{{Im}}(z) = {imag_fraction} = {imag_part:.4g}")
                else:
                    st.latex(f"\\text{{Im}}(z) = {imag_part:.4g}")
                
                # Properties
                st.write("### Properties:")
                if real_part == 0:
                    st.info("This is a purely imaginary number")
                elif imag_part == 0:
                    st.info("This is a real number")
                else:
                    st.info("This is a complex number with both real and imaginary parts")
                
                # Visualization with components highlighted
                fig = self.create_gaussian_plane_plot([
                    (z, f"z = {self.format_complex_latex(z)}")
                ], title="Complex Number Components")
                
                # Add component lines
                fig.add_trace(go.Scatter(
                    x=[0, real_part], y=[0, 0],
                    mode='lines+markers',
                    name=f'Real part = {real_part:.4g}',
                    line=dict(color='orange', width=2, dash='dash'),
                    marker=dict(size=[0, 8])
                ))
                
                fig.add_trace(go.Scatter(
                    x=[real_part, real_part], y=[0, imag_part],
                    mode='lines+markers',
                    name=f'Imaginary part = {imag_part:.4g}',
                    line=dict(color='purple', width=2, dash='dash'),
                    marker=dict(size=[0, 8])
                ))
                
                st.plotly_chart(fig)
    
    def _render_magnitude_argument(self):
        """Render magnitude and argument calculation interface."""
        st.subheader("Magnitude & Argument")
        
        z_input = st.text_input("Enter complex number z:", value="3+4i")
        
        if st.button("Calculate Magnitude & Argument"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                magnitude = abs(z)
                argument = cmath.phase(z)
                argument_deg = np.degrees(argument)
                
                st.write("### Results:")
                st.latex(f"z = {self.format_complex_latex(z)}")
                
                # Magnitude calculation
                st.write("**Magnitude (Modulus):**")
                st.latex(f"|z| = \\sqrt{{({z.real:.4g})^2 + ({z.imag:.4g})^2}}")
                st.latex(f"|z| = \\sqrt{{{z.real**2:.4g} + {z.imag**2:.4g}}} = {magnitude:.4g}")
                
                # Argument calculation
                st.write("**Argument (Phase Angle):**")
                if z.real > 0:
                    st.latex(f"\\arg(z) = \\arctan\\left(\\frac{{{z.imag:.4g}}}{{{z.real:.4g}}}\\right) = {argument:.4g} \\text{{ rad}}")
                elif z.real < 0 and z.imag >= 0:
                    st.latex(f"\\arg(z) = \\arctan\\left(\\frac{{{z.imag:.4g}}}{{{z.real:.4g}}}\\right) + \\pi = {argument:.4g} \\text{{ rad}}")
                elif z.real < 0 and z.imag < 0:
                    st.latex(f"\\arg(z) = \\arctan\\left(\\frac{{{z.imag:.4g}}}{{{z.real:.4g}}}\\right) - \\pi = {argument:.4g} \\text{{ rad}}")
                elif z.real == 0:
                    if z.imag > 0:
                        st.latex("\\arg(z) = \\frac{\\pi}{2}")
                    elif z.imag < 0:
                        st.latex("\\arg(z) = -\\frac{\\pi}{2}")
                    else:
                        st.latex("\\arg(z) \\text{ is undefined (z = 0)}")
                
                st.latex(f"\\arg(z) = {argument:.4g} \\text{{ rad}} = {argument_deg:.1f}°")
                
                # Polar form
                st.write("**Polar Form:**")
                st.latex(f"z = {magnitude:.4g}(\\cos({argument:.4g}) + i\\sin({argument:.4g}))")
                st.latex(f"z = {magnitude:.4g}e^{{i{argument:.4g}}}")
                
                # Properties
                st.write("### Properties:")
                st.write(f"- **Magnitude:** {magnitude:.4g}")
                st.write(f"- **Argument:** {argument:.4g} rad = {argument_deg:.1f}°")
                
                # Determine quadrant
                if z.real > 0 and z.imag >= 0:
                    quadrant = "First quadrant"
                elif z.real <= 0 and z.imag > 0:
                    quadrant = "Second quadrant"
                elif z.real < 0 and z.imag <= 0:
                    quadrant = "Third quadrant"
                elif z.real >= 0 and z.imag < 0:
                    quadrant = "Fourth quadrant"
                else:
                    quadrant = "On axis"
                
                st.write(f"- **Location:** {quadrant}")
                
                # Visualization
                fig = self.create_gaussian_plane_plot([
                    (z, f"z = {self.format_complex_latex(z)}")
                ], title="Magnitude and Argument Visualization")
                
                # Add magnitude circle
                theta_circle = np.linspace(0, 2*np.pi, 100)
                x_circle = magnitude * np.cos(theta_circle)
                y_circle = magnitude * np.sin(theta_circle)
                
                fig.add_trace(go.Scatter(
                    x=x_circle, y=y_circle,
                    mode='lines',
                    name=f'|z| = {magnitude:.4g}',
                    line=dict(color='yellow', width=1, dash='dot'),
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add argument line
                fig.add_trace(go.Scatter(
                    x=[0, magnitude * np.cos(argument)], 
                    y=[0, magnitude * np.sin(argument)],
                    mode='lines',
                    name=f'arg(z) = {argument_deg:.1f}°',
                    line=dict(color='red', width=3),
                    showlegend=True
                ))
                
                st.plotly_chart(fig)
    
    def _render_polar_arithmetic(self):
        """Render polar form arithmetic operations."""
        st.subheader("Polar Form Arithmetic Operations")
        
        with st.expander("ℹ️ About Polar Arithmetic", expanded=False):
            st.markdown("""
            **Polar Form Arithmetic Rules:**
            
            **Multiplication:** |z₁z₂| = |z₁||z₂|, arg(z₁z₂) = arg(z₁) + arg(z₂)
            
            **Division:** |z₁/z₂| = |z₁|/|z₂|, arg(z₁/z₂) = arg(z₁) - arg(z₂)
            
            **Powers:** |z^n| = |z|^n, arg(z^n) = n·arg(z)
            
            **Roots:** |z^(1/n)| = |z|^(1/n), arg(z^(1/n)) = (arg(z) + 2πk)/n
            """)
        
        operation = st.selectbox("Select Polar Operation:", 
                               ["Multiplication", "Division", "Power", "nth Root"])
        
        if operation == "Multiplication":
            self._polar_multiplication()
        elif operation == "Division":
            self._polar_division()
        elif operation == "Power":
            self._polar_power()
        elif operation == "nth Root":
            self._polar_nth_root()
    
    def _polar_multiplication(self):
        """Render polar multiplication interface."""
        st.write("### Polar Form Multiplication")
        
        col1, col2 = st.columns(2)
        with col1:
            z1_input = st.text_input("First complex number z₁:", value="2+2i")
        with col2:
            z2_input = st.text_input("Second complex number z₂:", value="1+i")
        
        if st.button("Multiply in Polar Form"):
            z1 = self.parse_complex_input(z1_input)
            z2 = self.parse_complex_input(z2_input)
            
            if z1 is not None and z2 is not None:
                # Convert to polar
                r1, theta1 = self.complex_to_polar(z1)
                r2, theta2 = self.complex_to_polar(z2)
                
                # Polar multiplication
                r_result = r1 * r2
                theta_result = theta1 + theta2
                
                # Convert back to rectangular
                result = self.polar_to_complex(r_result, theta_result)
                
                st.write("### Step-by-Step Solution:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Rectangular Form:**")
                    st.latex(f"z_1 = {self.format_complex_latex(z1)}")
                    st.latex(f"z_2 = {self.format_complex_latex(z2)}")
                    
                    st.write("**Convert to Polar:**")
                    st.latex(f"z_1 = {r1:.4g}e^{{i{theta1:.4g}}}")
                    st.latex(f"z_2 = {r2:.4g}e^{{i{theta2:.4g}}}")
                
                with col2:
                    st.write("**Polar Multiplication:**")
                    st.latex(f"z_1 \\cdot z_2 = {r1:.4g}e^{{i{theta1:.4g}}} \\cdot {r2:.4g}e^{{i{theta2:.4g}}}")
                    st.latex(f"= ({r1:.4g} \\cdot {r2:.4g})e^{{i({theta1:.4g} + {theta2:.4g})}}")
                    st.latex(f"= {r_result:.4g}e^{{i{theta_result:.4g}}}")
                    
                    st.write("**Convert back to Rectangular:**")
                    st.latex(f"= {self.format_complex_latex(result)}")
                
                # Verification with direct multiplication
                direct_result = z1 * z2
                st.write("### Verification:")
                st.latex(f"\\text{{Direct multiplication: }} {self.format_complex_latex(direct_result)}")
                
                if np.allclose([result.real, result.imag], [direct_result.real, direct_result.imag]):
                    st.success("✅ Results match!")
                
                # Visualization
                fig = self.create_gaussian_plane_plot([
                    (z1, "z₁"), (z2, "z₂"), (result, "z₁ × z₂")
                ], title="Polar Multiplication Visualization")
                st.plotly_chart(fig)
    
    def _polar_division(self):
        """Render polar division interface."""
        st.write("### Polar Form Division")
        
        col1, col2 = st.columns(2)
        with col1:
            z1_input = st.text_input("Numerator z₁:", value="4+4i")
        with col2:
            z2_input = st.text_input("Denominator z₂:", value="1+i")
        
        if st.button("Divide in Polar Form"):
            z1 = self.parse_complex_input(z1_input)
            z2 = self.parse_complex_input(z2_input)
            
            if z1 is not None and z2 is not None:
                if abs(z2) < 1e-10:
                    st.error("Cannot divide by zero!")
                    return
                
                # Convert to polar
                r1, theta1 = self.complex_to_polar(z1)
                r2, theta2 = self.complex_to_polar(z2)
                
                # Polar division
                r_result = r1 / r2
                theta_result = theta1 - theta2
                
                # Convert back to rectangular
                result = self.polar_to_complex(r_result, theta_result)
                
                st.write("### Step-by-Step Solution:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Rectangular Form:**")
                    st.latex(f"\\frac{{z_1}}{{z_2}} = \\frac{{{self.format_complex_latex(z1)}}}{{{self.format_complex_latex(z2)}}}")
                    
                    st.write("**Convert to Polar:**")
                    st.latex(f"z_1 = {r1:.4g}e^{{i{theta1:.4g}}}")
                    st.latex(f"z_2 = {r2:.4g}e^{{i{theta2:.4g}}}")
                
                with col2:
                    st.write("**Polar Division:**")
                    st.latex(f"\\frac{{z_1}}{{z_2}} = \\frac{{{r1:.4g}e^{{i{theta1:.4g}}}}}{{{r2:.4g}e^{{i{theta2:.4g}}}}}")
                    st.latex(f"= \\frac{{{r1:.4g}}}{{{r2:.4g}}}e^{{i({theta1:.4g} - {theta2:.4g})}}")
                    st.latex(f"= {r_result:.4g}e^{{i{theta_result:.4g}}}")
                    
                    st.write("**Convert back to Rectangular:**")
                    st.latex(f"= {self.format_complex_latex(result)}")
                
                # Verification
                direct_result = z1 / z2
                st.write("### Verification:")
                st.latex(f"\\text{{Direct division: }} {self.format_complex_latex(direct_result)}")
                
                if np.allclose([result.real, result.imag], [direct_result.real, direct_result.imag]):
                    st.success("✅ Results match!")
    
    def _polar_power(self):
        """Render polar power interface."""
        st.write("### Complex Powers using Polar Form")
        
        col1, col2 = st.columns(2)
        with col1:
            z_input = st.text_input("Complex number z:", value="1+i")
        with col2:
            n = st.number_input("Power n:", value=3, min_value=1, max_value=20)
        
        if st.button("Calculate Power"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                # Convert to polar
                r, theta = self.complex_to_polar(z)
                
                # Calculate power
                r_result = r ** n
                theta_result = n * theta
                
                # Convert back to rectangular
                result = self.polar_to_complex(r_result, theta_result)
                
                st.write("### Step-by-Step Solution:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**De Moivre's Theorem:**")
                    st.latex(f"z^n = (re^{{i\\theta}})^n = r^n e^{{in\\theta}}")
                    
                    st.write("**Given:**")
                    st.latex(f"z = {self.format_complex_latex(z)}")
                    st.latex(f"n = {n}")
                    
                    st.write("**Polar Form:**")
                    st.latex(f"z = {r:.4g}e^{{i{theta:.4g}}}")
                
                with col2:
                    st.write("**Apply De Moivre's Theorem:**")
                    st.latex(f"z^{{{n}}} = ({r:.4g})^{{{n}}} e^{{i({n} \\cdot {theta:.4g})}}")
                    st.latex(f"= {r_result:.4g} e^{{i{theta_result:.4g}}}")
                    
                    st.write("**Convert to Rectangular:**")
                    st.latex(f"= {r_result:.4g}(\\cos({theta_result:.4g}) + i\\sin({theta_result:.4g}))")
                    st.latex(f"= {self.format_complex_latex(result)}")
                
                # Verification
                direct_result = z ** n
                st.write("### Verification:")
                st.latex(f"\\text{{Direct calculation: }} {self.format_complex_latex(direct_result)}")
                
                if np.allclose([result.real, result.imag], [direct_result.real, direct_result.imag]):
                    st.success("✅ Results match!")
                
                # Show pattern for powers
                if n <= 8:
                    st.write("### Powers Pattern:")
                    powers_data = []
                    for i in range(1, n+1):
                        power_result = z ** i
                        powers_data.append({
                            "Power": f"z^{i}",
                            "Result": self.format_complex_latex(power_result),
                            "Magnitude": f"{abs(power_result):.3f}",
                            "Argument (deg)": f"{np.degrees(cmath.phase(power_result)):.1f}°"
                        })
                    df = pd.DataFrame(powers_data)
                    st.dataframe(df)
    
    def _polar_nth_root(self):
        """Render nth root interface."""
        st.write("### Complex nth Roots using Polar Form")
        
        col1, col2 = st.columns(2)
        with col1:
            z_input = st.text_input("Complex number z:", value="8")
        with col2:
            n = st.number_input("Root index n:", value=3, min_value=2, max_value=12)
        
        if st.button("Calculate nth Roots"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                # Convert to polar
                r, theta = self.complex_to_polar(z)
                
                st.write("### All nth Roots:")
                st.write(f"Finding all {n} roots of z = {self.format_complex_latex(z)}")
                
                # Calculate all n roots
                roots = []
                for k in range(n):
                    r_root = r ** (1/n)
                    theta_root = (theta + 2*np.pi*k) / n
                    root = self.polar_to_complex(r_root, theta_root)
                    roots.append(root)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Formula for nth Roots:**")
                    st.latex(f"z^{{1/{n}}} = r^{{1/{n}}} e^{{i(\\theta + 2\\pi k)/{n}}}")
                    st.latex(f"\\text{{where }} k = 0, 1, 2, ..., {n-1}")
                    
                    st.write("**Given:**")
                    st.latex(f"z = {self.format_complex_latex(z)}")
                    st.latex(f"r = |z| = {r:.4g}")
                    st.latex(f"\\theta = \\arg(z) = {theta:.4g} \\text{{ rad}}")
                
                with col2:
                    st.write("**All Roots:**")
                    for k, root in enumerate(roots):
                        theta_k = (theta + 2*np.pi*k) / n
                        st.latex(f"z_{{{k}}} = {r**(1/n):.4g} e^{{i{theta_k:.4g}}} = {self.format_complex_latex(root)}")
                
                # Visualization
                st.write("### Visualization:")
                root_points = [(root, f"z_{k}") for k, root in enumerate(roots)]
                root_points.append((z, "Original z"))
                
                fig = self.create_gaussian_plane_plot(root_points, f"All {n}th Roots")
                st.plotly_chart(fig)
                
                # Show roots in table
                st.write("### Roots Summary:")
                roots_data = []
                for k, root in enumerate(roots):
                    r_root, theta_root = self.complex_to_polar(root)
                    roots_data.append({
                        "Root": f"z_{k}",
                        "Rectangular": self.format_complex_latex(root),
                        "Magnitude": f"{r_root:.4g}",
                        "Argument (rad)": f"{theta_root:.4g}",
                        "Argument (deg)": f"{np.degrees(theta_root):.1f}°"
                    })
                df = pd.DataFrame(roots_data)
                st.dataframe(df)
    
    def _render_powers_roots(self):
        """Render powers and roots calculator."""
        st.subheader("Complex Powers & Roots")
        
        operation = st.selectbox("Select Operation:", ["Integer Powers", "Fractional Powers", "nth Roots", "Equation Solving"])
        
        if operation == "Integer Powers":
            self._integer_powers()
        elif operation == "Fractional Powers":
            self._fractional_powers()
        elif operation == "nth Roots":
            self._general_nth_roots()
        elif operation == "Equation Solving":
            self._power_equation_solving()
    
    def _integer_powers(self):
        """Handle integer powers of complex numbers."""
        st.write("### Integer Powers of Complex Numbers")
        
        col1, col2 = st.columns(2)
        with col1:
            z_input = st.text_input("Complex number z:", value="1+i")
        with col2:
            n = st.number_input("Integer power n:", value=4, min_value=-20, max_value=20)
        
        if st.button("Calculate z^n"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                if n < 0 and abs(z) < 1e-10:
                    st.error("Cannot calculate negative power of zero!")
                    return
                
                result = z ** n
                
                # Show both rectangular and polar methods
                r, theta = self.complex_to_polar(z)
                
                st.write("### Solution using De Moivre's Theorem:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Rectangular Form:**")
                    st.latex(f"z = {self.format_complex_latex(z)}")
                    st.latex(f"z^{{{n}}} = {self.format_complex_latex(result)}")
                    
                    if n < 0:
                        st.write("**Note:** Negative power means reciprocal")
                        st.latex(f"z^{{{n}}} = \\frac{{1}}{{z^{{{abs(n)}}}}}")
                
                with col2:
                    st.write("**Polar Form Method:**")
                    st.latex(f"z = {r:.4g}e^{{i{theta:.4g}}}")
                    st.latex(f"z^{{{n}}} = ({r:.4g})^{{{n}}} e^{{i({n} \\cdot {theta:.4g})}}")
                    st.latex(f"= {r**n:.4g} e^{{i{n*theta:.4g}}}")
                    
                    # Convert back to verify
                    polar_result = self.polar_to_complex(r**n, n*theta)
                    st.latex(f"= {self.format_complex_latex(polar_result)}")
                
                # Show progression for small positive powers
                if 1 <= n <= 6:
                    st.write("### Power Progression:")
                    progression_data = []
                    for i in range(1, n+1):
                        power_i = z ** i
                        progression_data.append({
                            "Power": f"z^{i}",
                            "Result": self.format_complex_latex(power_i),
                            "|z^i|": f"{abs(power_i):.4g}"
                        })
                    df = pd.DataFrame(progression_data)
                    st.dataframe(df)
    
    def _fractional_powers(self):
        """Handle fractional powers (principal values)."""
        st.write("### Fractional Powers (Principal Values)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            z_input = st.text_input("Complex number z:", value="4")
        with col2:
            p = st.number_input("Numerator p:", value=1, min_value=1)
        with col3:
            q = st.number_input("Denominator q:", value=2, min_value=2)
        
        if st.button(f"Calculate z^({p}/{q})"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                if abs(z) < 1e-10:
                    st.error("Cannot calculate fractional power of zero!")
                    return
                
                # Calculate principal value
                r, theta = self.complex_to_polar(z)
                
                # Principal value calculation
                r_result = r ** (p/q)
                theta_result = (p * theta) / q
                
                result = self.polar_to_complex(r_result, theta_result)
                
                st.write("### Principal Value Calculation:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Formula:**")
                    st.latex(f"z^{{\\frac{{{p}}}{{{q}}}}} = r^{{\\frac{{{p}}}{{{q}}}}} e^{{i\\frac{{{p}\\theta}}{{{q}}}}}")
                    
                    st.write("**Given:**")
                    st.latex(f"z = {self.format_complex_latex(z)}")
                    st.latex(f"r = |z| = {r:.4g}")
                    st.latex(f"\\theta = \\arg(z) = {theta:.4g}")
                
                with col2:
                    st.write("**Calculation:**")
                    st.latex(f"z^{{\\frac{{{p}}}{{{q}}}}} = {r:.4g}^{{\\frac{{{p}}}{{{q}}}}} e^{{i\\frac{{{p} \\cdot {theta:.4g}}}{{{q}}}}}")
                    st.latex(f"= {r_result:.4g} e^{{i{theta_result:.4g}}}")
                    st.latex(f"= {self.format_complex_latex(result)}")
                
                st.info(f"💡 This is the principal value. There are {q} total values for z^(1/{q})")
                
                # Show all values if reasonable
                if q <= 8:
                    st.write(f"### All {q} Values of z^(1/{q}):")
                    all_roots = []
                    for k in range(q):
                        theta_k = (theta + 2*np.pi*k) / q
                        root_k = self.polar_to_complex(r**(1/q), theta_k)
                        all_roots.append(root_k)
                        st.latex(f"z_{{{k}}} = {self.format_complex_latex(root_k)}")
                    
                    # Then raise to power p
                    st.write(f"### All Values of z^({p}/{q}):")
                    for k, root in enumerate(all_roots):
                        final_value = root ** p
                        st.latex(f"(z_{{{k}}})^{{{p}}} = {self.format_complex_latex(final_value)}")
    
    def _general_nth_roots(self):
        """Handle general nth roots with all branches."""
        st.write("### All nth Roots")
        
        col1, col2 = st.columns(2)
        with col1:
            z_input = st.text_input("Complex number z:", value="1")
        with col2:
            n = st.number_input("Root index n:", value=4, min_value=2, max_value=12)
        
        if st.button(f"Find all {n}th roots"):
            z = self.parse_complex_input(z_input)
            
            if z is not None:
                if abs(z) < 1e-10:
                    st.success("All nth roots of 0 are 0")
                    return
                
                # Calculate all nth roots
                r, theta = self.complex_to_polar(z)
                roots = []
                
                for k in range(n):
                    r_root = r ** (1/n)
                    theta_root = (theta + 2*np.pi*k) / n
                    root = self.polar_to_complex(r_root, theta_root)
                    roots.append((k, root, theta_root))
                
                st.write(f"### All {n} roots of z = {self.format_complex_latex(z)}:")
                
                # Display formula
                st.latex(f"z_k = \\sqrt[{n}]{{r}} \\cdot e^{{i\\frac{{\\theta + 2\\pi k}}{{{n}}}}}, \\quad k = 0, 1, ..., {n-1}")
                
                # Show each root
                for k, root, theta_k in roots:
                    st.latex(f"z_{{{k}}} = {r**(1/n):.4g} e^{{i{theta_k:.4g}}} = {self.format_complex_latex(root)}")
                
                # Verification: check that each root^n = z
                st.write("### Verification:")
                for k, root, _ in roots:
                    verification = root ** n
                    st.write(f"z_{k}^{n} = {self.format_complex_latex(verification)}")
                
                # Geometric visualization
                root_points = [(root, f"z_{k}") for k, root, _ in roots]
                root_points.append((z, f"z (original)"))
                
                fig = self.create_gaussian_plane_plot(root_points, f"All {n}th Roots of z")
                st.plotly_chart(fig)
                
                # Roots form regular polygon
                st.info(f"💡 The {n} roots form a regular {n}-gon centered at the origin")
    
    def _power_equation_solving(self):
        """Solve equations of the form z^n = w."""
        st.write("### Solve z^n = w")
        
        col1, col2 = st.columns(2)
        with col1:
            w_input = st.text_input("Right side w:", value="1")
        with col2:
            n = st.number_input("Power n:", value=3, min_value=2, max_value=12)
        
        if st.button(f"Solve z^{n} = w"):
            w = self.parse_complex_input(w_input)
            
            if w is not None:
                if abs(w) < 1e-10:
                    st.success(f"The only solution to z^{n} = 0 is z = 0")
                    return
                
                # Solutions are nth roots of w
                r, theta = self.complex_to_polar(w)
                solutions = []
                
                for k in range(n):
                    r_sol = r ** (1/n)
                    theta_sol = (theta + 2*np.pi*k) / n
                    solution = self.polar_to_complex(r_sol, theta_sol)
                    solutions.append((k, solution))
                
                st.write(f"### Solutions to z^{n} = {self.format_complex_latex(w)}:")
                
                for k, solution in solutions:
                    st.latex(f"z_{{{k}}} = {self.format_complex_latex(solution)}")
                
                # Verification
                st.write("### Verification:")
                all_correct = True
                for k, solution in solutions:
                    verification = solution ** n
                    is_correct = np.allclose([verification.real, verification.imag], [w.real, w.imag])
                    if is_correct:
                        st.success(f"✅ z_{k}^{n} = {self.format_complex_latex(verification)}")
                    else:
                        st.error(f"❌ z_{k}^{n} = {self.format_complex_latex(verification)}")
                        all_correct = False
                
                if all_correct:
                    st.success("All solutions verified!")
    
    def _render_polynomial_equations(self):
        """Render polynomial equation solver."""
        st.subheader("Polynomial Equations with Complex Coefficients")
        
        with st.expander("ℹ️ Fundamental Theorem of Algebra", expanded=False):
            st.markdown("""
            **Fundamental Theorem of Algebra:**
            Every polynomial of degree n ≥ 1 with complex coefficients has exactly n complex roots (counting multiplicities).
            
            **Examples:**
            - z² + 1 = 0 has roots z = ±i
            - z³ - 1 = 0 has roots z = 1, ω, ω² where ω = e^(2πi/3)
            - z⁴ + 4 = 0 has 4 complex roots
            """)
        
        equation_type = st.selectbox("Select Equation Type:", 
                                   ["Quadratic (z² + pz + q = 0)", "Cubic Roots of Unity", "General z^n = c", "Custom Polynomial"])
        
        if equation_type == "Quadratic (z² + pz + q = 0)":
            self._solve_quadratic()
        elif equation_type == "Cubic Roots of Unity":
            self._cubic_roots_unity()
        elif equation_type == "General z^n = c":
            self._general_equation()
        elif equation_type == "Custom Polynomial":
            self._custom_polynomial()
    
    def _solve_quadratic(self):
        """Solve quadratic equations with complex coefficients."""
        st.write("### Quadratic Equation: z² + pz + q = 0")
        
        col1, col2 = st.columns(2)
        with col1:
            p_input = st.text_input("Coefficient p:", value="0")
        with col2:
            q_input = st.text_input("Constant q:", value="1")
        
        if st.button("Solve Quadratic"):
            p = self.parse_complex_input(p_input)
            q = self.parse_complex_input(q_input)
            
            if p is not None and q is not None:
                st.write(f"### Solving: z² + ({self.format_complex_latex(p)})z + ({self.format_complex_latex(q)}) = 0")
                
                # Quadratic formula: z = (-p ± √(p² - 4q)) / 2
                discriminant = p*p - 4*q
                sqrt_discriminant = np.sqrt(discriminant)
                
                z1 = (-p + sqrt_discriminant) / 2
                z2 = (-p - sqrt_discriminant) / 2
                
                st.write("### Using Quadratic Formula:")
                st.latex(f"z = \\frac{{-p \\pm \\sqrt{{p^2 - 4q}}}}{{2}}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Discriminant:**")
                    st.latex(f"\\Delta = p^2 - 4q")
                    st.latex(f"= ({self.format_complex_latex(p)})^2 - 4({self.format_complex_latex(q)})")
                    st.latex(f"= {self.format_complex_latex(discriminant)}")
                    
                    st.write("**Square Root of Discriminant:**")
                    st.latex(f"\\sqrt{{\\Delta}} = {self.format_complex_latex(sqrt_discriminant)}")
                
                with col2:
                    st.write("**Solutions:**")
                    st.latex(f"z_1 = \\frac{{-({self.format_complex_latex(p)}) + {self.format_complex_latex(sqrt_discriminant)}}}{{2}} = {self.format_complex_latex(z1)}")
                    st.latex(f"z_2 = \\frac{{-({self.format_complex_latex(p)}) - {self.format_complex_latex(sqrt_discriminant)}}}{{2}} = {self.format_complex_latex(z2)}")
                
                # Verification
                st.write("### Verification:")
                for i, z in enumerate([z1, z2], 1):
                    result = z*z + p*z + q
                    st.latex(f"z_{i}^2 + pz_{i} + q = {self.format_complex_latex(result)}")
                    if abs(result) < 1e-10:
                        st.success(f"✅ z_{i} is correct")
                    else:
                        st.warning(f"⚠️ z_{i} has numerical error: {abs(result):.2e}")
                
                # Visualization
                if abs(p) < 10 and abs(q) < 10:  # Only for reasonable values
                    solutions_plot = [(z1, "z₁"), (z2, "z₂")]
                    fig = self.create_gaussian_plane_plot(solutions_plot, "Quadratic Equation Solutions")
                    st.plotly_chart(fig)
    
    def _cubic_roots_unity(self):
        """Show cubic roots of unity and their properties."""
        st.write("### Cubic Roots of Unity: z³ = 1")
        
        # Calculate the three cube roots of 1
        roots = []
        for k in range(3):
            theta = 2 * np.pi * k / 3
            root = complex(np.cos(theta), np.sin(theta))
            roots.append((k, root, theta))
        
        st.write("### The three cube roots of unity:")
        
        for k, root, theta in roots:
            if k == 0:
                st.latex(f"\\omega_0 = 1")
            else:
                st.latex(f"\\omega_{k} = e^{{i\\frac{{2\\pi \\cdot {k}}}{{3}}}} = \\cos\\left(\\frac{{{k} \\cdot 2\\pi}}{{3}}\\right) + i\\sin\\left(\\frac{{{k} \\cdot 2\\pi}}{{3}}\\right) = {self.format_complex_latex(root)}")
        
        # Properties
        st.write("### Properties:")
        omega = roots[1][1]  # ω₁
        omega2 = roots[2][1]  # ω₂
        
        st.latex(f"\\omega = {self.format_complex_latex(omega)}")
        st.latex(f"\\omega^2 = {self.format_complex_latex(omega2)}")
        st.latex(f"\\omega^3 = {self.format_complex_latex(omega**3)}")
        st.latex(f"1 + \\omega + \\omega^2 = {self.format_complex_latex(1 + omega + omega2)}")
        
        # Geometric visualization
        unity_roots = [(root, f"ω_{k}") for k, root, _ in roots]
        fig = self.create_gaussian_plane_plot(unity_roots, "Cube Roots of Unity")
        st.plotly_chart(fig)
        
        st.info("💡 The cube roots of unity form an equilateral triangle on the unit circle")
    
    def _general_equation(self):
        """Solve general equations z^n = c."""
        st.write("### General Equation: z^n = c")
        
        col1, col2 = st.columns(2)
        with col1:
            c_input = st.text_input("Constant c:", value="8")
        with col2:
            n = st.number_input("Power n:", value=3, min_value=2, max_value=12)
        
        if st.button(f"Solve z^{n} = c"):
            c = self.parse_complex_input(c_input)
            
            if c is not None:
                if abs(c) < 1e-10:
                    st.success(f"z^{n} = 0 has only one solution: z = 0")
                    return
                
                st.write(f"### Solving z^{n} = {self.format_complex_latex(c)}")
                
                # Find all nth roots of c
                r, theta = self.complex_to_polar(c)
                solutions = []
                
                for k in range(n):
                    r_sol = r ** (1/n)
                    theta_sol = (theta + 2*np.pi*k) / n
                    solution = self.polar_to_complex(r_sol, theta_sol)
                    solutions.append((k, solution, theta_sol))
                
                st.write(f"### All {n} solutions:")
                
                for k, solution, theta_k in solutions:
                    st.latex(f"z_{k} = \\sqrt[{n}]{{|c|}} \\cdot e^{{i\\frac{{\\arg(c) + 2\\pi \\cdot {k}}}{{{n}}}}} = {self.format_complex_latex(solution)}")
                
                # Show in terms of roots of unity if c is real
                if abs(c.imag) < 1e-10 and c.real > 0:
                    principal_root = abs(c) ** (1/n)
                    st.write("### In terms of nth roots of unity:")
                    st.latex(f"z_k = {principal_root:.4g} \\cdot \\omega_k")
                    st.write(f"where ω_k are the {n}th roots of unity")
                
                # Visualization
                solution_points = [(sol, f"z_{k}") for k, sol, _ in solutions]
                solution_points.append((c, "c"))
                
                fig = self.create_gaussian_plane_plot(solution_points, f"Solutions to z^{n} = c")
                st.plotly_chart(fig)
    
    def _custom_polynomial(self):
        """Handle custom polynomial equations."""
        st.write("### Custom Polynomial Equation")
        st.write("For higher-degree polynomials, numerical methods are used.")
        
        degree = st.number_input("Polynomial degree:", value=3, min_value=2, max_value=8)
        
        coefficients = []
        st.write("Enter coefficients (from highest to lowest degree):")
        
        for i in range(degree + 1):
            power = degree - i
            if power == 0:
                label = "Constant term:"
            elif power == 1:
                label = "z coefficient:"
            else:
                label = f"z^{power} coefficient:"
            
            coeff_input = st.text_input(label, value="1" if i == 0 else "0", key=f"coeff_{i}")
            coeff = self.parse_complex_input(coeff_input)
            if coeff is not None:
                coefficients.append(coeff)
        
        if st.button("Find Polynomial Roots") and len(coefficients) == degree + 1:
            # Create polynomial string for display
            poly_terms = []
            for i, coeff in enumerate(coefficients):
                power = degree - i
                if abs(coeff) > 1e-10:  # Only include non-zero terms
                    if power == 0:
                        poly_terms.append(f"{self.format_complex_latex(coeff)}")
                    elif power == 1:
                        if abs(coeff - 1) < 1e-10:
                            poly_terms.append("z")
                        elif abs(coeff + 1) < 1e-10:
                            poly_terms.append("-z")
                        else:
                            poly_terms.append(f"{self.format_complex_latex(coeff)}z")
                    else:
                        if abs(coeff - 1) < 1e-10:
                            poly_terms.append(f"z^{{{power}}}")
                        elif abs(coeff + 1) < 1e-10:
                            poly_terms.append(f"-z^{{{power}}}")
                        else:
                            poly_terms.append(f"{self.format_complex_latex(coeff)}z^{{{power}}}")
            
            poly_str = " + ".join(poly_terms).replace(" + -", " - ")
            st.write(f"### Finding roots of: {poly_str} = 0")
            
            try:
                # Convert to numpy polynomial format (reverse order)
                numpy_coeffs = coefficients[::-1]
                roots = np.roots(coefficients)
                
                st.write(f"### Found {len(roots)} roots:")
                
                for i, root in enumerate(roots):
                    if np.isreal(root):
                        st.latex(f"z_{{{i+1}}} = {root.real:.6g}")
                    else:
                        st.latex(f"z_{{{i+1}}} = {self.format_complex_latex(complex(root))}")
                
                # Visualization if not too many roots
                if len(roots) <= 12:
                    root_points = [(complex(root), f"z_{i+1}") for i, root in enumerate(roots)]
                    fig = self.create_gaussian_plane_plot(root_points, f"Polynomial Roots (degree {degree})")
                    st.plotly_chart(fig)
                
                st.info(f"💡 According to the Fundamental Theorem of Algebra, this degree-{degree} polynomial has exactly {degree} roots (counting multiplicities)")
                
            except Exception as e:
                st.error(f"Error finding roots: {str(e)}")
                st.write("This might be due to numerical instability for high-degree polynomials.")