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
            
            # Handle different formats
            if 'j' in input_str.lower():
                # Python format: 3+4j or 3+4J
                return complex(input_str.replace('J', 'j'))
            elif 'i' in input_str.lower():
                # Math format: 3+4i
                input_str = input_str.replace('i', 'j').replace('I', 'j')
                return complex(input_str)
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
            ["Addition", "Multiplication", "Division", "Conjugate", "Polar Form", "Gaussian Plane Visualization", "Exercise Examples"]
        )
        
        if operation == "Addition":
            self._render_addition()
        elif operation == "Multiplication":
            self._render_multiplication()
        elif operation == "Division":
            self._render_division()
        elif operation == "Conjugate":
            self._render_conjugate()
        elif operation == "Polar Form":
            self._render_polar_form()
        elif operation == "Gaussian Plane Visualization":
            self._render_gaussian_plane()
        elif operation == "Exercise Examples":
            self._render_exercise_examples()
    
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
                    st.latex(f"= {self.format_complex_latex(result['result'])}")
                    
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
    
    def _render_exercise_examples(self):
        """Render specific exercise examples from Week 20."""
        st.subheader("Week 20 Exercise Examples")
        
        example = st.selectbox(
            "Select Example:",
            ["14.2 Complex Plane (KICGBV)", "14.3 Gaussian Plane (KGX8MQ)", 
             "14.4 Addition & Multiplication (1EUM2V)", "14.5 Operations (2FXNNA)"]
        )
        
        if example == "14.2 Complex Plane (KICGBV)":
            st.write("### Example 14.2: Plot complex numbers on the Gaussian plane")
            
            # Example numbers
            examples = [
                ("2+3i", "a"),
                ("-1+2i", "b"), 
                ("3-i", "c"),
                ("-2-2i", "d")
            ]
            
            numbers = []
            for z_str, label in examples:
                z = self.parse_complex_input(z_str)
                if z:
                    numbers.append((z, f"z_{label} = {z_str}"))
            
            fig = self.create_gaussian_plane_plot(numbers, "Example 14.2: Complex Numbers")
            st.plotly_chart(fig)
            
        elif example == "14.4 Addition & Multiplication (1EUM2V)":
            st.write("### Example 14.4: Complex Arithmetic")
            
            # Predefined examples
            examples = [
                ("Addition", "2+3i", "1-i", "a"),
                ("Multiplication", "2+i", "3-2i", "g"),
                ("Division", "4+2i", "1+i", "h")
            ]
            
            for op_type, z1_str, z2_str, part in examples:
                with st.expander(f"Part {part}: {op_type} of {z1_str} and {z2_str}"):
                    z1 = self.parse_complex_input(z1_str)
                    z2 = self.parse_complex_input(z2_str)
                    
                    if z1 and z2:
                        if op_type == "Addition":
                            result = self.complex_addition(z1, z2)
                            st.latex(f"{z1_str} + {z2_str} = {self.format_complex_latex(result['result'])}")
                        elif op_type == "Multiplication":
                            result = self.complex_multiplication(z1, z2)
                            st.latex(f"({z1_str}) \\times ({z2_str}) = {self.format_complex_latex(result['result'])}")
                        elif op_type == "Division":
                            result = self.complex_division(z1, z2)
                            if 'error' not in result:
                                st.latex(f"\\frac{{{z1_str}}}{{{z2_str}}} = {self.format_complex_latex(result['result'])}")