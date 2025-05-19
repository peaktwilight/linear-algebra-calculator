#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Learning Resources
"""

import streamlit as st
import numpy as np
import sympy

def render_learning_resources_page():
    """
    Renders the main content for the Learning Resources page.
    This will be a series of expanders for different topics.
    """
    st.title("📚 Learning Resources")
    st.markdown("Explore fundamental concepts of linear algebra with interactive examples!")

    # Category: Introduction to Vectors
    with st.expander("Introduction to Vectors", expanded=True):
        st.header("Introduction to Vectors")

        # --- Polar to Cartesian Coordinates ---
        st.subheader("Polar to Cartesian Coordinates")
        st.markdown(
            "Vectors can be represented in different coordinate systems. "
            "Polar coordinates define a point or vector in terms of a radius `r` (distance from origin) "
            "and an angle `phi` (measured from the positive x-axis)."
        )
        st.markdown("To convert from polar coordinates `(r, φ)` to Cartesian coordinates `(vx, vy)`:")
        st.latex("v_x = r \\cdot \\cos(\\phi)")
        st.latex("v_y = r \\cdot \\sin(\\phi)")
        st.markdown("Where `φ` is typically in radians for calculations.")

        st.markdown("---")
        st.markdown("**Python Example (using NumPy):**")
        st.markdown("The following code demonstrates how to perform this conversion for given `r` and `phi` values.")
        st.code("""
import numpy as np

# Case a) r=5.0; phi=216.9 degrees
r1 = 5.0
phi1_deg = 216.9
phi1_rad = phi1_deg * np.pi / 180.0 # Convert degrees to radians
v1 = r1 * np.array([np.cos(phi1_rad), np.sin(phi1_rad)])
# Expected: v1 ≈ [-3.9984, -3.0021] (approx. [-4.0, -3.0])
print(f"Example 1: r={r1}, phi={phi1_deg}° -> v1 = {v1}")

# Case b) r=13.0; phi=-0.4 radians
r2 = 13.0
phi2_rad = -0.4 # Angle is already in radians
v2 = r2 * np.array([np.cos(phi2_rad), np.sin(phi2_rad)])
# Expected: v2 ≈ [11.9738, -5.0624]
print(f"Example 2: r={r2}, phi={phi2_rad} rad -> v2 = {v2}")
        """, language="python")
        st.markdown(
            "In Python, `np.cos()` and `np.sin()` expect the angle to be in radians. "
            "Remember to convert degrees to radians (`radians = degrees * π / 180`)."
        )

        st.markdown("---")
        st.markdown("**Interactive Conversion:**")
        st.write("Try converting your own polar coordinates to Cartesian coordinates:")

        col1_polar, col2_polar = st.columns(2)
        r_input = col1_polar.number_input("Enter radius (r):", value=5.0, min_value=0.0, step=0.1, key="lr_polar_r")
        phi_input_deg = col2_polar.number_input("Enter angle (φ in degrees):", value=45.0, step=0.1, key="lr_polar_phi_deg")

        if st.button("Convert to Cartesian", key="lr_polar_convert"):
            phi_input_rad = phi_input_deg * np.pi / 180.0
            vx = r_input * np.cos(phi_input_rad)
            vy = r_input * np.sin(phi_input_rad)
            st.success(f"Cartesian Coordinates: `vx = {vx:.4f}`, `vy = {vy:.4f}`")
            st.code(f"Vector: [{vx:.4f}, {vy:.4f}]", language="text")
        
        st.markdown("--- --- --- --- --- --- --- --- --- ---") # Visual separator for next sub-topic

        # --- Vector Magnitude (Length) & Normalization ---
        st.subheader("Vector Magnitude (Length) & Normalization")
        st.markdown(
            "The **magnitude** (or length/norm) of a vector `v = [v₁, v₂, ..., vₙ]` is a non-negative scalar "
            "representing its size. It's calculated as the square root of the sum of the squares of its components."
        )
        st.latex("||v|| = \\sqrt{v_1^2 + v_2^2 + \\dots + v_n^2}")
        st.markdown(
            "**Normalization** is the process of creating a **unit vector** (a vector with magnitude 1) "
            "that has the same direction as the original vector. This is done by dividing each component "
            "of the vector by its magnitude."
        )
        st.latex("\\hat{v} = \\frac{v}{||v||}")
        st.markdown("A unit vector is denoted by a hat symbol (e.g., `û`). Normalization is not possible for a zero vector.")

        st.markdown("---")
        st.markdown("**Python Example (using NumPy):**")
        st.markdown("The following code calculates the components of `u = B - A`, then normalizes `u` to `v`, and finally checks the length of `v`.")
        st.code("""
import numpy as np

# Given points A and B to define vector u = AB
A = np.array([3, 4])
B = np.array([6, 0])

u = B - A 
print(f"Vector u = B - A = {u}")
# Expected: u = [3 -4]

# Calculate magnitude of u
magnitude_u = np.linalg.norm(u)
print(f"Magnitude ||u|| = {magnitude_u:.4f}")
# Expected: ||u|| = 5.0

# Normalize u to get unit vector v
if magnitude_u == 0:
    v = u # Or handle as an error/special case, cannot normalize zero vector
    print("Cannot normalize a zero vector.")
else:
    v = u / magnitude_u
print(f"Normalized vector v = u / ||u|| = {v}")
# Expected: v = [0.6 -0.8]

# Calculate magnitude of v (should be 1 if normalization was successful)
magnitude_v = np.linalg.norm(v)
print(f"Magnitude ||v|| = {magnitude_v:.4f}")
# Expected: ||v|| = 1.0
        """, language="python")
        st.markdown(
            "Key NumPy function: `np.linalg.norm(vector)` calculates the magnitude. "
            "Vector division by a scalar performs element-wise division."
        )

        st.markdown("---")
        st.markdown("**Interactive Magnitude & Normalization:**")
        st.write("Enter a vector (comma-separated components) to calculate its magnitude and normalized form:")
        
        vector_input_str = st.text_input(
            "Enter vector components (e.g., 3, 4 or 1, 2, 2):", 
            value="3, 4", 
            key="lr_norm_vec_input"
        )

        if st.button("Calculate Magnitude & Normalize", key="lr_norm_calc"):
            try:
                components = [float(x.strip()) for x in vector_input_str.split(',')]
                if not components:
                    st.error("Please enter vector components.")
                else:
                    vec = np.array(components)
                    st.write(f"Input vector: `{vec}`")
                    
                    magnitude = np.linalg.norm(vec)
                    st.success(f"Magnitude (Length): `||v|| = {magnitude:.4f}`")
                    
                    if magnitude == 0:
                        st.warning("The input is a zero vector. It has magnitude 0 and cannot be normalized.")
                        st.code("Normalized vector: Not applicable for zero vector", language="text")
                    else:
                        normalized_vec = vec / magnitude
                        st.success(f"Normalized vector (Unit Vector): `û = {np.round(normalized_vec, 4)}`") # Rounded for display
                        st.code(f"Unit Vector: {normalized_vec}", language="text")
                        # Verify magnitude of normalized vector
                        # st.write(f"Magnitude of normalized vector: {np.linalg.norm(normalized_vec):.4f}") 
            except ValueError:
                st.error("Invalid input format. Please use comma-separated numbers (e.g., 3, 4 or 1, 2, 2).")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Placeholder for next category
    # with st.expander("Scalar (Dot) Product", expanded=False):
    #     st.header("Scalar (Dot) Product")
    #     ...

    st.markdown("---")
    st.info("More topics and interactive examples will be added soon!")

# Example of how this might be called in streamlit_app.py (for testing purposes)
if __name__ == '__main__':
    render_learning_resources_page() 