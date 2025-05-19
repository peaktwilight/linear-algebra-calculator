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
    st.markdown("Explore fundamental concepts of linear algebra, learn problem-solving approaches, and see how to use the calculator tools for various exercises.")

    # Category: Introduction to Vectors
    with st.expander("Introduction to Vectors", expanded=True):
        st.header("Introduction to Vectors")

        # --- What is a Vector? ---
        st.subheader("What is a Vector?")
        st.markdown(
            "A vector is a mathematical object that has both **magnitude** (size or length) and **direction**. "
            "Vectors can be visualized as arrows in a coordinate system.\n"
            "- In 2D (a plane): `v = [x, y]` or `v = (x, y)`\n"
            "- In 3D (space): `v = [x, y, z]` or `v = (x, y, z)`\n"
            "The numbers `x, y, z` are called the **components** of the vector."
        )
        st.markdown(
            "**Representation in Python (NumPy):** Conceptually, vectors are often handled using NumPy arrays in Python for computational purposes, like `np.array([1, 2, 3])`."
        )
        st.markdown("---")

        # --- Polar to Cartesian Coordinates ---
        st.subheader("Polar to Cartesian Coordinates")
        st.markdown(
            "While Cartesian coordinates (`x, y`) are common, 2D vectors can also be defined by polar coordinates: "
            "radius `r` (distance from origin) and an angle `phi (φ)` (from the positive x-axis)."
        )
        st.markdown("To convert from polar `(r, φ)` to Cartesian `(vx, vy)`:")
        st.latex("v_x = r \\cdot \\cos(\\phi)")
        st.latex("v_y = r \\cdot \\sin(\\phi)")
        st.markdown("Note: `φ` should be in radians for trigonometric functions in most programming libraries.")
        
        st.markdown("**Conceptual Example (Exercise NNHCXF adaptation):**")
        st.markdown("- If `r=5.0` and `phi=216.9°` (which is approx. `3.785` radians):\n"
            "  `vx = 5.0 * cos(3.785) ≈ -3.9984`\n"
            "  `vy = 5.0 * sin(3.785) ≈ -3.0021`\n"
            "  So, the vector is approximately `[-4.0, -3.0]`.")
        st.markdown("Our calculator primarily uses Cartesian coordinates for vector input. Understanding polar-to-Cartesian conversion is useful for interpreting problems that might provide vectors in polar form.")

        st.markdown("**Interactive Concept Explorer:**")
        col1_polar, col2_polar = st.columns(2)
        r_input = col1_polar.number_input("Radius (r):", value=5.0, min_value=0.0, step=0.1, key="lr_polar_r_concept")
        phi_input_deg = col2_polar.number_input("Angle (φ in degrees):", value=45.0, step=0.1, key="lr_polar_phi_deg_concept")

        if st.button("Show Cartesian Equivalent", key="lr_polar_convert_concept"):
            phi_input_rad = phi_input_deg * np.pi / 180.0
            vx = r_input * np.cos(phi_input_rad)
            vy = r_input * np.sin(phi_input_rad)
            st.info(f"Equivalent Cartesian Coordinates: `vx ≈ {vx:.4f}`, `vy ≈ {vy:.4f}`")
        
        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Vector Magnitude (Length) & Normalization ---
        st.subheader("Vector Magnitude (Length) & Normalization")
        st.markdown(
            "The **magnitude** (or length/norm) of a vector `v = [v₁, v₂, ..., vₙ]` is its size."
        )
        st.latex("||v|| = \\sqrt{v_1^2 + v_2^2 + \\dots + v_n^2}")
        st.markdown(
            "**Normalization** creates a **unit vector** (magnitude 1) in the same direction."
        )
        st.latex("\\hat{v} = \\frac{v}{||v||} \\text{ (if } ||v|| \\neq 0)")
        st.markdown("A zero vector cannot be normalized as it would involve division by zero.")

        st.markdown("**Conceptual Example (Exercise 5VRS99 adaptation):**")
        st.markdown(
            "1. Problem: Given points `A=(3,4)` and `B=(6,0)`. Find the unit vector in the direction of `u = B - A`.\n"
            "   Step 1 (Find u): `u = [6-3, 0-4] = [3, -4]`.\n"
            "   Step 2 (Calculate ||u||): `||u|| = \\sqrt{3^2 + (-4)^2} = \\sqrt{9 + 16} = \\sqrt{25} = 5`.\n"
            "   Step 3 (Normalize u): `û = u / ||u|| = [3/5, -4/5] = [0.6, -0.8]`. This is the unit vector."
        )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. If your problem involves points (like A and B to define vector AB), first calculate the components of the vector yourself (e.g., `u = B - A = [3, -4]`).\n"
            "2. Navigate to **Vector Operations** in the sidebar.\n"
            "3. Select the **Vector Normalization** tool.\n"
            "4. Enter the components of your calculated vector (e.g., `3, -4`) into the input field.\n"
            "5. The calculator will display the normalized (unit) vector and the original vector's magnitude."
        )
        st.markdown("**Quick Check (for concept understanding):**")
        vector_input_norm_concept = st.text_input("Enter vector components (e.g., 3,-4):", value="3,-4", key="lr_norm_concept_input")
        if st.button("Show Magnitude & Normalized (Concept)", key="lr_norm_concept_calc"):
            try:
                components = [float(x.strip()) for x in vector_input_norm_concept.split(',')]
                if components:
                    vec = np.array(components)
                    mag = np.linalg.norm(vec)
                    st.info(f"For vector `{vec}`: Magnitude ≈ `{mag:.4f}`")
                    if mag != 0:
                        norm_vec = vec / mag
                        st.info(f"Normalized vector ≈ `{np.round(norm_vec, 4)}`")
                    else:
                        st.warning("A zero vector has magnitude 0 and cannot be normalized.")
                else:
                    st.error("Please enter vector components.")
            except ValueError:
                st.error("Invalid format. Use comma-separated numbers e.g., 3,-4 or 1,2,2.5")

        st.markdown("--- --- --- --- --- --- --- --- --- ---")
        
        # --- Vector Addition & Subtraction ---
        st.subheader("Vector Addition & Subtraction")
        st.markdown(
            "Vectors of the **same dimension** can be added or subtracted by performing the operation on their corresponding components."
        )
        st.markdown("If `u = [u₁, u₂, ..., uₙ]` and `v = [v₁, v₂, ..., vₙ]`:")
        st.latex("u + v = [u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ]")
        st.latex("u - v = [u₁ - v₁, u₂ - v₂, ..., uₙ - vₙ]")
        st.markdown("Geometrically, `u+v` can be visualized using the 'tip-to-tail' or parallelogram method.")

        st.markdown("**Conceptual Example:**")
        st.markdown("- If `u = [1, 2]` and `v = [3, -1]`:\n"
            "  `u + v = [1+3, 2+(-1)] = [4, 1]`\n"
            "  `u - v = [1-3, 2-(-1)] = [-2, 3]`")

        st.markdown("**How this applies to calculator usage:**")
        st.markdown(
            "Our calculator does not feature a separate 'Vector Addition/Subtraction' tool, as these are fundamental operations often performed as a preliminary step.\n"
            "- **Example Scenario:** If a problem requires you to normalize `w = u + v`:\n"
            "    1.  First, calculate the components of `w` manually (by adding `u` and `v`).\n"
            "    2.  Then, use the resulting vector `w` in the **Vector Operations -> Vector Normalization** tool in our calculator.\n"
            "Many exercises, such as finding the vector `AB` from points `A` and `B` (calculated as `B-A`), will require you to perform subtraction manually before using the result in another calculator function."
        )
        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Scalar Multiplication of Vectors ---
        st.subheader("Scalar Multiplication of Vectors")
        st.markdown(
            "Multiplying a vector by a scalar (a single number) scales its magnitude and/or reverses its direction (if the scalar is negative). "
            "Each component of the vector is multiplied by the scalar."
        )
        st.markdown("If `v = [v₁, v₂, ..., vₙ]` and `k` is a scalar:")
        st.latex("k \cdot v = [k \cdot v₁, k \cdot v₂, ..., k \cdot vₙ]")
        st.markdown("- If `k > 0`, `kv` has the same direction as `v` but scaled length.\n"
            "- If `k < 0`, `kv` has the opposite direction to `v` and scaled length.\n"
            "- If `k = 0`, `kv` is the zero vector `[0, 0, ..., 0]`."
        )

        st.markdown("**Conceptual Example (Exercise SWI49N adaptation):**")
        st.markdown("- Problem: Given vector `af = [8, -0.5]`, find a vector `a` in the same direction but with length 10.\n"
            "  Step 1 (Find unit vector an): First normalize `af`. `||af|| = \\sqrt{8^2 + (-0.5)^2} = \\sqrt{64.25} \\approx 8.0156`. So, `an = af / ||af|| \approx [0.998, -0.062]`.\n"
            "  Step 2 (Scale to desired length): `a = 10 * an \approx [9.98, -0.62]`."
        )

        st.markdown("**How this applies to calculator usage:**")
        st.markdown(
            "Scalar multiplication, like addition/subtraction, is often an intermediate step performed manually.\n"
            "- **Example Scenario:** If a problem asks to find `3v` and then calculate its dot product with vector `w`:\n"
            "    1.  Calculate the components of `3v` manually.\n"
            "    2.  Then, use this resulting vector and `w` in the **Vector Operations -> Vector Angle** tool (which calculates the dot product as part of its process)."
            "The **Vector Normalization** tool itself is an application of scalar multiplication, where the scalar is `1/||v||`."
        )

    # Placeholder for next category: Scalar (Dot) Product
    # with st.expander("Scalar (Dot) Product", expanded=False):
    #     st.header("Scalar (Dot) Product")
    #     # ... content for dot product ...

    st.markdown("---")
    st.info("More topics and interactive examples will be added soon!")

# Example of how this might be called in streamlit_app.py (for testing purposes)
if __name__ == '__main__':
    render_learning_resources_page() 