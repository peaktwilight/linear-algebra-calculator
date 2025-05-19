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
    with st.expander("Introduction to Vectors", expanded=False):
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

    # Category: Scalar (Dot) Product
    with st.expander("Scalar (Dot) Product", expanded=False):
        st.header("Scalar (Dot) Product")

        # --- Definition and Properties ---
        st.subheader("Definition and Calculation")
        st.markdown(
            "The **scalar product** (or **dot product**) of two vectors `u` and `v` (of the same dimension) results in a single scalar value. "
            "It is calculated by multiplying corresponding components and summing the results."
        )
        st.markdown("If `u = [u₁, u₂, ..., uₙ]` and `v = [v₁, v₂, ..., vₙ]`:")
        st.latex("u \\cdot v = u_1 v_1 + u_2 v_2 + \\dots + u_n v_n = \\sum_{i=1}^{n} u_i v_i")
        st.markdown("**Properties:**\n"
                    "- Commutative: `u ⋅ v = v ⋅ u`\n"
                    "- Distributive over vector addition: `u ⋅ (v + w) = u ⋅ v + u ⋅ w`\n"
                    "- Associative with scalar multiplication: `(k * u) ⋅ v = k * (u ⋅ v)`"
                   )
        st.markdown("**Conceptual Example:**")
        st.markdown("If `u = [1, 2, 3]` and `v = [4, -5, 6]`:\n"
                    "`u ⋅ v = (1)(4) + (2)(-5) + (3)(6) = 4 - 10 + 18 = 12`.")
        st.markdown("The main calculator tools for 'Vector Angle' and 'Vector Projection' inherently use the dot product. You typically don't need to calculate *just* the dot product in isolation using a dedicated tool, but understanding its calculation is crucial for these other operations.")
        st.markdown("---")

        # --- Geometric Interpretation: Angle Between Two Vectors ---
        st.subheader("Geometric Interpretation: Angle Between Two Vectors")
        st.markdown("The dot product can also be defined geometrically:")
        st.latex("u \\cdot v = ||u|| \\cdot ||v|| \\cdot \\cos(\\theta)")
        st.markdown("Where `θ` is the angle between the vectors `u` and `v`.")
        st.markdown("This formula can be rearranged to find the angle `θ`:")
        st.latex("\\cos(\\theta) = \\frac{u \\cdot v}{||u|| \\cdot ||v||}")
        st.latex("\\theta = \\arccos\\left(\\frac{u \\cdot v}{||u|| \\cdot ||v||}\\right)")
        st.markdown("The angle `θ` will be between 0° and 180° (0 and π radians).\n"
                    "- If `u ⋅ v > 0`, then `θ` is acute (0° ≤ θ < 90°).\n"
                    "- If `u ⋅ v < 0`, then `θ` is obtuse (90° < θ ≤ 180°).\n"
                    "- If `u ⋅ v = 0`, then `θ` is a right angle (90°), meaning vectors are orthogonal (see next section)."
                   )
        st.markdown("**Conceptual Example (Exercise 520784 adaptation):**")
        st.markdown("Problem: Calculate the angle between `a = [1, 1, -1]` and `b = [1, -1, 1]`.\n"
                    "1. Calculate dot product: `a ⋅ b = (1)(1) + (1)(-1) + (-1)(1) = 1 - 1 - 1 = -1`.\n"
                    "2. Calculate magnitudes: `||a|| = \\sqrt{1^2+1^2+(-1)^2} = \\sqrt{3}`. `||b|| = \\sqrt{1^2+(-1)^2+1^2} = \\sqrt{3}`.\n"
                    "3. Calculate cos(θ): `cos(θ) = -1 / (\\sqrt{3} * \\sqrt{3}) = -1 / 3`.\n"
                    "4. Calculate θ: `θ = arccos(-1/3) ≈ 1.9106` radians (or `109.47°`)."
        )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Vector Operations** in the sidebar.\n"
            "2. Select the **Vector Angle** tool.\n"
            "3. Enter the components for Vector a (e.g., `1,1,-1`) and Vector b (e.g., `1,-1,1`).\n"
            "4. The calculator will output the angle in both radians and degrees, and will show the intermediate dot product and magnitude calculations."
        )
        st.markdown("---")
        
        # --- Orthogonality ---
        st.subheader("Orthogonality (Perpendicular Vectors)")
        st.markdown(
            "Two non-zero vectors `u` and `v` are **orthogonal** (perpendicular) if and only if their dot product is zero."
        )
        st.latex("u \\perp v \\iff u \\cdot v = 0")
        st.markdown("This is a direct consequence of the angle formula: if `u ⋅ v = 0`, then `cos(θ) = 0`, which means `θ = 90°` (or π/2 radians). The zero vector is considered orthogonal to all vectors.")

        st.markdown("**Conceptual Example (Exercise 891584 adaptation):**")
        st.markdown("Problem: Determine which of the vectors `a=[263,-35,-44]`, `b=[-121,15,-48]`, `c=[71,5,-48]` are orthogonal to `v=[1,5,2]`.\n"
                    "- Test `v ⋅ a`: `(1)(263) + (5)(-35) + (2)(-44) = 263 - 175 - 88 = 0`. So, `v` is orthogonal to `a`.\n"
                    "- Test `v ⋅ b`: `(1)(-121) + (5)(15) + (2)(-48) = -121 + 75 - 96 = -142`. Not 0, so not orthogonal.\n"
                    "- Test `v ⋅ c`: `(1)(71) + (5)(5) + (2)(-48) = 71 + 25 - 96 = 0`. So, `v` is orthogonal to `c`."
        )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "While there isn't a specific 'Check Orthogonality' tool, you can use the **Vector Angle** tool:\n"
            "1. Navigate to **Vector Operations -> Vector Angle**.\n"
            "2. Input the two vectors you want to check.\n"
            "3. If the calculated angle is 90° (or π/2 radians), or if the displayed dot product is 0 (or very close to 0 due to potential floating-point inaccuracies), the vectors are orthogonal."
        )
        st.markdown("---")

        # --- Vector Projection (Shadow) ---
        st.subheader("Vector Projection (Shadow)")
        st.markdown(
            "The **projection** of vector `b` onto vector `a` (denoted `projₐb`) is like the 'shadow' that `b` casts on the line defined by `a`."
        )
        st.markdown("First, the **scalar projection** of `b` onto `a` (length of the shadow):")
        st.latex("s = \\text{comp}_a b = \\frac{a \\cdot b}{||a||}")
        st.markdown("This scalar `s` can be positive (projection in same direction as `a`), negative (opposite direction), or zero (if orthogonal). ")
        st.markdown("Then, the **vector projection** of `b` onto `a` is this scalar length multiplied by the unit vector in the direction of `a`:")
        st.latex("\\text{proj}_a b = \\left(\\frac{a \\cdot b}{||a||^2}\\right) a = \\left(\\frac{a \\cdot b}{a \\cdot a}\\right) a")

        st.markdown("**Conceptual Example (Exercise QHZIHW adaptation):**")
        st.markdown("Problem: Find the length of the shadow of `b=[12, -1]` on `a=[3, -4]`. (Note: in the exercise, `a` was already normalized, here we use the general form). Also, find the shadow vector.\n"
                    "1. Given `a=[3,-4]` and `b=[12,-1.0]`. `||a|| = \\sqrt{3^2 + (-4)^2} = 5`. `a \\cdot a = ||a||^2 = 25`.\n"
                    "2. Dot product: `a ⋅ b = (3)(12) + (-4)(-1) = 36 + 4 = 40`.\n"
                    "3. Scalar projection (length of shadow): `s = (a ⋅ b) / ||a|| = 40 / 5 = 8`.\n"
                    "   (Since `s > 0`, the angle between `a` and `b` is acute.)\n"
                    "4. Vector projection (shadow vector): `projₐb = ( (a ⋅ b) / (a ⋅ a) ) * a = (40 / 25) * [3, -4] = 1.6 * [3, -4] = [4.8, -6.4]`."
        )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Vector Operations** in the sidebar.\n"
            "2. Select the **Vector Projection/Shadow** tool.\n"
            "3. Enter Vector `a` (the vector being projected onto) and Vector `b` (the vector to be projected).\n"
            "4. The calculator will output the scalar projection (length of shadow) and the vector projection."
        )

    # Category: Vector (Cross) Product
    with st.expander("Vector (Cross) Product", expanded=False):
        st.header("Vector (Cross) Product")

        st.subheader("Definition and Calculation (3D Vectors)")
        st.markdown(
            "The **vector product** (or **cross product**) of two 3D vectors `a = [a₁, a₂, a₃]` and `b = [b₁, b₂, b₃]` results in a new 3D vector `c = a × b` "
            "that is **orthogonal** (perpendicular) to both `a` and `b`."
        )
        st.latex("a \\times b = [a_2 b_3 - a_3 b_2, \\quad a_3 b_1 - a_1 b_3, \\quad a_1 b_2 - a_2 b_1]")
        st.markdown("**Properties:**\n"
                    "- Anti-commutative: `a × b = - (b × a)` (Order matters!)\n"
                    "- Distributive over vector addition: `a × (b + c) = a × b + a × c`\n"
                    "- Not associative in general."
                   )
        st.markdown("**Conceptual Example (Exercise BT8J1D adaptation):**")
        st.markdown("Problem: Calculate `a × b` for `a = [-2, 0, 0]` and `b = [0, 9, 8]`.\n"
                    "`c₁ = (0)(8) - (0)(9) = 0`\n"
                    "`c₂ = (0)(0) - (-2)(8) = 16`\n"
                    "`c₃ = (-2)(9) - (0)(0) = -18`\n"
                    "So, `a × b = [0, 16, -18]`.")
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Vector Operations** in the sidebar.\n"
            "2. Select the **Cross Product** tool.\n"
            "3. Enter the components for Vector a and Vector b (they must be 3D, or 2D vectors which will be treated as 3D with z=0).\n"
            "4. The calculator will output the resulting cross product vector."
        )
        st.markdown("---")

        st.subheader("Geometric Interpretation: Area and Orthogonality")
        st.markdown(
            "- The **magnitude** of the cross product `||a × b||` is equal to the **area of the parallelogram** formed by vectors `a` and `b`.\n"
            "- Consequently, the **area of the triangle** formed by `a` and `b` (if they share a starting point) is `(1/2) * ||a × b||`. (Or, if A,B,C are points, Area = `(1/2) * ||AB × AC||` ).\n"
            "- The **direction** of `a × b` is given by the right-hand rule. It is always perpendicular to the plane containing `a` and `b`."
        )
        st.latex("\\text{Area of Parallelogram} = ||a \\times b|| = ||a|| \\cdot ||b|| \\cdot |\\sin(\\theta)|")
        st.latex("\\text{Area of Triangle (vertices A,B,C)} = \\frac{1}{2} || \\vec{AB} \\times \\vec{AC} ||")

        st.markdown("**Conceptual Example (Triangle Area - Exercise 62FVCH adaptation):**")
        st.markdown("Problem: Calculate the area of the triangle with vertices `A=(0,4,2)`, `B=(0,8,5)`, `C=(0,8,-1)`.\n"
                    "1. Define vectors from one vertex: `AB = B - A = [0, 4, 3]`. `AC = C - A = [0, 4, -3]`.\n"
                    "2. Calculate cross product `AB × AC`:\n"
                    "    `c₁ = (4)(-3) - (3)(4) = -12 - 12 = -24`\n"
                    "    `c₂ = (3)(0) - (0)(-3) = 0`\n"
                    "    `c₃ = (0)(4) - (4)(0) = 0`\n"
                    "    `AB × AC = [-24, 0, 0]`.\n"
                    "3. Calculate magnitude: `||AB × AC|| = \\sqrt{(-24)^2 + 0^2 + 0^2} = \\sqrt{576} = 24`.\n"
                    "4. Area of triangle = `(1/2) * 24 = 12`."
        )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "- For **Cross Product Calculation**: Use the **Vector Operations -> Cross Product** tool.\n"
            "- For **Triangle Area**: \n"
            "    1. If given vertices A,B,C, first manually calculate two vectors forming two sides of the triangle (e.g., AB = B-A, AC = C-A).\n"
            "    2. Navigate to **Vector Operations -> Triangle Area**.\n"
            "    3. Input the three original points A, B, C. The calculator handles the vector creation and cross product internally.\n"
            "    Alternatively, if you already have the two side vectors, calculate their cross product using the Cross Product tool, then find its magnitude (e.g., using Vector Normalization tool which shows magnitude) and divide by 2 manually."
        )
    
    # Category: Lines and Planes
    with st.expander("Lines and Planes", expanded=False):
        st.header("Lines and Planes")

        st.subheader("Distance from a Point to a Line")
        st.markdown(
            "To find the shortest distance from a point `P` to a line (in 2D or 3D), where the line is defined by a point `A` on the line and a direction vector `v`."
        )
        st.latex("\text{Line Equation: } X = A + t \cdot v")
        st.latex("\text{Distance } h = \frac{|| \vec{AP} \times v ||}{||v||}")
        st.markdown("Where `AP` is the vector from point `A` on the line to point `P` (i.e., `P - A`). Note that `v` must be a non-zero vector.")
        
        st.markdown("**Conceptual Example (Exercise CJ1IXZ adaptation):**")
        st.markdown("Problem: Find the distance between point `P=(5,10,0)` and the line defined by `A=(3,0,0)` and direction `v=[2,0,0]`.\n"
            "1. Vector `AP = P - A = [5-3, 10-0, 0-0] = [2, 10, 0]`.\n"
            "2. Cross product `AP × v`:\n"
            "    `c₁ = (10)(0) - (0)(0) = 0`\n"
            "    `c₂ = (0)(2) - (2)(0) = 0`\n"
            "    `c₃ = (2)(0) - (10)(2) = -20`\n"
            "    `AP × v = [0, 0, -20]`.\n"
            "3. Magnitudes: `||AP × v|| = \\sqrt{0^2+0^2+(-20)^2} = 20`. `||v|| = \\sqrt{2^2+0^2+0^2} = 2`.\n"
            "4. Distance `h = 20 / 2 = 10`."
        )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Vector Operations** in the sidebar.\n"
            "2. Select the **Point-Line Distance** tool.\n"
            "3. Enter the coordinates for point `A` on the line.\n"
            "4. Enter the components for the direction vector `v` of the line.\n"
            "5. Enter the coordinates for point `P` (the point to find the distance from).\n"
            "6. The calculator will compute and display the shortest distance."
        )
        # Placeholder for Hessian Normal Form / Point-Plane distance to be added later

    # Category: Matrix Operations
    with st.expander("Matrix Operations", expanded=False):
        st.header("Matrix Operations")

        # --- What is a Matrix? ---
        st.subheader("What is a Matrix?")
        st.markdown(
            "A matrix is a rectangular array of numbers, symbols, or expressions, arranged in rows and columns. "
            "It's a fundamental tool for representing and solving systems of linear equations, transformations, and much more."
        )
        st.markdown("An `m × n` matrix has `m` rows and `n` columns. Example of a 2x3 matrix A:")
        st.latex("A = \\begin{bmatrix} a_{11} & a_{12} & a_{13} \\newline a_{21} & a_{22} & a_{23} \\end{bmatrix}")
        st.markdown("Matrices are typically denoted by uppercase letters (e.g., A, B), and their elements by lowercase letters with subscripts (e.g., `aᵢⱼ` is the element in the i-th row and j-th column). Handling matrices often involves libraries like NumPy in Python.")
        st.markdown("---")

        # --- Matrix Addition and Subtraction ---
        st.subheader("Matrix Addition and Subtraction")
        st.markdown(
            "Matrices can be added or subtracted if and only if they have the **same dimensions** (same number of rows and same number of columns). "
            "The operation is performed element-wise."
        )
        st.markdown("If A and B are both `m × n` matrices:")
        st.latex("(A+B)_{ij} = a_{ij} + b_{ij}")
        st.latex("(A-B)_{ij} = a_{ij} - b_{ij}")
        st.markdown("**Conceptual Example (Exercise C85VFS / 4I8XHJ adaptation):**")
        st.markdown("Given `A = [[1,0], [0,1]]` and `B = [[1,0], [0,1]]` (both are 2x2 identity matrices in this case):\n"
                    "`A + B = [[1+1, 0+0], [0+0, 1+1]] = [[2,0], [0,2]]`\n"
                    "`A - B = [[1-1, 0-0], [0-0, 1-1]] = [[0,0], [0,0]]` (the zero matrix)")
        st.markdown("These operations are generally performed manually as preliminary steps for more complex problems. The calculator does not have separate tools for just matrix addition/subtraction.")
        st.markdown("---")

        # --- Scalar Multiplication of Matrices ---
        st.subheader("Scalar Multiplication of Matrices")
        st.markdown("To multiply a matrix by a scalar, multiply every element of the matrix by that scalar.")
        st.markdown("If `k` is a scalar and `A` is a matrix:")
        st.latex("(k \cdot A)_{ij} = k \cdot a_{ij}")
        st.markdown("**Conceptual Example:**")
        st.markdown("If `k = 3` and `A = [[1,2], [3,4]]`:\n"
                    "`3A = [[3*1, 3*2], [3*3, 3*4]] = [[3,6], [9,12]]`")
        st.markdown("Like addition/subtraction, this is usually a manual step. The calculator does not have a dedicated tool for scalar matrix multiplication.")
        st.markdown("---")

        # --- Matrix Multiplication (Product) ---
        st.subheader("Matrix Multiplication (Product)")
        st.markdown(
            "The product `AB` of two matrices `A` and `B` is defined if and only if the number of columns in `A` is equal to the number of rows in `B`. "
            "If `A` is an `m × n` matrix and `B` is an `n × p` matrix, then their product `AB` will be an `m × p` matrix."
        )
        st.latex("(AB)_{ij} = \\sum_{k=1}^{n} a_{ik} b_{kj}")
        st.markdown("This means the element `(AB)ᵢⱼ` is the dot product of the i-th row of `A` with the j-th column of `B`.")
        st.markdown("**Key Properties:**\n"
                    "- **Not Commutative in general:** `AB ≠ BA`\n"
                    "- Associative: `(AB)C = A(BC)`\n"
                    "- Distributive: `A(B+C) = AB + AC` and `(A+B)C = AC + BC`")
        st.markdown("**Conceptual Example (Exercise N2L782 / 7QG9AW adaptation):**")
        st.markdown("Let `A = [[1,2], [3,4]]` (2x2) and `B = [[5,6], [7,8]]` (2x2). Their product `C = AB` will be 2x2.\n"
                    "`C₁₁ = (1*5) + (2*7) = 5 + 14 = 19` (Row 1 of A ⋅ Col 1 of B)\n"
                    "`C₁₂ = (1*6) + (2*8) = 6 + 16 = 22` (Row 1 of A ⋅ Col 2 of B)\n"
                    "`C₂₁ = (3*5) + (4*7) = 15 + 28 = 43` (Row 2 of A ⋅ Col 1 of B)\n"
                    "`C₂₂ = (3*6) + (4*8) = 18 + 32 = 50` (Row 2 of A ⋅ Col 2 of B)\n"
                    "So, `AB = [[19, 22], [43, 50]]`.")
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Matrix Operations** in the sidebar."
            "2. Select the **Matrix Multiplication** tool."
            "3. Enter Matrix A (e.g., `1,2;3,4`)."
            "4. Enter Matrix B (e.g., `5,6;7,8`). Ensure dimensions are compatible."
            "5. The calculator will display the resulting product matrix."
        )

    st.markdown("---")
    st.info("More topics and interactive examples will be added soon!")

# Example of how this might be called in streamlit_app.py (for testing purposes)
if __name__ == '__main__':
    render_learning_resources_page() 