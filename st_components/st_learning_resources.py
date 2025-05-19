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
        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Scalar Triple Product (Spatprodukt) ---
        st.subheader("Scalar Triple Product (Spatprodukt)")
        st.markdown("""
            The **scalar triple product** (also known as the mixed product or box product) involves three 3D vectors. 
            It is calculated as the dot product of one vector with the cross product of the other two.
            """)
        st.latex("V = a \cdot (b \times c)")
        st.markdown("""
            **Geometric Interpretation: Volume**
            - The absolute value of the scalar triple product, `|a ⋅ (b × c)|`, represents the **volume of the parallelepiped** whose adjacent sides are defined by the vectors `a`, `b`, and `c`.
            - The volume of the **tetrahedron** formed by these three vectors (if they share a common origin, along with the origin itself as the fourth vertex) is `(1/6) * |a ⋅ (b × c)|`.
            """)
        st.markdown("""
            **Properties & Calculation:**
            - The scalar triple product can also be calculated as the determinant of the 3x3 matrix whose rows (or columns) are the components of the vectors `a`, `b`, and `c`:
            """)
        st.latex("a \cdot (b \times c) = \begin{vmatrix} a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \\ c_1 & c_2 & c_3 \end{vmatrix}")
        st.markdown("""
            - Cyclic permutation of the vectors does not change the value: `a ⋅ (b × c) = b ⋅ (c × a) = c ⋅ (a × b)`.
            - Swapping any two vectors negates the result: `a ⋅ (b × c) = - a ⋅ (c × b)`.
            - If the scalar triple product is zero, the three vectors are **coplanar** (they lie in the same plane). This means the parallelepiped they form has zero volume.
            """)

        st.markdown("""**Conceptual Example (Exercise T6WEYD adaptation):**""")
        st.markdown("""
            Problem: Find the volume of the parallelepiped spanned by `a = [1,4,-7]`, `b = [2,5,0]`, and `c = [3,6,12]`.
            1. First, calculate the cross product `b × c`:
               `b × c = [(5)(12) - (0)(6), (0)(3) - (2)(12), (2)(6) - (5)(3)]`
                       `= [60 - 0, 0 - 24, 12 - 15] = [60, -24, -3]`.
            2. Then, calculate the dot product `a ⋅ (b × c)`:
               `a ⋅ (b × c) = (1)(60) + (4)(-24) + (-7)(-3)`
                             `= 60 - 96 + 21 = -15`.
            3. The volume of the parallelepiped is `| -15 | = 15` cubic units.
            """)
        st.markdown("""**How to use the calculator for this:**""")
        st.markdown("""
            - **Direct Calculation:** Many calculators (including potentially this one if a 'Scalar Triple Product' or 'Volume of Parallelepiped' tool is added) might take three vectors as input.
            - **Manual Steps with existing tools:**
                1. Use the **Cross Product** tool to find `d = b × c`.
                2. Then, manually calculate the dot product `a ⋅ d`. The **Vector Angle** tool can be used to find a dot product if you observe its intermediate calculations, or you can do it component-wise: `a₁d₁ + a₂d₂ + a₃d₃`.
                3. The absolute value of this result is the volume.
            - **Using Determinant Tool:**
                1. Form a 3x3 matrix where the rows are your vectors `a`, `b`, and `c`.
                2. Use the **Determinant** tool under **Matrix Operations**.
                3. The absolute value of the determinant is the volume of the parallelepiped.
            """)

    # Category: Lines and Planes
    with st.expander("Lines and Planes", expanded=False):
        st.header("Lines and Planes")

        # --- Hessian Normal Form of a Line (2D) ---
        st.subheader("Hessian Normal Form of a Line (2D)")
        st.markdown(
            "The Hessian Normal Form (HNF) is a specific way to represent a line in 2D (or a plane in 3D). "
            "For a line in 2D, it's particularly useful for calculating the perpendicular distance from a point to the line."
        )
        st.markdown("The general equation of a line in 2D is `Ax + By + C = 0`. The Hessian Normal Form is:")
        st.latex(r"x \cdot \cos(\alpha) + y \cdot \sin(\alpha) - p = 0 ") # Raw string
        st.markdown(
            "Where:\n"
            "- `p` is the perpendicular distance from the origin to the line (`p >= 0`).\n"
            "- `α` (alpha) is the angle the normal vector to the line makes with the positive x-axis."
        )
        st.markdown("The vector `n = [cos(α), sin(α)]` is a unit normal vector to the line.")
        st.markdown("If you have the general form `Ax + By + C = 0`, you can convert it to HNF:")
        st.latex(r"p = \frac{|C|}{\sqrt{A^2 + B^2}}") # Raw string
        st.latex(r"\cos(\alpha) = \frac{A}{\pm\sqrt{A^2 + B^2}}, \quad \sin(\alpha) = \frac{B}{\pm\sqrt{A^2 + B^2}}") # Raw string
        st.markdown(
            "The sign of the denominator `±√(A² + B²)` is chosen to be opposite to the sign of `C` if `C ≠ 0`. "
            "If `C = 0`, the sign can be chosen to match the sign of `B` (or `A` if `B=0`). This ensures `p >= 0`."
        )
        st.markdown("**Distance from a Point to the Line (using HNF):**")
        st.markdown(
            "Given a point `P(x₀, y₀)`, the signed distance `d` to the line `x cos(α) + y sin(α) - p = 0` is:"
        )
        st.latex(r"d = x_0 \cos(\alpha) + y_0 \sin(\alpha) - p") # Raw string
        st.markdown(
            "- If `d > 0`, the point is on one side of the line (opposite to where the origin is, if `p > 0`).\n"
            "- If `d < 0`, the point is on the other side (same side as the origin, if `p > 0`).\n"
            "- If `d = 0`, the point is on the line.\n"
            "The absolute distance is `|d|`."
        )
        st.markdown("**Conceptual Example (Exercise 3NSMD9 adaptation):**")
        st.markdown(
            "Problem: Find the HNF and distance from origin for the line `3x - 4y + 10 = 0`. "
            "Then find the distance from point `P(5, 2)` to this line.\n"
            "1. Identify `A=3, B=-4, C=10`. Denominator: `√(3² + (-4)²) = √25 = 5`. Since `C=10` is positive, we use `-5`.\n"
            "   `cos(α) = 3/(-5) = -0.6`\n"
            "   `sin(α) = -4/(-5) = 0.8`\n"
            "   `p = 10/5 = 2` (using `|C|/√(A²+B²)`, or `p = -C / (-√(A²+B²))` to match `x cos α + y sin α - p = 0` structure, so `p = -10/(-5) = 2`).\n"
            "   HNF: `-0.6x + 0.8y - 2 = 0`. Distance from origin is `p=2`.\n"
            "2. Distance from `P(5, 2)`:\n"
            "   `d = (5)(-0.6) + (2)(0.8) - 2 = -3 + 1.6 - 2 = -3.4`.\n"
            "   The absolute distance is `|-3.4| = 3.4`."
        )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. The **Point-Line Distance** tool in the calculator (under **Lines and Planes**) typically uses the vector cross product method. "
            "To use HNF concepts directly:\n"
            "   - You would first convert your line to the form `Ax + By + C = 0`.\n"
            "   - Manually calculate `p`, `cos(α)`, `sin(α)` as shown above if you need the HNF parameters explicitly.\n"
            "   - For distance calculation: The calculator's Point-Line Distance tool is effective if you define the line using two points or a point and a direction vector. "
            "     You can convert `Ax+By+C=0` to this form (e.g., find two points on the line) if needed for the tool. "
            "     Alternatively, once you have the HNF formula `x₀ cos(α) + y₀ sin(α) - p`, you can compute the distance with simple arithmetic."
        )
        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Distance from a Point to a Line ---
        st.subheader("Distance from a Point to a Line (Vector Method)")
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
        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Hessian Normal Form of a Plane (3D) ---
        st.subheader("Hessian Normal Form of a Plane (3D)")
        st.markdown("""
            Similar to lines in 2D, the Hessian Normal Form (HNF) for a plane in 3D provides a standard way to represent the plane and is useful for calculating distances.
            """)
        st.markdown("""The general equation of a plane in 3D is `Ax + By + Cz + D = 0`. The Hessian Normal Form is:""")
        st.latex("n_x x + n_y y + n_z z - p = 0 \quad \text{or} \quad \mathbf{n} \cdot \mathbf{x} - p = 0")
        st.markdown("""
            Where:\n
            - `p` is the perpendicular distance from the origin to the plane (`p >= 0`).\n
            - `\mathbf{n} = [n_x, n_y, n_z]` is the unit normal vector to the plane (pointing from the origin towards the plane if p > 0).
            """)
        st.markdown("""If you have the general form `Ax + By + Cz + D = 0`, you can convert it to HNF:""")
        st.latex("p = \frac{|D|}{\sqrt{A^2 + B^2 + C^2}}")
        st.latex("n_x = \frac{A}{\pm\sqrt{A^2 + B^2 + C^2}}, \quad n_y = \frac{B}{\pm\sqrt{A^2 + B^2 + C^2}}, \quad n_z = \frac{C}{\pm\sqrt{A^2 + B^2 + C^2}}")
        st.markdown("""
            The sign of the denominator `±√(A² + B² + C²)` is chosen to be opposite to the sign of `D` if `D ≠ 0`. 
            If `D = 0`, the sign can be chosen arbitrarily (e.g., to make `n_z` positive, or `n_y` if `n_z=0`, etc.), though consistency is good. This ensures `p >= 0`.
            """)
        st.markdown("""**Distance from a Point to the Plane (using HNF):**""")
        st.markdown("""
            Given a point `P(x₀, y₀, z₀)`, the signed distance `d_signed` to the plane `n_x x + n_y y + n_z z - p = 0` is:
            """)
        st.latex("d_{signed} = n_x x_0 + n_y y_0 + n_z z_0 - p")
        st.markdown("""
            - If `d_signed > 0`, the point is on the side of the plane pointed to by the normal vector `n` (opposite to the origin if `p > 0`).\n
            - If `d_signed < 0`, the point is on the other side (same side as the origin if `p > 0`).\n
            - If `d_signed = 0`, the point is on the plane.\n

            The absolute distance is `|d_signed|`.
            """)
        st.markdown("""**Conceptual Example (Exercise NBI87W/BXCVJ7 adaptation):**""")
        st.markdown("""
            Problem: Find the HNF for the plane `2x - y + 2z - 6 = 0`. 
            Then find the distance from point `P(3, 5, 1)` to this plane.\n
            1. Identify `A=2, B=-1, C=2, D=-6`. Denominator: `√(2² + (-1)² + 2²) = √9 = 3`. Since `D=-6` is negative, we use `+3`.\n
               `n_x = 2/3`\n
               `n_y = -1/3`\n
               `n_z = 2/3`\n
               `p = |-6|/3 = 2`. (Or `p = -D/√(A²+B²+C²) = -(-6)/3 = 2` to fit `... - p = 0`)\n
               HNF: `(2/3)x - (1/3)y + (2/3)z - 2 = 0`. Distance from origin is `p=2`.\n
            2. Distance from `P(3, 5, 1)`:\n
               `d_signed = (2/3)(3) + (-1/3)(5) + (2/3)(1) - 2 = 2 - 5/3 + 2/3 - 2 = -3/3 = -1`.\n
               The absolute distance is `|-1| = 1`.
            """)
        st.markdown("""**How to use the calculator for this:**""")
        st.markdown("""
            1. If a **Point-Plane Distance** tool is available under **Lines and Planes** in the calculator, it will likely ask for the plane equation (e.g., `Ax+By+Cz+D=0` or coefficients) and the point's coordinates.\n
            2. If you need to use HNF concepts directly or no direct tool exists:\n
               - Convert your plane to `Ax + By + Cz + D = 0` form.\n
               - Manually calculate `p` and the components of the unit normal vector `n` as shown above if you need the HNF parameters explicitly.\n
               - For distance calculation, once you have the formula `n_x x_0 + n_y y_0 + n_z z_0 - p`, you can compute the distance with simple arithmetic.
            """)

        # --- Parametric Representation of a Line ---
        st.subheader("Parametric Representation of a Line")
        st.markdown("""
            The parametric representation is a common way to define a line in both 2D and 3D space. 
            It describes the coordinates of any point on the line as a function of a single parameter, usually denoted by `t`.
            """)
        st.markdown("""**Formula:**""")
        st.latex("L(t) = P_0 + t \cdot \mathbf{v}")
        st.markdown("""
            Where:
            - `P_0` is a specific known point on the line (e.g., `[x₀, y₀]` in 2D, or `[x₀, y₀, z₀]` in 3D).
            - `\mathbf{v}` is a non-zero direction vector parallel to the line (e.g., `[v_x, v_y]` in 2D, or `[v_x, v_y, v_z]` in 3D).
            - `t` is a scalar parameter. As `t` varies over all real numbers, `L(t)` traces out all points on the line.
            """)
        st.markdown("""
            This can be written component-wise:
            - **In 2D:** `x(t) = x₀ + t \cdot v_x`, `y(t) = y₀ + t \cdot v_y`
            - **In 3D:** `x(t) = x₀ + t \cdot v_x`, `y(t) = y₀ + t \cdot v_y`, `z(t) = z₀ + t \cdot v_z`
            """)
        st.markdown("""
            **Finding the Direction Vector from Two Points:**
            If you are given two distinct points `P_0` and `P_1` on the line, the direction vector `\mathbf{v}` can be found by: 
            `\mathbf{v} = P_1 - P_0`.
            """)
        st.markdown("""**Conceptual Example (2D - IOVMXJ adaptation):**""")
        st.markdown("""
            Problem: Find the parametric equation of the line passing through `P_0 = (1, 2)` and `P_1 = (4, 6)`.
            1. Find the direction vector: `\mathbf{v} = P_1 - P_0 = [4-1, 6-2] = [3, 4]`.
            2. Choose `P_0 = (1, 2)` as the base point.
            3. The parametric equations are:
               `x(t) = 1 + 3t`
               `y(t) = 2 + 4t`
            Or in vector form: `L(t) = [1, 2] + t[3, 4]`.
            """)
        st.markdown("""**Conceptual Example (3D - V8X27N adaptation):**""")
        st.markdown("""
            Problem: Find the parametric equation of the line passing through `A = (1, 0, 2)` and `B = (3, -1, 5)`.
            1. Find the direction vector: `\mathbf{v} = B - A = [3-1, -1-0, 5-2] = [2, -1, 3]`.
            2. Choose `A = (1, 0, 2)` as the base point `P_0`.
            3. The parametric equations are:
               `x(t) = 1 + 2t`
               `y(t) = 0 - 1t = -t`
               `z(t) = 2 + 3t`
            Or in vector form: `L(t) = [1, 0, 2] + t[2, -1, 3]`.
            """)
        st.markdown("""**How this relates to calculator usage:**""")
        st.markdown("""
            Many tools in our calculator that deal with lines (e.g., Point-Line Distance, Line Intersection, etc.) will ask you to define a line by providing a point on the line (`P_0`) and its direction vector (`\mathbf{v}`). 
            Understanding the parametric form helps you provide this input correctly, especially if a problem gives you two points on the line instead of a point and direction vector directly.
            """)

        # --- Parametric Representation of a Plane ---
        st.subheader("Parametric Representation of a Plane")
        st.markdown("""
            Similar to lines, planes in 3D space can be described using a parametric representation. 
            This form uses a point on the plane and two non-parallel vectors that lie within the plane (direction vectors).
            """)
        st.markdown("""**Formula:**""")
        st.latex("\Pi(s, t) = P_0 + s \cdot \mathbf{u} + t \cdot \mathbf{v}")
        st.markdown("""
            Where:
            - `P_0` is a specific known point on the plane (e.g., `[x₀, y₀, z₀]` in 3D).
            - `\mathbf{u}` and `\mathbf{v}` are two non-parallel direction vectors that lie in the plane. 
            - `s` and `t` are scalar parameters. As `s` and `t` vary over all real numbers, `Π(s, t)` traces out all points on the plane.
            """)
        st.markdown("""
            This can be written component-wise:
            `x(s, t) = x₀ + s \cdot u_x + t \cdot v_x`
            `y(s, t) = y₀ + s \cdot u_y + t \cdot v_y`
            `z(s, t) = z₀ + s \cdot u_z + t \cdot v_z`
            """)
        st.markdown("""
            **Finding Direction Vectors from Three Points:**
            If you are given three non-collinear points `P_0`, `P_1`, and `P_2` on the plane:
            1. Choose one point as the base, e.g., `P_0`.
            2. Form two vectors from this base point to the other two points: 
               `\mathbf{u} = P_1 - P_0`
               `\mathbf{v} = P_2 - P_0`
            These vectors `\mathbf{u}` and `\mathbf{v}` will be non-parallel (since the points are non-collinear) and lie in the plane.
            """)
        st.markdown("""**Conceptual Example (PDI7FV / QLCRW1 adaptation):**""")
        st.markdown("""
            Problem: Find the parametric equation of the plane passing through `P_0 = (1,1,1)`, `P_1 = (1,2,3)`, and `P_2 = (-1,0,1)`.
            1. Base point: `P_0 = (1,1,1)`.
            2. Direction vector `\mathbf{u} = P_1 - P_0 = [1-1, 2-1, 3-1] = [0, 1, 2]`.
            3. Direction vector `\mathbf{v} = P_2 - P_0 = [-1-1, 0-1, 1-1] = [-2, -1, 0]`.
            4. The parametric equation of the plane is:
               `Π(s, t) = [1,1,1] + s[0,1,2] + t[-2,-1,0]`
            Component-wise:
               `x(s,t) = 1 - 2t`
               `y(s,t) = 1 + s - t`
               `z(s,t) = 1 + 2s`
            """)
        st.markdown("""**How this relates to calculator usage:**""")
        st.markdown("""
            While many plane operations in calculators use the general form `Ax + By + Cz + D = 0` (often derived from the normal vector), understanding the parametric form is crucial for defining a plane from three points or when working with geometric transformations. Some advanced tools might allow plane definition using a point and two direction vectors.
            To convert from parametric form (`P_0, u, v`) to the general form, you can find the normal vector `n = u × v` and then use `n ⋅ (X - P_0) = 0` to get `n_x x + n_y y + n_z z - (n ⋅ P_0) = 0`.
            """)

        # --- Intersection of Two Lines ---
        st.subheader("Intersection of Two Lines")
        st.markdown("""
            Finding the intersection point of two lines is a common problem in geometry. The approach differs slightly between 2D and 3D space.
            """)
        st.markdown("""**Approach in 2D:**""")
        st.markdown("""
            Given two lines in parametric form:
            - Line 1: `L₁(t₁) = P₁ + t₁ \cdot \mathbf{v₁} = [x₁, y₁] + t₁[v₁ₓ, v₁y]`
            - Line 2: `L₂(t₂) = P₂ + t₂ \cdot \mathbf{v₂} = [x₂, y₂] + t₂[v₂ₓ, v₂y]`
            
            At the point of intersection, the coordinates must be equal: `L₁(t₁) = L₂(t₂)`.
            This gives a system of two linear equations for two unknowns (`t₁`, `t₂`):
            `x₁ + t₁ \cdot v₁ₓ = x₂ + t₂ \cdot v₂ₓ`
            `y₁ + t₁ \cdot v₁y = y₂ + t₂ \cdot v₂y`
            
            Rearranging:
            `t₁ \cdot v₁ₓ - t₂ \cdot v₂ₓ = x₂ - x₁`
            `t₁ \cdot v₁y - t₂ \cdot v₁y = y₂ - y₁`
            """)
        st.markdown("""
            **Solving and Interpreting the System (2D):**
            1.  **Unique Solution for `t₁`, `t₂`**: If the system has a unique solution, the lines intersect at a single point. Substitute either `t₁` back into `L₁(t₁)` or `t₂` into `L₂(t₂)` to find the coordinates of the intersection point.
            2.  **No Solution**: If the system is inconsistent (e.g., leads to `0 = 1`), the lines are parallel and distinct; they do not intersect.
            3.  **Infinitely Many Solutions**: If the system has infinitely many solutions (e.g., one equation is a multiple of the other, leading to `0 = 0`), the lines are collinear (they are the same line).
            """)
        st.markdown("""**Approach in 3D:**""")
        st.markdown("""
            The same principle applies, but with three equations (for x, y, and z components):
            `x₁ + t₁ \cdot v₁ₓ = x₂ + t₂ \cdot v₂ₓ`
            `y₁ + t₁ \cdot v₁y = y₂ + t₂ \cdot v₂y`
            `z₁ + t₁ \cdot v₁z = z₂ + t₂ \cdot v₂z`
            
            - Solve any two of these equations (e.g., for x and y) to find potential values for `t₁` and `t₂`.
            - Substitute these `t₁` and `t₂` values into the third equation (e.g., for z).
            - If the third equation holds true, the lines intersect at the point found by substituting `t₁` or `t₂` back into their respective line equations.
            - If the third equation does not hold, the lines do not intersect in 3D space (they are **skew** lines, or parallel if their direction vectors are proportional but they don't satisfy the equations).
            - If the system of all three equations yields infinitely many solutions, they are collinear.
            """)
        st.markdown("""**Conceptual Example (2D - IXZ345/7XTEAY adaptation):**""")
        st.markdown("""
            Problem: Find the intersection of Line 1: `P₁=(1,1)`, `v₁=(2,1)` and Line 2: `P₂=(0,2)`, `v₂=(1,-1)`.
            `L₁(t₁): x = 1 + 2t₁, y = 1 + t₁`
            `L₂(t₂): x = 0 + t₂,  y = 2 - t₂`
            
            Set components equal:
            1) `1 + 2t₁ = t₂`
            2) `1 + t₁  = 2 - t₂`
            
            Substitute `t₂` from (1) into (2):
            `1 + t₁ = 2 - (1 + 2t₁)`
            `1 + t₁ = 2 - 1 - 2t₁`
            `1 + t₁ = 1 - 2t₁`
            `3t₁ = 0  => t₁ = 0`
            
            Substitute `t₁ = 0` back into (1) to find `t₂`:
            `t₂ = 1 + 2(0) = 1`
            
            Since we found unique values for `t₁` and `t₂`, the lines intersect.
            Substitute `t₁ = 0` into `L₁`: `x = 1 + 2(0) = 1`, `y = 1 + 0 = 1`.
            Intersection point: `(1, 1)`.
            (Check with `t₂ = 1` in `L₂`: `x = 1`, `y = 2 - 1 = 1`. Matches.)
            """)
        st.markdown("""**How to use the calculator for this:**""")
        st.markdown("""
            A dedicated "Line Intersection" tool (if available under **Lines and Planes**) would typically:
            1. Ask for the definition of Line 1 (e.g., point `P₁` and direction vector `v₁`).
            2. Ask for the definition of Line 2 (e.g., point `P₂` and direction vector `v₂`).
            3. The calculator would then solve the system and report:
                - The intersection point if one exists.
                - That the lines are parallel if they don't intersect but have proportional direction vectors.
                - That the lines are collinear if they are identical.
                - That the lines are skew (in 3D) if they don't intersect and are not parallel.
            If no direct tool exists, you would set up and solve the system of linear equations manually, possibly using the "Solve System (Gauss)" tool if you formulate the system in matrix form.
            """)

        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Intersection Line of Two Planes ---
        st.subheader("Intersection Line of Two Planes")
        st.markdown("""
            In 3D space, two distinct planes can either be parallel or they intersect in a straight line.
            """)
        st.markdown("""
            **Conditions for Intersection:**
            - Let the two planes be given by their general equations:
              `Π₁: A₁x + B₁y + C₁z + D₁ = 0` (with normal vector `\mathbf{n₁} = [A₁, B₁, C₁]`)
              `Π₂: A₂x + B₂y + C₂z + D₂ = 0` (with normal vector `\mathbf{n₂} = [A₂, B₂, C₂]`)
            - **Parallel Planes**: If `\mathbf{n₁}` and `\mathbf{n₂}` are parallel (i.e., `\mathbf{n₁} = k \cdot \mathbf{n₂}` for some scalar `k`, or `\mathbf{n₁} \times \mathbf{n₂} = \mathbf{0}`), the planes are parallel.
                - If they are parallel and distinct (e.g., `D₁/||n1|| ≠ D₂/||n2||` after normalizing D by normal vector length, or the equations cannot be made identical by scaling), they do not intersect.
                - If they are parallel and the equations are equivalent (one is a scalar multiple of the other, including the D term), they are the same plane and intersect everywhere.
            - **Intersecting Planes**: If `\mathbf{n₁}` and `\mathbf{n₂}` are not parallel, the planes intersect in a line.
            """)
        st.markdown("""
            **1. Finding the Direction Vector (v) of the Intersection Line:**
            The intersection line lies in both planes. Therefore, its direction vector `\mathbf{v}` must be perpendicular to both normal vectors `\mathbf{n₁}` and `\mathbf{n₂}`.
            Thus, the direction vector of the intersection line can be found using the cross product:
            """)
        st.latex("\mathbf{v} = \mathbf{n₁} \times \mathbf{n₂}")
        st.markdown("""If `\mathbf{v} = \mathbf{0}`, it means the normal vectors were parallel, and the planes do not intersect in a single line (they are parallel or identical).""")
        st.markdown("""
            **2. Finding a Point (P₀) on the Intersection Line:**
            To find a specific point `P₀ = (x₀, y₀, z₀)` on the line, we need to find a common solution to the two plane equations:
            `A₁x + B₁y + C₁z = -D₁`
            `A₂x + B₂y + C₂z = -D₂`
            This is a system of two linear equations with three unknowns. 
            - **Method**: Assume one variable is a constant (e.g., set `z₀ = 0`). This reduces the system to two equations with two unknowns (x, y):
              `A₁x + B₁y = -D₁`
              `A₂x + B₂y = -D₂` (if `C₁` and `C₂` were zero, then this `z=0` step wouldn't remove z. More generally, `A₁x + B₁y = -D₁ - C₁z₀`)
            - Solve this 2x2 system for `x` and `y`. If a unique solution exists, then `(x, y, 0)` is a point on the line.
            - **If the 2x2 system has no unique solution (or is inconsistent) after setting `z=0`**: This might mean the intersection line is parallel to the xy-plane (e.g., `v_z = 0`), or that our choice `z=0` was unlucky (e.g. the line passes through the origin and `z=0` made the equations trivial). Try setting another variable to zero (e.g., `y=0` or `x=0`) and solve for the remaining two. At least one such choice should work if the planes truly intersect in a line.
            """)
        st.markdown("""
            **3. Parametric Equation of the Intersection Line:**
            Once you have a point `P₀` on the line and the direction vector `\mathbf{v}`, the parametric equation of the intersection line is:
            """)
        st.latex("L(t) = P_0 + t \cdot \mathbf{v}")
        st.markdown("""**Conceptual Example (P23RCO adaptation):**""")
        st.markdown("""
            Problem: Find the intersection line of `Π₁: x - y + z - 1 = 0` and `Π₂: 2x + y - z - 2 = 0`.
            1. Normal vectors: `\mathbf{n₁} = [1, -1, 1]`, `\mathbf{n₂} = [2, 1, -1]`.
            2. Direction vector of intersection line `\mathbf{v} = \mathbf{n₁} \times \mathbf{n₂}`:
               `v_x = (-1)(-1) - (1)(1) = 1 - 1 = 0`
               `v_y = (1)(2) - (1)(-1) = 2 + 1 = 3`
               `v_z = (1)(1) - (-1)(2) = 1 + 2 = 3`
               So, `\mathbf{v} = [0, 3, 3]`. (Since `v ≠ 0`, the planes intersect in a line).
            3. Find a point on the line. Set `z = 0` in the plane equations:
               `x - y - 1 = 0  => x - y = 1`
               `2x + y - 2 = 0 => 2x + y = 2`
               Adding the two equations: `3x = 3 => x = 1`.
               Substitute `x = 1` into `x - y = 1`: `1 - y = 1 => y = 0`.
               So, a point on the line is `P₀ = (1, 0, 0)`.
            4. Parametric equation of the line: `L(t) = [1, 0, 0] + t[0, 3, 3]`.
               Component-wise: `x(t) = 1`, `y(t) = 3t`, `z(t) = 3t`.
            """)
        st.markdown("""**How to use the calculator for this:**""")
        st.markdown("""
            A "Plane-Plane Intersection" tool (if available under **Lines and Planes**) would typically:
            1. Ask for the coefficients of Plane 1 (`A₁, B₁, C₁, D₁`).
            2. Ask for the coefficients of Plane 2 (`A₂, B₂, C₂, D₂`).
            3. The calculator would then:
                - Determine if the planes are parallel, identical, or intersect.
                - If they intersect, provide the parametric equation of the intersection line (a point on the line and its direction vector).

            If no direct tool exists:
            1. Calculate `\mathbf{v} = \mathbf{n₁} \times \mathbf{n₂}` using the **Cross Product** tool.
            2. Manually set one variable (e.g., `z=0`), form the 2x2 system for `x` and `y`, and solve it. (For example, using the **Solve System (Gauss)** tool by inputting the 2x3 augmented matrix like `A₁,B₁,-D₁;A₂,B₂,-D₂`).
            3. Combine the point and direction vector to write the line equation.
            """)
        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Distance from a Point to a Plane ---
        st.subheader("Distance from a Point to a Plane")
        st.markdown("""
            To find the shortest distance from a point `P(x₀, y₀, z₀)` to a plane defined by the equation `Ax + By + Cz + D = 0`.
            """)
        st.markdown("""**Formula:**""")
        st.latex("\text{Distance} = \frac{|Ax_0 + By_0 + Cz_0 + D|}{\sqrt{A^2 + B^2 + C^2}}")
        st.markdown("""
            Where:
            - `(x₀, y₀, z₀)` are the coordinates of the point.
            - `A, B, C` are the coefficients of the plane equation, forming the normal vector `\mathbf{n} = [A, B, C]` to the plane.
            - `D` is the constant term in the plane equation.
            - The denominator `√(A² + B² + C²)` is the magnitude of the normal vector `||\mathbf{n}||`.
            """)
        st.markdown("""
            **Relation to Hessian Normal Form (HNF):**
            Recall the HNF of a plane: `n_x x + n_y y + n_z z - p = 0`, where `\mathbf{n_{unit}} = [n_x, n_y, n_z]` is the unit normal vector and `p` is the distance from the origin to the plane.
            The signed distance from a point `P(x₀, y₀, z₀)` to this plane is `d_s = n_x x₀ + n_y y₀ + n_z z₀ - p`.
            The formula `|Ax₀ + By₀ + Cz₀ + D| / ||n||` is equivalent to `|d_s|`.
            If the plane equation `Ax + By + Cz + D = 0` is normalized by dividing by `||n||` (and adjusting D such that `p = -D_{norm} >=0`), it becomes the HNF, and the distance formula simplifies to `|x_0 n_x + y_0 n_y + z_0 n_z + D_{norm}|` which is `|x_0 n_x + y_0 n_y + z_0 n_z - p|`.
            """)
        st.markdown("""**Conceptual Example (Exercise NBI87W/3PG76V adaptation):**""")
        st.markdown("""
            Problem: Find the distance from point `P(3, 5, 1)` to the plane `2x - y + 2z - 6 = 0`.
            1. Identify point coordinates: `x₀=3, y₀=5, z₀=1`.
            2. Identify plane coefficients: `A=2, B=-1, C=2, D=-6`.
            3. Calculate the numerator: `|A x₀ + B y₀ + C z₀ + D| = |(2)(3) + (-1)(5) + (2)(1) - 6| = |6 - 5 + 2 - 6| = |-3| = 3`.
            4. Calculate the magnitude of the normal vector: `√(A² + B² + C²) = √(2² + (-1)² + 2²) = √(4 + 1 + 4) = √9 = 3`.
            5. Distance = `3 / 3 = 1`.
            (This matches the result obtained using the HNF method in the "Hessian Normal Form of a Plane (3D)" section for the same example.)
            """)
        st.markdown("""**How to use the calculator for this:**""")
        st.markdown("""
            A "Point-Plane Distance" tool (if available under **Lines and Planes**) would typically require:
            1. The coefficients of the plane equation (`A, B, C, D`).
            2. The coordinates of the point (`x₀, y₀, z₀`).
            The calculator would then directly apply the formula to compute the distance.
            
            Alternatively, if the plane is defined by three points, or a point and two vectors, the calculator might first determine the `Ax + By + Cz + D = 0` form before calculating the distance.
            """)
        st.markdown("--- --- --- --- --- --- --- --- --- ---")

        # --- Intersection of a Line and a Plane ---
        st.subheader("Intersection of a Line and a Plane")
        st.markdown("""
            A line in 3D space can intersect a plane at a single point, lie entirely within the plane, or be parallel to the plane without intersecting it.
            """)
        st.markdown("""
            **Setup:**
            - Line `L`: Parametric form `\mathbf{x} = P_0 + t \cdot \mathbf{v}`, where `P_0 = (x₀, y₀, z₀)` is a point on the line and `\mathbf{v} = [v_x, v_y, v_z]` is its direction vector.
            - Plane `Π`: General form `Ax + By + Cz + D = 0`, with normal vector `\mathbf{n} = [A, B, C]`.
            """)
        st.markdown("""
            **Condition for Intersection and Finding the Point:**
            1.  **Check for Parallelism**: Calculate the dot product `\mathbf{n} \cdot \mathbf{v}`.
                - If `\mathbf{n} \cdot \mathbf{v} = 0`: The line is parallel to the plane (or lies within it).
                    - To check if it lies within the plane, take any point on the line (e.g., `P_0`) and substitute its coordinates into the plane equation (`Ax₀ + By₀ + Cz₀ + D`).
                    - If `Ax₀ + By₀ + Cz₀ + D = 0`, the point is on the plane, so the entire line lies in the plane (infinite intersection points).
                    - If `Ax₀ + By₀ + Cz₀ + D ≠ 0`, the line is parallel to the plane and does not intersect it (no intersection points).
                - If `\mathbf{n} \cdot \mathbf{v} ≠ 0`: The line intersects the plane at exactly one point.

            2.  **Finding the Intersection Point (if `\mathbf{n} \cdot \mathbf{v} ≠ 0`):**
                Substitute the parametric equations of the line into the plane equation:
                `A(x₀ + t v_x) + B(y₀ + t v_y) + C(z₀ + t v_z) + D = 0`
                
                Solve for the parameter `t`:
                `A x₀ + A t v_x + B y₀ + B t v_y + C z₀ + C t v_z + D = 0`
                `t (A v_x + B v_y + C v_z) = -(A x₀ + B y₀ + C z₀ + D)`
                `t (\mathbf{n} \cdot \mathbf{v}) = -(\mathbf{n} \cdot P_0 + D)`
                """)
        st.latex("t = - \frac{\mathbf{n} \cdot P_0 + D}{\mathbf{n} \cdot \mathbf{v}}")
        st.markdown("""
            Once `t` is found, substitute this value back into the line's parametric equation `\mathbf{x} = P_0 + t \cdot \mathbf{v}` to get the coordinates of the intersection point.
            """)
        st.markdown("""**Conceptual Example (XF7SAW adaptation):**""")
        st.markdown("""
            Line: `P₀ = (1,2,3)`, `v = (1,1,1)` (so `x=1+t, y=2+t, z=3+t`)
            Plane: `2x - y + z - 4 = 0` (so `n = [2, -1, 1]`, `D = -4`)
            
            1. Check `n ⋅ v`: `(2)(1) + (-1)(1) + (1)(1) = 2 - 1 + 1 = 2`. Since `2 ≠ 0`, they intersect at one point.
            2. Calculate `n ⋅ P₀`: `(2)(1) + (-1)(2) + (1)(3) = 2 - 2 + 3 = 3`.
            3. Solve for `t`:
               `t = - ( (n ⋅ P₀) + D ) / (n ⋅ v) = - (3 - 4) / 2 = - (-1) / 2 = 1/2`.
            4. Substitute `t = 1/2` into line equation:
               `x = 1 + 1/2 = 1.5`
               `y = 2 + 1/2 = 2.5`
               `z = 3 + 1/2 = 3.5`
               Intersection point: `(1.5, 2.5, 3.5)`.
            (Check in plane: `2(1.5) - 2.5 + 3.5 - 4 = 3 - 2.5 + 3.5 - 4 = 0.5 + 3.5 - 4 = 4 - 4 = 0`. Correct.)
            """)
        st.markdown("""**How to use the calculator for this:**""")
        st.markdown("""
            A "Line-Plane Intersection" tool (if available) would typically require:
            1. Definition of the line (e.g., point `P₀` and direction vector `\mathbf{v}`).
            2. Definition of the plane (e.g., coefficients `A, B, C, D`, or normal vector `\mathbf{n}` and point `P₀` on plane).
            The calculator would then:
            - Determine if the line is parallel to the plane, lies within it, or intersects at a single point.
            - If a single intersection point exists, provide its coordinates.
            If no direct tool exists, you would perform the dot products and solve for `t` manually, then substitute back into the line equation.
            """)

        st.markdown("*(Placeholder for Point-Plane distance and other advanced Line/Plane topics)*")

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
        st.latex("A = \\begin{bmatrix} a_{11} & a_{12} & a_{13} \\\\newline a_{21} & a_{22} & a_{23} \\end{bmatrix}")
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
        st.markdown("""Given `A = [[1,0], [0,1]]` and `B = [[1,0], [0,1]]` (both are 2x2 identity matrices in this case):\n
`A + B = [[1+1, 0+0], [0+0, 1+1]] = [[2,0], [0,2]]`\n
`A - B = [[1-1, 0-0], [0-0, 1-1]] = [[0,0], [0,0]]` (the zero matrix)""" )
        st.markdown("These operations are generally performed manually as preliminary steps for more complex problems. The calculator does not have separate tools for just matrix addition/subtraction.")
        st.markdown("---")

        # --- Scalar Multiplication of Matrices ---
        st.subheader("Scalar Multiplication of Matrices")
        st.markdown("To multiply a matrix by a scalar, multiply every element of the matrix by that scalar.")
        st.markdown("If `k` is a scalar and `A` is a matrix:")
        st.latex("(k \cdot A)_{ij} = k \cdot a_{ij}")
        st.markdown("**Conceptual Example:**")
        st.markdown("""If `k = 3` and `A = [[1,2], [3,4]]`:\n
`3A = [[3*1, 3*2], [3*3, 3*4]] = [[3,6], [9,12]]`""")
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
        st.markdown("""**Key Properties:**\n
- **Not Commutative in general:** `AB ≠ BA`\n
- Associative: `(AB)C = A(BC)`\n
- Distributive: `A(B+C) = AB + AC` and `(A+B)C = AC + BC`""")
        st.markdown("**Conceptual Example (Exercise N2L782 / 7QG9AW adaptation):**")
        st.markdown("""Let `A = [[1,2], [3,4]]` (2x2) and `B = [[5,6], [7,8]]` (2x2). Their product `C = AB` will be 2x2.\n
`C₁₁ = (1*5) + (2*7) = 5 + 14 = 19` (Row 1 of A ⋅ Col 1 of B)\n
`C₁₂ = (1*6) + (2*8) = 6 + 16 = 22` (Row 1 of A ⋅ Col 2 of B)\n
`C₂₁ = (3*5) + (4*7) = 15 + 28 = 43` (Row 2 of A ⋅ Col 1 of B)\n
`C₂₂ = (3*6) + (4*8) = 18 + 32 = 50` (Row 2 of A ⋅ Col 2 of B)\n
So, `AB = [[19, 22], [43, 50]]`.""" )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Matrix Operations** in the sidebar.\n"
            "2. Select the **Matrix Multiplication** tool.\n"
            "3. Enter Matrix A (e.g., `1,2;3,4`).\n"
            "4. Enter Matrix B (e.g., `5,6;7,8`). Ensure dimensions are compatible.\n"
            "5. The calculator will display the resulting product matrix."
        )
        st.markdown("---")

        # --- Matrix Transpose ---
        st.subheader("Matrix Transpose")
        st.markdown(
            "The **transpose** of a matrix `A`, denoted `Aᵀ` or `A\'`, is obtained by swapping its rows and columns. "
            "If `A` is an `m × n` matrix, `Aᵀ` will be an `n × m` matrix."
        )
        st.latex("(A^T)_{ij} = A_{ji}")
        st.markdown("**Conceptual Example (Exercise G3XVHK adaptation):**")
        st.markdown("""If `A = [[1,2,3], [4,5,6]]` (a 2x3 matrix):\n
`Aᵀ = [[1,4], [2,5], [3,6]]` (a 3x2 matrix).""")
        st.markdown("The calculator has a **Matrix Transpose** tool under **Matrix Operations**.")
        st.markdown("---")

        # --- Determinants ---
        st.subheader("Determinants")
        st.markdown(
            "The determinant is a scalar value that can be computed from the elements of a **square matrix**. "
            "It is denoted as `det(A)` or `|A|`. The determinant provides important information about the matrix, such as whether it is invertible."
        )
        st.markdown("**For a 2x2 matrix:** `A = [[a,b], [c,d]]`")
        st.latex("det(A) = ad - bc")
        st.markdown("**For a 3x3 matrix:** `A = [[a,b,c], [d,e,f], [g,h,i]]` (Rule of Sarrus)")
        st.latex("det(A) = a(ei − fh) − b(di − fg) + c(dh − eg)")
        st.markdown("**Key Property:** A square matrix `A` is invertible if and only if `det(A) ≠ 0`.")
        
        st.markdown("**Conceptual Example (2x2 - Exercise 3L8XHF adaptation):**")
        st.markdown("`A = [[3,8], [4,6]]`. `det(A) = (3*6) - (8*4) = 18 - 32 = -14`.")
        st.markdown("**Conceptual Example (3x3 - Exercise VFS72X adaptation):**")
        st.markdown("""`B = [[6,1,1], [4,-2,5], [2,8,7]]`.\n
`det(B) = 6((-2)*7 - 5*8) - 1(4*7 - 5*2) + 1(4*8 - (-2)*2)`\n
`= 6(-14 - 40) - 1(28 - 10) + 1(32 + 4)`\n
`= 6(-54) - 1(18) + 1(36) = -324 - 18 + 36 = -306`.""")
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Matrix Operations** in the sidebar.\n"
            "2. Select the **Determinant** tool.\n"
            "3. Enter your square matrix (e.g., `3,8;4,6`).\n"
            "4. The calculator will display the determinant."
        )
        st.markdown("---")

        # --- Matrix Inverse ---
        st.subheader("Matrix Inverse")
        st.markdown(
            "The **inverse** of a square matrix `A`, denoted `A⁻¹`, is a matrix such that when multiplied by `A`, it yields the identity matrix `I`."
        )
        st.latex("A A^{-1} = A^{-1} A = I")
        st.markdown("A matrix has an inverse if and only if its determinant is non-zero (`det(A) ≠ 0`). Such a matrix is called **invertible** or **non-singular**.")
        st.markdown("**For a 2x2 matrix:** `A = [[a,b], [c,d]]`")
        st.latex("A^{-1} = \\frac{1}{ad-bc} \\begin{bmatrix} d & -b \\\\newline -c & a \\end{bmatrix}")
        st.markdown("For 3x3 matrices (and larger), the calculation is more complex, often involving methods like Gaussian elimination or the adjugate matrix method. Our calculator handles this for you.")
        
        st.markdown("**Conceptual Example (2x2 - Exercise P32VFS adaptation):**")
        st.markdown("""If `A = [[3,8], [4,6]]`. We found `det(A) = -14`.\n
`A⁻¹ = (1/-14) * [[6, -8], [-4, 3]] = [[-6/14, 8/14], [4/14, -3/14]] = [[-3/7, 4/7], [2/7, -3/14]]`.""")
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Matrix Operations** in the sidebar.\n"
            "2. Select the **Matrix Inverse** tool.\n"
            "3. Enter your square matrix.\n"
            "4. The calculator will display the inverse matrix if it exists, or state if the matrix is singular (not invertible)."
        )
    # Category: Systems of Linear Equations
    with st.expander("Systems of Linear Equations", expanded=False):
        st.header("Systems of Linear Equations")

        st.subheader("Solving Systems with Gaussian Elimination")
        st.markdown(
            "A system of linear equations can be represented by an **augmented matrix** `[A|b]`, where `A` is the coefficient matrix and `b` is the vector of constants."
            "Gaussian elimination is a method to solve such systems by transforming this augmented matrix into **Row Echelon Form (REF)** or **Reduced Row Echelon Form (RREF)** using elementary row operations."
        )
        st.markdown("**Elementary Row Operations:**")
        st.markdown(
            "1. Swapping two rows.\n"
            "2. Multiplying a row by a non-zero scalar.\n"
            "3. Adding a multiple of one row to another row."
        )
        st.markdown("""**Goal (REF):**\n
- All non-zero rows are above any rows of all zeros.\n
- The leading coefficient (pivot) of a non-zero row is always strictly to the right of the leading coefficient of the row above it.\n
- All entries in a column below a leading coefficient are zero.""" )
        st.markdown("""**Goal (RREF):**\n
- It is in REF.\n
- Every leading coefficient is 1.\n
- Every leading coefficient is the only non-zero entry in its column.""" )
        st.markdown("""**Interpreting the Result (from RREF):**\n
- **Unique Solution:** If the number of non-zero rows equals the number of variables, and there are no contradictions (e.g., `0 = 1`). The RREF will look like an identity matrix on the left side `[I|s]`, where `s` is the solution vector.\n
- **No Solution:** If any row has the form `[0 0 ... 0 | c]` where `c ≠ 0`. This indicates a contradiction like `0 = c`.\n
- **Infinitely Many Solutions:** If the number of non-zero rows is less than the number of variables, and there are no contradictions. This means there are free variables.""" )
        st.markdown("**Conceptual Example (Adapted from E6PZXQ / N7QLXH / S7A4P2):**")
        st.markdown("""Consider the system: `x + 2y = 5`, `3x + 4y = 11`.\n
Augmented Matrix: `[[1,2,5], [3,4,11]]`.\n
1. `R₂ → R₂ - 3R₁`: `[[1,2,5], [0,-2,-4]]`.\n
2. `R₂ → R₂ / -2`: `[[1,2,5], [0,1,2]]` (This is REF).\n
3. `R₁ → R₁ - 2R₂`: `[[1,0,1], [0,1,2]]` (This is RREF).\n
Solution: `x = 1`, `y = 2`.""" )
        st.markdown("**How to use the calculator for this:**")
        st.markdown(
            "1. Navigate to **Matrix Operations** in the sidebar.\n"
            "2. Select the **Solve System (Gauss)** tool.\n"
            "3. Enter the augmented matrix (coefficient matrix and constant vector combined, e.g., `1,2,5;3,4,11` for the example above).\n"
            "4. The calculator will perform Gaussian elimination, show the RREF, and provide the solution or indicate the type of solution (no solution, infinitely many)."
        )

        st.markdown("---")
    st.info("More topics and interactive examples will be added soon!")

# Example of how this might be called in streamlit_app.py (for testing purposes)
if __name__ == '__main__':
    render_learning_resources_page() 