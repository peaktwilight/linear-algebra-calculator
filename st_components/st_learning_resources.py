#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Learning Resources
"""

import streamlit as st

def render_learning_resources():
    """Renders the learning resources section in the Streamlit app."""
    st.sidebar.subheader("Learning Topics")
    learning_topic = st.sidebar.radio(
        "Choose a topic:",
        ["Browse Help Topics", "Show Example Exercises", "Linear Algebra Guide", 
         "Common Homework Patterns", "Recognize Problem Types"]
    )
    
    st.title("📚 Learning Resources")
    st.markdown("---")

    if learning_topic == "Browse Help Topics":
        st.subheader("📖 Browse Help Topics")
        topics = {
            "Vector Basics": """
[bold]Vector Representation and Components[/bold]
- A vector is an ordered list of numbers representing magnitude and direction
- In 2D: v = [x, y] represents a point or displacement
- In 3D: v = [x, y, z] represents a point or displacement in space
- Components can be real numbers, including decimals and negative values

[bold]Vector Addition and Subtraction[/bold]
- Add/subtract corresponding components
- Example: [1, 2] + [3, 4] = [4, 6]
- Geometric interpretation: Tip-to-tail method
- Properties: Commutative, Associative, Identity element [0, 0]

[bold]Scalar Multiplication[/bold]
- Multiply each component by a scalar
- Example: 2 * [1, 2] = [2, 4]
- Geometric interpretation: Scaling the vector
- Properties: Distributive, Associative

[bold]Vector Magnitude and Direction[/bold]
- Magnitude (length): |v| = √(x² + y² + z²)
- Direction: Unit vector in same direction
- Example: |[3, 4]| = 5
- Normalization: v/|v| gives unit vector

[bold]Dot Product[/bold]
- Algebraic: a·b = a₁b₁ + a₂b₂ + a₃b₃
- Geometric: a·b = |a||b|cos(θ)
- Properties: Commutative, Distributive
- Applications: Projections, angles, orthogonality

[bold]Cross Product (3D)[/bold]
- Result is perpendicular to both vectors
- |a × b| = |a||b|sin(θ)
- Direction follows right-hand rule
- Applications: Areas, volumes, torque
            """,
            "Matrix Operations": """
[bold]Matrix Representation[/bold]
- Rectangular array of numbers
- Dimensions: m × n (rows × columns)
- Element notation: aᵢⱼ (row i, column j)
- Special matrices: Identity, Zero, Diagonal

[bold]Matrix Addition/Subtraction[/bold]
- Add/subtract corresponding elements
- Matrices must have same dimensions
- Example: [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
- Properties: Commutative, Associative

[bold]Matrix Multiplication[/bold]
- (AB)ᵢⱼ = Σₖ aᵢₖbₖⱼ
- Number of columns of A must equal rows of B
- Result has dimensions: rows of A × columns of B
- Properties: Associative, Distributive
- Not commutative: AB ≠ BA in general

[bold]Transpose[/bold]
- Flip rows and columns: (A^T)ᵢⱼ = Aⱼᵢ
- Properties: (A^T)^T = A, (AB)^T = B^T A^T
- Symmetric matrix: A = A^T
- Skew-symmetric: A = -A^T

[bold]Determinants[/bold]
- 2×2: det([[a,b],[c,d]]) = ad - bc
- 3×3: Use cofactor expansion
- Properties: det(AB) = det(A)det(B)
- Applications: Inverses, areas, volumes

[bold]Matrix Inverse[/bold]
- AA⁻¹ = A⁻¹A = I
- Only square matrices can have inverses
- Not all square matrices have inverses
- 2×2 formula: [[a,b],[c,d]]⁻¹ = (1/det) * [[d,-b],[-c,a]]
            """,
            "Linear Systems": """
[bold]Systems of Linear Equations[/bold]
- General form: Ax = b
- A is coefficient matrix
- x is vector of variables
- b is right-hand side vector

[bold]Gaussian Elimination[/bold]
- Convert to row echelon form
- Use elementary row operations
- Back substitution to solve

[bold]Row Echelon Form[/bold]
- Leading coefficient (pivot) is 1
- Pivots move right as you go down
- Zeros below pivots
- Example:
  [[1,2,3],
   [0,1,4],
   [0,0,1]]

[bold]Reduced Row Echelon Form[/bold]
- Row echelon form plus:
- Zeros above pivots
- Pivots are only non-zero entries in their columns
- Example:
  [[1,0,0],
   [0,1,0],
   [0,0,1]]

[bold]Homogeneous Systems[/bold]
- Form: Ax = 0
- Always has solution x = 0
- May have non-trivial solutions
- Solution space is null space of A

[bold]Particular Solutions[/bold]
- Any solution to Ax = b
- Can be found by:
  * Gaussian elimination
  * Matrix inverse (if A is invertible)
  * Cramer's rule

[bold]Free Variables[/bold]
- Variables not corresponding to pivots
- Can be assigned arbitrary values
- Lead to parametric solutions
- Number = n - rank(A)
            """,
            "Geometric Applications": """
[bold]Lines in 3D[/bold]
- Parametric form: r = r₀ + tv
- r₀ is point on line
- v is direction vector
- t is parameter
- Example: r = [1,2,3] + t[4,5,6]

[bold]Planes in 3D[/bold]
- Point-normal form: n·(r - r₀) = 0
- n is normal vector
- r₀ is point on plane
- General form: ax + by + cz + d = 0
- Example: 2x + 3y + 4z = 5

[bold]Distance Calculations[/bold]
- Point to line: d = |v × (P-P₀)|/|v|
- Point to plane: d = |n·(P-P₀)|/|n|
- Line to line: d = |(P₂-P₁)·(v₁×v₂)|/|v₁×v₂|
- Example: Distance from [1,2,3] to plane 2x+3y+4z=5

[bold]Projections[/bold]
- Vector onto vector: projᵥu = (u·v/|v|²)v
- Vector onto plane: u - projₙu
- Example: Project [1,2,3] onto [4,5,6]

[bold]Areas and Volumes[/bold]
- Triangle area: |AB × AC|/2
- Parallelogram area
- Parallelepiped volume
- Use determinants

[bold]Intersections[/bold]
- Line-plane intersection
- Plane-plane intersection
- Line-line intersection
- Use parametric equations

[bold]Orthogonality[/bold]
- Vectors: a·b = 0
- Lines: direction vectors are perpendicular
- Planes: normal vectors are perpendicular
- Example: [1,0] and [0,1] are orthogonal
            """
        }
        for topic_title, content in topics.items():
            content_md = content.replace("[bold]", "**").replace("[/bold]", "**")
            with st.expander(topic_title, expanded=False):
                st.markdown(content_md)
    
    elif learning_topic == "Show Example Exercises":
        st.subheader("🧪 Show Example Exercises")
        examples = {
            "Vector Operations": """
**Example 1: Vector Addition and Subtraction**
Given: a = [1, 2, 3], b = [4, 5, 6]
Find: a + b and a - b

Solution:
a + b = [1+4, 2+5, 3+6] = [5, 7, 9]
a - b = [1-4, 2-5, 3-6] = [-3, -3, -3]

**Example 2: Dot Product and Angle**
Given: a = [1, 2, 3], b = [4, 5, 6]
Find: a·b and the angle between a and b

Solution:
a·b = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
|a| = √(1² + 2² + 3²) = √14
|b| = √(4² + 5² + 6²) = √77
cos(θ) = (a·b)/(|a||b|) = 32/(√14 * √77)
θ ≈ 0.225 radians ≈ 12.9 degrees

**Example 3: Cross Product**
Given: a = [1, 0, 0], b = [0, 1, 0]
Find: a × b

Solution:
a × b = [0*0 - 0*1, 0*0 - 1*0, 1*1 - 0*0]
      = [0, 0, 1]

**Example 4: Vector Projection**
Given: a = [3, -4], b = [12, -1]
Find: projᵦa

Solution:
projᵦa = (a·b/|b|²)b
a·b = 3*12 + (-4)*(-1) = 36 + 4 = 40
|b|² = 12² + (-1)² = 144 + 1 = 145
projᵦa = (40/145)[12, -1] ≈ [3.31, -0.28]
            """,
            "Matrix Operations": """
**Example 1: Matrix Multiplication**
Given: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
Find: AB

Solution:
AB = [[1*5 + 2*7, 1*6 + 2*8],
      [3*5 + 4*7, 3*6 + 4*8]]
   = [[19, 22],
      [43, 50]]

**Example 2: Gaussian Elimination**
Given: System:
  2x + y = 5
  3x + 2y = 8
Find: Solution

Solution:
Augmented matrix: [[2, 1, 5], [3, 2, 8]]
Step 1: R₂ - (3/2)R₁ → [[2, 1, 5], [0, 1/2, 1/2]]
Step 2: 2R₂ → [[2, 1, 5], [0, 1, 1]]
Step 3: R₁ - R₂ → [[2, 0, 4], [0, 1, 1]]
Step 4: (1/2)R₁ → [[1, 0, 2], [0, 1, 1]]
Therefore: x = 2, y = 1

**Example 3: Determinant and Inverse**
Given: A = [[1, 2], [3, 4]]
Find: det(A) and A⁻¹

Solution:
det(A) = 1*4 - 2*3 = 4 - 6 = -2
A⁻¹ = (1/-2) * [[4, -2], [-3, 1]]
    = [[-2, 1], [1.5, -0.5]]

**Example 4: Eigenvalues**
Given: A = [[1, 2], [3, 4]]
Find: Eigenvalues and eigenvectors

Solution:
Characteristic equation: det(A - λI) = 0
det([[1-λ, 2], [3, 4-λ]]) = 0
(1-λ)(4-λ) - 6 = 0
λ² - 5λ - 2 = 0
λ = (5 ± √33)/2
Eigenvalues: λ₁ ≈ 5.37, λ₂ ≈ -0.37
            """,
            "Geometric Problems": """
**Example 1: Point-Line Distance**
Given: Point P(1, 2, 3), Line through A(0, 0, 0) with direction v[1, 1, 1]
Find: Distance from P to line

Solution:
d = |v × (P-A)|/|v|
P-A = [1, 2, 3]
v × (P-A) = [1, -2, 1]
|v × (P-A)| = √6
|v| = √3
d = √6/√3 = √2 ≈ 1.414

**Example 2: Plane Intersection**
Given: Planes:
  x + y + z = 1
  2x + 3y + 4z = 5
Find: Intersection

Solution:
Augmented matrix: [[1, 1, 1, 1], [2, 3, 4, 5]]
RREF: [[1, 0, -1, -2], [0, 1, 2, 3]]
Parametric solution:
x = -2 + t
y = 3 - 2t
z = t
Direction vector: [1, -2, 1]
Point on line: [-2, 3, 0]

**Example 3: Triangle Area**
Given: Points A(0, 0), B(3, 0), C(0, 4)
Find: Area

Solution:
AB = [3, 0]
AC = [0, 4]
Cross product magnitude = |3*4 - 0*0| = 12
Area = 12/2 = 6 square units

**Example 4: Point-Plane Distance**
Given: Plane 2x + 3y - z = 5, Point P(1, 2, 3)
Find: Distance from P to plane

Solution:
Normal vector n = [2, 3, -1]
Point on plane Q(0, 0, -5)
d = |n·(P-Q)|/|n|
P-Q = [1, 2, 8]
n·(P-Q) = 0
|n| = √14
d = 0/√14 = 0 (point lies on plane)
            """
        }
        for category_title, content in examples.items():
            content_md = content.replace("[bold]", "**").replace("[/bold]", "**")
            with st.expander(category_title, expanded=False):
                st.markdown(content_md)

    elif learning_topic == "Linear Algebra Guide":
        st.subheader("📘 Linear Algebra Guide")
        guide = """**Linear Algebra: A Comprehensive Guide**

**1. Vector Spaces**
**Definition and Properties**
- A vector space is a set of vectors with operations of addition and scalar multiplication
- Must satisfy 10 axioms (closure, associativity, commutativity, etc.)
- Examples: R², R³, Rⁿ, space of polynomials, space of matrices

**Subspaces**
- A subset of a vector space that is itself a vector space
- Must be closed under addition and scalar multiplication
- Examples: Lines through origin, planes through origin
- Null space and column space of a matrix are subspaces

**Linear Independence**
- A set of vectors is linearly independent if no vector can be written as a linear combination of others
- Test: Set up equation c₁v₁ + c₂v₂ + ... + cₙvₙ = 0
- If only solution is c₁ = c₂ = ... = cₙ = 0, vectors are independent
- Maximum number of independent vectors = dimension of space

**Basis and Dimension**
- Basis: Set of linearly independent vectors that span the space
- Dimension: Number of vectors in any basis
- Standard basis: [1,0,0], [0,1,0], [0,0,1] for R³
- Change of basis: Use transition matrix

**Orthogonal and Orthonormal Bases**
- Orthogonal: All vectors are perpendicular
- Orthonormal: Orthogonal and all vectors have length 1
- Gram-Schmidt process to create orthonormal basis
- QR decomposition of matrices

**2. Linear Transformations**
**Definition and Properties**
- Function T: V → W that preserves vector operations
- T(u + v) = T(u) + T(v)
- T(cu) = cT(u)
- Examples: Rotation, reflection, projection

**Matrix Representation**
- Every linear transformation can be represented by a matrix
- Columns are images of basis vectors
- Composition of transformations = matrix multiplication
- Change of basis affects matrix representation

**Kernel and Image**
- Kernel: Set of vectors mapped to zero
- Image: Set of all possible outputs
- Rank-nullity theorem: dim(ker) + dim(im) = dim(domain)
- Applications to solving systems of equations

**Eigenvalues and Eigenvectors**
- λ is eigenvalue if Av = λv for some v ≠ 0
- v is eigenvector corresponding to λ
- Characteristic polynomial: det(A - λI) = 0
- Applications: Diagonalization, stability analysis

**3. Matrix Operations**
**Basic Operations**
- Addition: Element-wise
- Multiplication: Dot product of rows and columns
- Transpose: Flip rows and columns
- Properties: Associative, distributive, not commutative

**Special Matrices**
- Identity: Iᵢⱼ = 1 if i=j, 0 otherwise
- Diagonal: Non-zero only on main diagonal
- Symmetric: A = A^T
- Orthogonal: A^T A = I

**Determinants**
- Scalar value associated with square matrix
- Properties: det(AB) = det(A)det(B)
- det(A) = 0 iff A is singular
- Applications: Volume, area, solving systems

**Matrix Inverse**
- A⁻¹ exists iff det(A) ≠ 0
- AA⁻¹ = A⁻¹A = I
- (AB)⁻¹ = B⁻¹A⁻¹
- Methods: Adjugate, row operations

**4. Systems of Linear Equations**
**Gaussian Elimination**
- Convert to row echelon form
- Use elementary row operations
- Back substitution to solve
- Pivoting for numerical stability

**Row Echelon Form**
- Leading coefficient (pivot) is 1
- Pivots move right as you go down
- Zeros below pivots
- Reduced form: Zeros above pivots

**Solution Spaces**
- Unique solution: No free variables
- Infinitely many solutions: Free variables
- No solution: Inconsistent system
- Parametric form of solution

**5. Inner Product Spaces**
**Dot Product**
- Algebraic: a·b = Σaᵢbᵢ
- Geometric: a·b = |a||b|cos(θ)
- Properties: Commutative, distributive
- Applications: Projections, angles

**Orthogonality**
- Vectors are orthogonal if a·b = 0
- Orthogonal complement
- Projection onto subspace
- Least squares approximation

**6. Applications**
**Computer Graphics**
- Rotation matrices
- Projection matrices
- Transformation composition
- 3D rendering

**Quantum Mechanics**
- State vectors
- Hermitian operators
- Eigenvalue problems
- Wave functions

**Machine Learning**
- Principal Component Analysis
- Singular Value Decomposition
- Linear regression
- Feature transformation

**Network Analysis**
- Adjacency matrices
- Graph theory
- PageRank algorithm
- Social network analysis

**Optimization**
- Linear programming
- Quadratic programming
- Constraint optimization
- Gradient descent
        """
        st.markdown(guide.replace("[bold]", "**").replace("[/bold]", "**"))

    elif learning_topic == "Common Homework Patterns":
        st.subheader("🔍 Common Homework Patterns")
        patterns = """**Common Linear Algebra Homework Patterns**

**1. Vector Operations**
**Finding Unit Vectors**
- Given vector v, find v/|v|
- Example: v = [3, 4] → [3/5, 4/5]
- Common in physics and computer graphics
- Used in normalization and direction calculations

**Computing Projections**
- Project vector a onto vector b
- Formula: projᵦa = (a·b/|b|²)b
- Applications: Force decomposition, shadow calculations
- Related to least squares approximation

**Checking Orthogonality**
- Test if vectors are perpendicular
- Check if dot product is zero
- Example: [1, 0] and [0, 1] are orthogonal
- Applications: Basis construction, coordinate systems

**Calculating Angles**
- Use dot product formula: cos(θ) = (a·b)/(|a||b|)
- Convert between radians and degrees
- Example: angle between [1, 1] and [1, 0] is π/4
- Applications: Geometry, physics, computer graphics

**Cross Products**
- Only defined in 3D
- Result is perpendicular to both vectors
- Magnitude gives area of parallelogram
- Applications: Torque, angular momentum

**2. Matrix Operations**
**Matrix Multiplication**
- Check dimensions match
- Use row-column method
- Properties: Associative, distributive
- Not commutative in general

**Finding Determinants**
- 2×2: ad - bc
- 3×3: Use cofactor expansion
- Properties: det(AB) = det(A)det(B)
- Applications: Inverses, areas, volumes

**Computing Inverses**
- Check if matrix is invertible
- Use adjugate method or row operations
- Verify AA⁻¹ = I
- Applications: Solving systems, transformations

**Gaussian Elimination**
- Convert to row echelon form
- Use back substitution
- Identify free variables
- Write parametric solution

**3. Linear Systems**
**Unique Solutions**
- System has exactly one solution
- No free variables
- Matrix is invertible
- Example: 2x + y = 5, 3x + 2y = 8

**Parametric Solutions**
- System has infinitely many solutions
- Free variables present
- Write solution in terms of parameters
- Example: x = 2 + t, y = 1 - 2t

**Consistency Checking**
- System has no solution
- Inconsistent equations
- Row of zeros with non-zero constant
- Example: x + y = 1, x + y = 2

**4. Eigenvalue Problems**
**Finding Eigenvalues**
- Solve characteristic equation
- det(A - λI) = 0
- Factor polynomial
- Find roots

**Finding Eigenvectors**
- For each eigenvalue λ
- Solve (A - λI)v = 0
- Find basis for null space
- Normalize if needed

**Diagonalization**
- Find eigenvalues and eigenvectors
- Construct P and D matrices
- Verify A = PDP⁻¹
- Applications: Powers of matrices

**5. Geometric Applications**
**Distance Calculations**
- Point to line
- Point to plane
- Line to line
- Use vector formulas

**Area and Volume**
- Triangle area using cross product
- Parallelogram area
- Parallelepiped volume
- Use determinants

**Intersections**
- Line-plane intersection
- Plane-plane intersection
- Line-line intersection
- Use parametric equations

**6. Proof Problems**
**Vector Space Properties**
- Verify closure under addition
- Verify closure under scalar multiplication
- Check all 10 axioms
- Use properties of real numbers

**Linear Independence**
- Set up equation c₁v₁ + ... + cₙvₙ = 0
- Show only solution is cᵢ = 0
- Use properties of vectors
- May need to use contradiction

**Subspace Verification**
- Check non-empty
- Verify closure under addition
- Verify closure under scalar multiplication
- May need to find basis

**Matrix Properties**
- Prove properties of operations
- Use definition of operations
- Apply properties of real numbers
- May need to use induction
        """
        st.markdown(patterns.replace("[bold]", "**").replace("[/bold]", "**"))

    elif learning_topic == "Recognize Problem Types":
        st.subheader("📝 Recognize Problem Types")
        problem_recognition_guide = """**How to Recognize Linear Algebra Problems**

This guide will help you identify the type of problem you're dealing with and choose the appropriate solution method.

**1. Vector Problems**
**Points, Directions, Magnitudes**
- Problem mentions coordinates or points
- Involves distances or lengths
- Uses terms like "direction", "magnitude", "unit vector"
- If you need to find angles, distances, or projections
- If the problem mentions perpendicularity or parallel lines
- If you need to compute areas or volumes

**2. Matrix Problems**
**Grid of Numbers**
- Problem shows a grid or array of numbers
- Involves rows and columns
- Uses terms like "matrix", "array", "table"
- If you need to perform operations on multiple equations
- If the problem involves transformations
- If you need to find determinants or inverses

**3. System of Equations**
**Multiple Variables**
- Problem has several variables
- Involves multiple equations
- Uses terms like "solve", "find", "determine"
- If you need to find values that satisfy all equations
- If the problem involves consistency or uniqueness
- If you need to find parametric solutions

**4. Eigenvalue Problems**
- If the problem involves repeated transformations
- If you need to find special vectors that don't change direction
- If the problem mentions stability or oscillations
- If you need to diagonalize a matrix

**5. Geometric Problems**
- If the problem involves points, lines, or planes
- If you need to find intersections or distances
- If the problem mentions areas or volumes
- If you need to find projections or shadows
        """
        st.markdown(problem_recognition_guide.replace("[bold]", "**").replace("[/bold]", "**"))

        st.markdown("---")
        st.subheader("Problem Recognition Quiz (Examples)")
        
        quiz_questions_static = [
            {
                "question": "Given vectors a = [1, 2, 3] and b = [4, 5, 6], find a·b",
                "type": "Vector",
                "explanation": "This is a vector problem because it involves the dot product of two vectors. The problem gives vectors in component form and asks for a vector operation (dot product)."
            },
            {
                "question": "Solve the system: 2x + y = 5, 3x + 2y = 8",
                "type": "System",
                "explanation": "This is a system of equations problem because it involves multiple equations with multiple variables. The problem asks to find values that satisfy all equations simultaneously."
            },
            {
                "question": "Find the eigenvalues of matrix A = [[1, 2], [3, 4]]",
                "type": "Eigenvalue",
                "explanation": "This is an eigenvalue problem because it involves finding special values that satisfy the characteristic equation. The problem gives a matrix and asks for an eigenvalue operation."
            },
            {
                "question": "Calculate the area of the triangle with vertices (0,0), (3,0), and (0,4)",
                "type": "Geometric",
                "explanation": "This is a geometric problem because it involves calculating the area of a shape defined by points. The problem mentions geometric objects (triangle) and measurements (area)."
            },
            {
                "question": "Find the inverse of matrix A = [[1, 2], [3, 4]]",
                "type": "Matrix",
                "explanation": "This is a matrix problem because it involves operations on a matrix to find its inverse. The problem gives a matrix and asks for a matrix operation (inverse)."
            }
        ]
        for i, q_static in enumerate(quiz_questions_static):
            with st.expander(f"Quiz Question {i+1}: {q_static['question']}", expanded=False):
                st.write(f"**Correct Answer Type:** {q_static['type']}")
                st.write(f"**Explanation:** {q_static['explanation']}") 