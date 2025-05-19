#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Help Guides and Explanations for Linear Algebra Framework
"""

from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Dictionary of common linear algebra questions with explanations and solutions
COMMON_QUESTIONS = {
    "polar_to_cartesian": {
        "title": "Converting Polar to Cartesian Coordinates",
        "common_questions": [
            "How do I convert polar coordinates to rectangular coordinates?",
            "What are the x,y coordinates for a point given radius and angle?",
            "Converting from r,θ to x,y coordinates"
        ],
        "explanation": """
In polar coordinates, a point is described by:
- A radius r (distance from origin)
- An angle θ (angle from positive x-axis)

The conversion formulas are:
- x = r·cos(θ)
- y = r·sin(θ)

This is useful for problems involving circular motion, rotation, or when positions are given in terms of distance and direction.

**Example Problem:**
*Convert the polar coordinates (r=5, θ=60°) to Cartesian coordinates.*

Solution:
- x = 5·cos(60°) = 5·0.5 = 2.5
- y = 5·sin(60°) = 5·0.866 = 4.33

So the point is at (2.5, 4.33) in Cartesian coordinates.
        """
    },
    
    "vector_basics": {
        "title": "Understanding Vectors",
        "common_questions": [
            "What is a vector?",
            "How are vectors different from regular numbers?",
            "How do I represent vectors in mathematics?"
        ],
        "explanation": """
A vector is a mathematical object with both magnitude (size) and direction. Unlike regular numbers (scalars) that just have magnitude, vectors can represent quantities like:
- Displacement
- Velocity
- Force
- Acceleration

In 2D space, vectors have two components (x,y). In 3D space, they have three components (x,y,z).

**Ways to denote vectors:**
- By components: (3, 4) or [3, 4]
- Using unit vectors: 3i + 4j
- As a magnitude and direction: 5 units at 53° from horizontal

The length (magnitude) of a vector v = (x, y) is calculated as |v| = √(x² + y²).

When you need to solve vector problems, focus on the components rather than trying to visualize the entire vector.
        """
    },
    
    "vector_operations": {
        "title": "Vector Operations",
        "common_questions": [
            "How do I add or subtract vectors?",
            "What is scalar multiplication of vectors?",
            "How do I calculate the dot product?"
        ],
        "explanation": """
**Vector Addition:**
Add corresponding components: (a₁, a₂) + (b₁, b₂) = (a₁+b₁, a₂+b₂)
Example: (3, 4) + (2, -1) = (5, 3)

**Vector Subtraction:**
Subtract corresponding components: (a₁, a₂) - (b₁, b₂) = (a₁-b₁, a₂-b₂)
Example: (3, 4) - (2, -1) = (1, 5)

**Scalar Multiplication:**
Multiply each component by the scalar: k·(a₁, a₂) = (k·a₁, k·a₂)
Example: 3·(2, 4) = (6, 12)

**Dot Product (inner product):**
a·b = a₁·b₁ + a₂·b₂ + a₃·b₃
Example: (1, 2, 3)·(4, 5, 6) = 1·4 + 2·5 + 3·6 = 32

The dot product is used to find:
- Angle between vectors: cos(θ) = (a·b)/(|a|·|b|)
- Projection of one vector onto another
- Work done by a force

**Cross Product (only in 3D):**
a×b = (a₂·b₃-a₃·b₂, a₃·b₁-a₁·b₃, a₁·b₂-a₂·b₁)
The cross product gives a vector perpendicular to both a and b.
        """
    },
    
    "vector_applications": {
        "title": "Applications of Vectors",
        "common_questions": [
            "When do I use the dot product versus the cross product?",
            "How do I find the angle between two vectors?",
            "How do I find if vectors are orthogonal/perpendicular?"
        ],
        "explanation": """
**Use dot product when you need:**
- Angle between vectors: θ = arccos((a·b)/(|a|·|b|))
- Projection of one vector onto another
- To check if vectors are orthogonal (perpendicular): a·b = 0
- To calculate work: W = F·d

**Use cross product when you need:**
- A vector perpendicular to two given vectors
- Area of a parallelogram: Area = |a×b|
- Torque calculation: τ = r×F
- To determine if vectors are parallel: a×b = 0

**Finding the angle between vectors:**
1. Calculate the dot product: a·b
2. Calculate magnitudes: |a| and |b|
3. Use formula: θ = arccos((a·b)/(|a|·|b|))

**Testing for orthogonality (perpendicular vectors):**
Vectors are orthogonal if their dot product equals zero.
Example: Are (1, 2, 3) and (3, -1, -1) orthogonal?
(1, 2, 3)·(3, -1, -1) = 3 - 2 - 3 = -2 ≠ 0, so they're not orthogonal.
        """
    },
    
    "vector_normalization": {
        "title": "Vector Normalization",
        "common_questions": [
            "How do I find a unit vector in a given direction?",
            "What is vector normalization?",
            "How do I convert a vector to a unit vector?"
        ],
        "explanation": """
**Normalizing a vector** means converting it to a unit vector (length 1) pointing in the same direction.

The formula is: unit_vector = v / |v| where |v| is the magnitude/length of v.

**Steps to normalize:**
1. Calculate the magnitude: |v| = √(x² + y² + z²)
2. Divide each component by magnitude: unit_v = (x/|v|, y/|v|, z/|v|)

**Example:**
Normalize v = (3, 4)
1. |v| = √(3² + 4²) = √25 = 5
2. unit_v = (3/5, 4/5) = (0.6, 0.8)

**Applications:**
- Direction vectors (when you only care about direction, not magnitude)
- Simplifying calculations
- Creating orthonormal bases
- Computer graphics (normal vectors)

**Verification:**
You can verify normalization by calculating the magnitude of the unit vector - it should equal 1.
        """
    },
    
    "vector_projection": {
        "title": "Vector Projection (Shadow)",
        "common_questions": [
            "How do I calculate the projection of one vector onto another?",
            "What is a vector shadow?",
            "How do I find the component of a vector in a specific direction?"
        ],
        "explanation": """
The **projection of vector b onto vector a** (also called the shadow of b on a) represents how much of b is in the direction of a.

There are two types of projection:
1. **Scalar projection** - the length of the shadow
2. **Vector projection** - the actual shadow vector

**Formulas:**
- Scalar projection: comp_a(b) = (a·b)/|a|
- Vector projection: proj_a(b) = ((a·b)/(a·a)) * a

**When to use projection:**
- Decomposing vectors into components
- Finding work done by a force (physics)
- Calculating distance from a point to a line
- Breaking forces into perpendicular components

**Example:**
Project b = (3, 2) onto a = (4, 0)
- a·b = 4·3 + 0·2 = 12
- |a| = 4
- Scalar projection = 12/4 = 3
- Vector projection = (12/16)·(4, 0) = 0.75·(4, 0) = (3, 0)

This means b extends 3 units in the direction of a.
        """
    },
    
    "linear_systems": {
        "title": "Systems of Linear Equations",
        "common_questions": [
            "How do I solve multiple equations with multiple unknowns?",
            "What is a system of linear equations?",
            "How do I use matrices to solve linear systems?"
        ],
        "explanation": """
A **system of linear equations** is a collection of two or more linear equations with the same variables.

General form:
a₁x + b₁y + c₁z = d₁
a₂x + b₂y + c₂z = d₂
a₃x + b₃y + c₃z = d₃

**Matrix representation:**
[a₁ b₁ c₁ | d₁]
[a₂ b₂ c₂ | d₂]
[a₃ b₃ c₃ | d₃]

**Solving methods:**
1. **Gaussian elimination** - systematically eliminate variables
2. **Matrix row reduction** - transform to reduced row echelon form
3. **Cramer's rule** - using determinants (for smaller systems)

**Possible outcomes:**
- Unique solution - exactly one solution
- Infinitely many solutions - typically expressed with parameters
- No solution - contradictory equations (inconsistent system)

**Example:**
For the system:
x + 2y = 5
3x - y = 2

Step 1: Convert to augmented matrix:
[1  2 | 5]
[3 -1 | 2]

Step 2: Eliminate x from second row by subtracting 3 times the first row:
[1  2 | 5]
[0 -7 | -13]

Step 3: Solve for y: -7y = -13, so y = 13/7

Step 4: Substitute to find x: x + 2(13/7) = 5, so x = 5 - 26/7 = 9/7

Solution: x = 9/7, y = 13/7
        """
    },
    
    "matrices": {
        "title": "Understanding Matrices",
        "common_questions": [
            "What is a matrix?",
            "How do I perform matrix operations?",
            "What are the applications of matrices?"
        ],
        "explanation": """
A **matrix** is a rectangular array of numbers arranged in rows and columns. Matrices are used to represent systems of linear equations, transformations, and data structures.

**Matrix notation:**
A = [a₁₁ a₁₂ ... a₁ₙ]
    [a₂₁ a₂₂ ... a₂ₙ]
    [... ... ... ...]
    [aₘ₁ aₘ₂ ... aₘₙ]

The size of a matrix is described as m×n (m rows, n columns).

**Basic operations:**
- **Addition:** Add corresponding elements: (A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ
- **Scalar multiplication:** Multiply each element by scalar: (kA)ᵢⱼ = k · Aᵢⱼ
- **Matrix multiplication:** (AB)ᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ
- **Transpose:** Swap rows and columns: (Aᵀ)ᵢⱼ = Aⱼᵢ

**Special matrices:**
- **Identity matrix** (I): Ones on diagonal, zeros elsewhere
- **Zero matrix** (O): All elements are zero
- **Diagonal matrix:** Non-zero elements only on diagonal
- **Symmetric matrix:** A = Aᵀ

**Applications:**
- Solving systems of linear equations
- Representing transformations (rotation, scaling, reflection)
- Data analysis and statistics
- Computer graphics and simulations
- Quantum mechanics
        """
    },
    
    "matrix_operations": {
        "title": "Matrix Operations",
        "common_questions": [
            "How do I multiply matrices?",
            "When is matrix multiplication possible?",
            "How do I calculate the transpose of a matrix?"
        ],
        "explanation": """
**Matrix Addition (A + B):**
- Matrices must have the same dimensions
- Add corresponding elements
- Example: [1 2] + [5 6] = [6 8]
           [3 4]   [7 8]   [10 12]

**Matrix Subtraction (A - B):**
- Matrices must have the same dimensions
- Subtract corresponding elements
- Example: [5 6] - [1 2] = [4 4]
           [7 8]   [3 4]   [4 4]

**Scalar Multiplication (kA):**
- Multiply each element by scalar k
- Example: 2·[1 2] = [2 4]
             [3 4]   [6 8]

**Matrix Multiplication (A·B):**
- Only possible when A's columns = B's rows
- Result has dimensions: A's rows × B's columns
- Calculate each element by dot product of row and column

Example: [1 2] · [5 6] = [(1·5 + 2·7) (1·6 + 2·8)] = [19 22]
         [3 4]   [7 8]   [(3·5 + 4·7) (3·6 + 4·8)]   [43 50]

**Transpose (Aᵀ):**
- Swap rows and columns
- Example: [1 2 3]ᵀ = [1 4]
           [4 5 6]     [2 5]
                       [3 6]

**Important properties:**
- A + B = B + A (addition is commutative)
- A·B ≠ B·A in general (multiplication is NOT commutative)
- (A·B)·C = A·(B·C) (multiplication is associative)
- (A + B)·C = A·C + B·C (distributive property)
        """
    },
    
    "gauss_elimination": {
        "title": "Gaussian Elimination",
        "common_questions": [
            "How do I solve systems of equations with matrices?",
            "What is row echelon form?",
            "How do I apply Gaussian elimination step by step?"
        ],
        "explanation": """
**Gaussian elimination** is a systematic method for solving systems of linear equations by transforming the augmented matrix into row echelon form.

**The process:**
1. Write the system as an augmented matrix
2. Apply elementary row operations to get row echelon form:
   - Swap rows (if necessary)
   - Multiply a row by a non-zero scalar
   - Add a multiple of one row to another
3. Use back-substitution to find the values of variables

**Example:**
Solve the system:
x + 2y + 3z = 14
2x + y + z = 8
3x + y + z = 11

Step 1: Convert to augmented matrix
[1 2 3 | 14]
[2 1 1 | 8]
[3 1 1 | 11]

Step 2: Eliminate x from row 2
[1 2 3 | 14]
[0 -3 -5 | -20]
[3 1 1 | 11]

Step 3: Eliminate x from row 3
[1 2 3 | 14]
[0 -3 -5 | -20]
[0 -5 -8 | -31]

Step 4: Eliminate y from row 3
[1 2 3 | 14]
[0 -3 -5 | -20]
[0 0 -1/3 | -1]

Step 5: Back-substitution
z = 3
-3y - 5(3) = -20 → y = 5/3
x + 2(5/3) + 3(3) = 14 → x = 2

Solution: x = 2, y = 5/3, z = 3
        """
    },
    
    "planes_intersection": {
        "title": "Plane Intersections in 3D Space",
        "common_questions": [
            "How do I find where planes intersect?",
            "What are the possible intersections of two or three planes?",
            "How do I determine if planes are parallel?"
        ],
        "explanation": """
Planes in 3D space are represented by equations of the form ax + by + cz + d = 0.

**Possible intersections of planes:**

1. **Two planes:**
   - Line: The most common case
   - Coincident: The planes are identical
   - Parallel but distinct: No intersection

2. **Three planes:**
   - Single point: The most common case
   - Line: Two planes are parallel to a third
   - Empty/no intersection: Parallel or inconsistent constraints
   - Plane: All three planes are identical

**Finding intersections:**
1. Express each plane equation as ax + by + cz + d = 0
2. Form the augmented matrix with coefficients
3. Use Gaussian elimination
4. Analyze the result:
   - Unique solution: Planes intersect at a point
   - One parameter: Planes intersect along a line
   - Two parameters: Planes coincide
   - Inconsistent: No intersection

**Example:**
Find the intersection of:
2x + y - 2z = -1
x + 8y - 4z = 10
6x - y + 18z = 81

Augmented matrix:
[2  1  -2  | -1]
[1  8  -4  | 10]
[6  -1  18 | 81]

After Gaussian elimination, we get a single point of intersection.
        """
    },
    
    "linear_independence": {
        "title": "Linear Independence and Dependence",
        "common_questions": [
            "How do I check if vectors are linearly independent?",
            "What does linear dependence mean?",
            "How are rank and linear independence related?"
        ],
        "explanation": """
**Linear independence** means no vector in the set can be expressed as a linear combination of the others.

Vectors v₁, v₂, ..., vₙ are linearly independent if the equation:
c₁v₁ + c₂v₂ + ... + cₙvₙ = 0

has only the trivial solution c₁ = c₂ = ... = cₙ = 0.

If there are other solutions, the vectors are **linearly dependent**.

**Checking for linear independence:**
1. Form a matrix with the vectors as columns or rows
2. Calculate the rank of the matrix
3. If rank equals the number of vectors, they are linearly independent

**Geometric interpretation:**
- 2 vectors: Independent if not parallel (not on same line)
- 3 vectors: Independent if not coplanar (not on same plane)
- n vectors: Independent if they span an n-dimensional space

**Common cases:**
- More vectors than dimensions: Always linearly dependent
- Vectors including the zero vector: Always linearly dependent
- Vectors with duplicates: Always linearly dependent

**Example:**
Check if v₁ = (1, 0, 1), v₂ = (0, 1, 0), v₃ = (2, 1, 2) are linearly independent.

Form the matrix:
[1 0 2]
[0 1 1]
[1 0 2]

After row reduction:
[1 0 2]
[0 1 1]
[0 0 0]

The rank is 2 < 3 vectors, so they are linearly dependent.
Indeed, v₃ = 2v₁ + v₂.
        """
    },
    
    "parametric_solutions": {
        "title": "Parametric Solutions and Free Variables",
        "common_questions": [
            "What are free and pivot variables?",
            "How do I express solutions with parameters?",
            "How do I find the general solution to a system?"
        ],
        "explanation": """
When a system of linear equations has infinitely many solutions, we express them using **parametric form** with free variables.

**Key concepts:**
- **Pivot variables**: Variables corresponding to leading positions in row echelon form
- **Free variables**: Variables that don't have a leading 1, can be assigned any value
- **Parametric form**: Expressing pivot variables in terms of free variables

**Steps to find parametric solutions:**
1. Solve the system using Gaussian elimination
2. Identify pivot and free variables
3. Express each pivot variable in terms of free variables
4. Write the general solution with parameters (often t, s, etc.)

**Example:**
Solve the system:
x + 2y - z = 3
2x + 4y - 2z = 6

Step 1: Row reduce the augmented matrix
[1 2 -1 | 3]
[2 4 -2 | 6]

After elimination:
[1 2 -1 | 3]
[0 0 0 | 0]

Step 2: Identify variables
- Pivot: x
- Free: y, z

Step 3: Express pivots in terms of free variables
From row 1: x = 3 - 2y + z

Step 4: Write general solution
x = 3 - 2y + z
y = y (free)
z = z (free)

Or using parameters s and t:
x = 3 - 2s + t
y = s
z = t

This represents a plane in 3D space.
        """
    },
    
    "practical_tips": {
        "title": "Practical Tips for Solving Linear Algebra Problems",
        "common_questions": [
            "How do I approach complex linear algebra problems?",
            "What are common mistakes to avoid?",
            "How can I check my answers?"
        ],
        "explanation": """
**General Approach:**
1. **Identify the problem type** first (vector operation, linear system, etc.)
2. **Convert word problems** to mathematical notation
3. **Set up the appropriate matrices** or vectors
4. **Apply the right technique** (elimination, projection, etc.)
5. **Check your answer** by substituting back

**Common Mistakes to Avoid:**
- **Matrix multiplication errors**: Verify dimensions match before multiplying
- **Row operation errors**: Keep track of all changes during elimination
- **Sign errors**: Be careful with negative numbers
- **Forgetting conditions**: Check when operations are valid
- **Misidentifying problem types**: Make sure you're using the right approach

**Checking Your Answers:**
- For linear systems: Substitute solutions back into original equations
- For vector problems: Verify properties (length, angles, etc.)
- For matrices: Check if AB = C by multiplying A and B
- For elimination: Verify that your row operations are valid

**Simplifying Complex Problems:**
- Break problems into smaller steps
- Draw diagrams for geometric interpretation
- When stuck, try a different approach
- Use the framework's tools to handle tedious calculations

**Memory Aids:**
- Dot product → scalar output
- Cross product → vector output
- Matrix is rows × columns
- For multiplication (A×B), columns(A) must equal rows(B)
        """
    },
    
    "geometric_interpretation": {
        "title": "Geometric Interpretation of Linear Algebra",
        "common_questions": [
            "What does a matrix transformation look like visually?",
            "How can I visualize vectors and their operations?",
            "What is the geometric meaning of determinants and eigenvalues?"
        ],
        "explanation": """
**Vectors as Arrows:**
- Vectors represent directed line segments with magnitude and direction
- Addition: Place tail of second vector at head of first (parallelogram rule)
- Scalar multiplication: Stretches or shrinks vector length

**Dot Product:**
- a·b = |a|·|b|·cos(θ)
- Positive: Vectors point in similar directions (acute angle)
- Zero: Vectors are perpendicular
- Negative: Vectors point in opposite directions (obtuse angle)

**Cross Product:**
- a×b produces a vector perpendicular to both a and b
- |a×b| equals the area of the parallelogram formed by a and b
- Direction follows right-hand rule

**Matrix as Transformation:**
- 2×2 matrices represent transformations of the 2D plane
- Each column shows where the basis vectors (1,0) and (0,1) go
- Rotation, scaling, shearing, reflection are all matrix transformations

**Linear Systems:**
- Each equation represents a line (2D) or plane (3D)
- Solutions are the intersection points of these lines/planes
- No solutions: Lines are parallel
- Infinite solutions: Lines coincide

**Determinants:**
- The determinant of a 2×2 matrix is the area scaling factor of its transformation
- |A| = 0 means the transformation collapses space to a lower dimension
- |A| < 0 means the transformation includes a reflection

**Eigenvalues and Eigenvectors:**
- Eigenvectors: Directions that only get stretched/shrunk (not rotated)
- Eigenvalues: The factor by which eigenvectors are stretched/shrunk
        """
    }
}

# Function to display help on a specific topic
def display_topic_help(console, topic):
    """Display help on a specific linear algebra topic"""
    if topic in COMMON_QUESTIONS:
        help_data = COMMON_QUESTIONS[topic]
        
        # Create a panel for the main explanation
        explanation_panel = Panel(
            Markdown(help_data["explanation"]),
            title=f"[bold cyan]{help_data['title']}[/bold cyan]",
            border_style="blue",
            expand=False
        )
        
        # Create a table for common questions
        questions_table = Table(title="Common Questions", box=None)
        questions_table.add_column("Questions", style="yellow")
        
        for question in help_data["common_questions"]:
            questions_table.add_row(f"• {question}")
        
        # Display
        console.print(explanation_panel)
        console.print(questions_table)
        
        return True
    else:
        console.print(f"[bold red]Topic '{topic}' not found in help system.[/bold red]")
        return False

# Function to list all available help topics
def list_help_topics(console):
    """List all available help topics"""
    topics_table = Table(title="Available Help Topics", box=None)
    topics_table.add_column("Topic", style="cyan")
    topics_table.add_column("Description", style="green")
    
    for topic, data in COMMON_QUESTIONS.items():
        topics_table.add_row(topic, data["title"])
    
    console.print(topics_table)

# Map of operation names to help topics
OPERATION_TO_HELP = {
    "polar_to_cartesian": "polar_to_cartesian",
    "normalize_vector": "vector_normalization",
    "vector_direction_length": "vector_basics",
    "vector_shadow": "vector_projection",
    "check_orthogonal": "vector_applications",
    "vector_angle": "vector_applications",
    "cross_product": "vector_operations",
    "triangle_area": "vector_applications",
    "point_line_distance": "vector_applications",
    "check_collinear": "linear_independence",
    "solve_gauss": "gauss_elimination",
    "check_coplanar": "linear_independence",
    "check_solution": "linear_systems",
    "vector_equation": "vector_operations",
    "matrix_element": "matrices",
    "intersection_planes": "planes_intersection",
    "homogeneous_intersection": "planes_intersection",
    "find_pivot_free_vars": "parametric_solutions",
    "matrix_operations": "matrix_operations",
    "matrix_product": "matrix_operations",
    "sum_series": "practical_tips",
    "check_particular_solution": "linear_systems",
    "point_plane_distance": "vector_applications"
}

# Exercise templates with step-by-step solutions
EXERCISE_TEMPLATES = {
    "polar_to_cartesian": {
        "template": "Convert polar coordinates (r={radius}, θ={angle}{unit}) to Cartesian coordinates.",
        "solution": """
Step 1: Identify the formulas for conversion:
   x = r·cos(θ)
   y = r·sin(θ)

Step 2: Convert angle to radians if given in degrees:
   {conversion}

Step 3: Calculate x-coordinate:
   x = {radius} · cos({angle_rad}) = {x_result}

Step 4: Calculate y-coordinate:
   y = {radius} · sin({angle_rad}) = {y_result}

Therefore, the Cartesian coordinates are ({x_result}, {y_result}).
        """
    },
    
    "normalize_vector": {
        "template": "Normalize the vector v = {vector}.",
        "solution": """
Step 1: Calculate the magnitude of the vector:
   |v| = √({components_squared}) = {magnitude}

Step 2: Divide each component by the magnitude:
   v̂ = {vector} / {magnitude} = {result}

Step 3: Verify that the result is a unit vector:
   |v̂| = √({result_components_squared}) = {result_magnitude}

Therefore, the normalized vector is {result}.
        """
    },
    
    "check_orthogonal": {
        "template": "Check if the vectors u = {vector1} and v = {vector2} are orthogonal.",
        "solution": """
Step 1: Calculate the dot product of the vectors:
   u·v = {dot_product_calculation} = {dot_product}

Step 2: Determine if the vectors are orthogonal:
   Vectors are orthogonal if and only if their dot product equals zero.
   {dot_product} {is_equal_zero}

Therefore, the vectors are {orthogonal_result}.
        """
    },
    
    "vector_angle": {
        "template": "Find the angle between vectors u = {vector1} and v = {vector2}.",
        "solution": """
Step 1: Calculate the dot product of the vectors:
   u·v = {dot_product_calculation} = {dot_product}

Step 2: Calculate the magnitudes of the vectors:
   |u| = √({u_squared_sum}) = {u_magnitude}
   |v| = √({v_squared_sum}) = {v_magnitude}

Step 3: Use the formula for the angle between two vectors:
   cos(θ) = (u·v) / (|u|·|v|) = {dot_product} / ({u_magnitude} · {v_magnitude}) = {cos_theta}

Step 4: Calculate the angle in radians:
   θ = arccos({cos_theta}) = {angle_rad} radians

Step 5: Convert to degrees (if needed):
   θ = {angle_rad} · (180°/π) = {angle_deg}°

Therefore, the angle between the vectors is {angle_rad} radians or {angle_deg} degrees.
        """
    },
    
    "solve_linear_system": {
        "template": "Solve the system of linear equations:\n{equations}",
        "solution": """
Step 1: Write the augmented matrix:
{augmented_matrix}

Step 2: Apply Gaussian elimination to reach row echelon form:
{row_operations}

Step 3: Identify pivot and free variables:
   Pivot variables: {pivot_vars}
   Free variables: {free_vars}

Step 4: Express the solution:
{solution_expression}

Therefore, the solution is {final_solution}.
        """
    },
    
    "matrix_product": {
        "template": "Calculate the product of matrices A = {matrix1} and B = {matrix2}.",
        "solution": """
Step 1: Check if multiplication is possible:
   Matrix A is {m1}×{n1} and matrix B is {m2}×{n2}.
   For multiplication to be possible, the number of columns in A ({n1}) must equal 
   the number of rows in B ({m2}).
   {multiplication_possible}

Step 2: Calculate each entry of the result C = A×B:
{calculation_steps}

Therefore, the product A×B = {result}.
        """
    }
}

# Function to generate exercise from template
def generate_exercise(exercise_type, params=None):
    """Generate an exercise from a template with given parameters"""
    if exercise_type not in EXERCISE_TEMPLATES:
        return "Exercise type not found."
    
    template = EXERCISE_TEMPLATES[exercise_type]
    
    if params is None:
        # Generate default parameters if none provided
        if exercise_type == "polar_to_cartesian":
            params = {
                "radius": 5,
                "angle": 45,
                "unit": "°",
                "conversion": "θ = 45° × (π/180) = 0.7854 radians",
                "angle_rad": 0.7854,
                "x_result": 3.536,
                "y_result": 3.536
            }
        elif exercise_type == "normalize_vector":
            params = {
                "vector": "(3, 4)",
                "components_squared": "3² + 4²",
                "magnitude": 5,
                "result": "(0.6, 0.8)",
                "result_components_squared": "0.6² + 0.8²",
                "result_magnitude": 1
            }
        elif exercise_type == "check_orthogonal":
            params = {
                "vector1": "(1, 2, 3)",
                "vector2": "(4, -2, 0)",
                "dot_product_calculation": "1×4 + 2×(-2) + 3×0",
                "dot_product": 0,
                "is_equal_zero": "= 0",
                "orthogonal_result": "orthogonal"
            }
    
    # Format the exercise and solution
    exercise = template["template"].format(**params)
    solution = template["solution"].format(**params)
    
    return {
        "exercise": exercise,
        "solution": solution
    }