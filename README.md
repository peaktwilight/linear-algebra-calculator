# Doruk's Linear Algebra Calculator [![Version](https://img.shields.io/badge/version-1.9.5-blue.svg)](https://github.com/peaktwilight/linear-algebra-calculator/releases/tag/v1.9.5)
![Web App Demo](public/linear-algebra-calculator.gif)

## Features

-   🔢 **Comprehensive Calculators:** Vectors, matrices, systems of linear equations, and series/summations.
-   📐 **3D Geometry Operations:** Planes, lines, intersections, and distance calculations in 3D space.
-   🧮 **Batch Matrix Operations:** Automatically evaluate multiple matrix expressions from calculation lists.
-   🔍 **Linear Mapping Analysis:** Test if mappings are linear and generate matrix representations.
-   📊 **Series & Pattern Recognition:** Analyze sequences, calculate geometric/arithmetic series, and solve summation problems.
-   🎨 **Interactive Visualizations:** Understand concepts with dynamic graphical representations and LaTeX formatting.
-   📚 **Step-by-Step Explanations:** Detailed walkthroughs with mathematical formulas and solution steps.
-   🧠 **Practice & Learn:** Comprehensive learning resources.
-   🌐 **Modern Web Interface:** Beautiful, responsive design with search functionality and organized categories.

A comprehensive toolkit for learning and solving linear algebra problems, built as a modern web application with Streamlit.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/peaktwilight/linear-algebra-calculator.git
cd linear-algebra-calculator

# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run streamlit_app.py
```

## Available Operations

### 🔢 **Vector Operations**
- Vector Normalization, Projection, Angle Calculation
- Cross Product, Triangle Area, Point-Line Distance
- Collinearity Checking

### 📐 **Lines and Planes**
- **Plane from 3 Points** - Calculate plane equation from three non-collinear points
- **Point-to-Plane Distance** - Find shortest distance from point to plane
- **Line-Plane Intersection** - Calculate where lines intersect planes in 3D
- **Plane Equation Converter** - Convert between coordinate, vector, and parametric forms

### 📊 **Matrix Operations**
- Matrix Addition, Subtraction, Multiplication, Transpose
- Determinant, Inverse, Eigenvalues & Eigenvectors
- **Batch Expression Calculator** - Automatically solve multiple matrix expressions

### ⚖️ **Systems of Linear Equations**
- Gaussian Elimination, Row Operations Analysis
- Null Space Calculation, Free Parameter Analysis
- Vector Solution Checking, Geometric Interpretation

### 📈 **Series & Summations**
- Geometric and Arithmetic Series Calculation
- Pattern Recognition in Sequences
- Summation Problem Solver with step-by-step solutions

### 🔍 **Linear Mappings**
- Linearity Testing for Various Function Types
- Matrix Representation Generation

## Architecture

This is a **Streamlit web application** (v1.9.5) for linear algebra calculations. The architecture is modular and self-sufficient:

**Entry Point:**
- `streamlit_app.py` - Main application with routing between calculator modes

**Core Components (`st_components/`):**
- `st_math_utils.py` - Self-sufficient mathematical utilities (vectors, matrices, equations)
- `st_calculator_operations.py` - Main calculator operations class
- Operation mixins:
  - `st_vector_operations_mixin.py` - Vector calculations
  - `st_matrix_operations_mixin.py` - Matrix operations
  - `st_plane_operations_mixin.py` - 3D geometry (planes, lines, intersections)
- Specialized calculators:
  - `st_batch_calculator.py` - Batch matrix expression evaluator
  - `st_summation_calculator.py` - Series and summation calculator
  - `st_linearity_operations.py` - Linear mapping analysis
- UI components:
  - `st_learning_resources.py` - Educational content
  - `st_visualization_utils.py` - Plotting and visualizations

**Key Design Principles:**
- All mathematical operations use numpy, sympy, and scipy directly
- No external API dependencies
- Organized as mixins for maintainability
- Rich visualizations using plotly and matplotlib
- LaTeX rendering for mathematical expressions

## File Structure

-   `streamlit_app.py`: Main web application entry point
-   `st_components/`: Streamlit interface components
-   `given_reference/`: Core algorithms and reference implementations
-   `requirements.txt`: Python dependencies

## Online Version

A hosted version of the web interface is available at:
**[lag.doruk.ch](https://lag.doruk.ch)**

---

© 2025 Doruk | FHNW Linear Algebra Module