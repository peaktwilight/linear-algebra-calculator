# Doruk's Linear Algebra Calculator [![Version](https://img.shields.io/badge/version-1.8.4-blue.svg)](https://github.com/peaktwilight/linear-algebra-calculator/releases/tag/v1.8.4)
![Web App Demo](public/linear-algebra-calculator.gif)

## Features

-   🔢 **Comprehensive Calculators:** Vectors, matrices, systems of linear equations, and series/summations.
-   🧮 **Batch Matrix Operations:** Automatically evaluate multiple matrix expressions from exercise lists.
-   🔍 **Linear Mapping Analysis:** Test if mappings are linear and generate matrix representations.
-   📊 **Series & Pattern Recognition:** Analyze sequences, calculate geometric/arithmetic series, and solve summation exercises.
-   🎨 **Interactive Visualizations:** Understand concepts with dynamic graphical representations and LaTeX formatting.
-   📚 **Step-by-Step Explanations:** Detailed walkthroughs with mathematical formulas and solution steps.
-   🧠 **Practice & Learn:** Interactive quizzes and comprehensive learning resources.
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
- Summation Exercise Solver with step-by-step solutions

### 🔍 **Linear Mappings**
- Linearity Testing for Various Function Types
- Matrix Representation Generation

## File Structure

-   `streamlit_app.py`: Main web application entry point
-   `st_components/`: Streamlit interface components
    -   `st_math_utils.py`: Core mathematical utilities and functions
    -   `st_calculator_operations.py`: Main calculator operations
    -   `st_batch_calculator.py`: Batch matrix expression evaluator
    -   `st_summation_calculator.py`: Series and summation calculator
    -   `st_quiz_ui.py`: Interactive quiz interface
    -   `st_learning_resources.py`: Educational content and tutorials
    -   Various operation mixins for specialized functionality
-   `given_reference/`: Core algorithms and reference implementations
-   `requirements.txt`: Python dependencies
-   `CLAUDE.md`: Development notes and configuration

## Online Version

A hosted version of the web interface is available at:
**[lag.doruk.ch](https://lag.doruk.ch)**

---

© 2025 Doruk | FHNW Linear Algebra Module