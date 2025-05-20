# Doruk's Linear Algebra Calculator [![Version](https://img.shields.io/badge/version-1.6-blue.svg)](https://github.com/peaktwilight/linear-algebra-calculator/releases/tag/v1.6)

![CLI Demo](public/Doruks_Algebra_Calculator_CLI.gif)

A comprehensive toolkit for learning and solving linear algebra problems. This project offers three main interfaces:
-   **TUI (Text-based User Interface)**: An interactive, terminal-based experience (default).
-   **CLI (Command-Line Interface)**: For direct command execution and scripting.
-   **Web App**: A graphical interface with interactive visualizations, hosted at [lag.doruk.ch](https://lag.doruk.ch).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/peaktwilight/linear-algebra-calculator.git
cd linear-algebra-calculator

# Install all dependencies
pip install -r requirements.txt

# Run the interactive UI (recommended for first-time users)
python linalg.py

# OR run the web interface
streamlit run streamlit_app.py
```

## Choose Your Interface

### 1. Interactive Terminal UI (Default)
Best for learning with guided step-by-step solutions.
```bash
python linalg.py
```

### 2. Command-Line Interface
Best for quick calculations or scripting.
```bash
# Show available commands
python linalg.py --cli --help

# Example: Normalize a vector
python linalg.py --cli --command "normalize_vector --vector '3,4'"

# Example: Solve a system of equations
python linalg.py --cli --command "solve_gauss --matrix '1,2,3; 4,5,6; 7,8,9'"
```

### 3. Web Interface (Streamlit)
Best for visualizations and interactive learning.
```bash
streamlit run streamlit_app.py
# Then open http://localhost:8501 in your browser
```

## Features

-   Core linear algebra operations:
    -   **Vectors**: Normalization, projections, angles, cross products
    -   **Matrices**: Addition, multiplication, Gaussian elimination
    -   **Geometric**: Distances, plane intersections, triangle areas
-   Interactive visualizations
-   Step-by-step solution explanations
-   Practice quizzes and learning resources

## Installation Options

Choose one of these options based on your needs:

### Full Installation (All Interfaces)
```bash
pip install -r requirements.txt
```

### Minimal CLI Version Only
```bash
pip install numpy sympy scipy
```

### Terminal UI Version
```bash
pip install numpy sympy scipy rich questionary
```

### Web UI Version
```bash
pip install numpy sympy scipy streamlit pandas plotly matplotlib
```

## File Structure

-   `linalg.py`: Main entry point (recommended starting point)
-   `linalg_cli.py`: Command-line interface implementation
-   `linalg_ui.py`: Rich terminal UI implementation
-   `streamlit_app.py`: Web interface implementation
-   `st_components/`: Streamlit interface components
-   `given_reference/`: Reference code and core algorithms
-   `help_guides.py`: Learning resources and tutorials

## Docker Support

For quick testing of the Streamlit application:
```bash
./run_docker.sh  # Access at http://localhost:8501
```

## Online Version

A hosted version of the web interface is available at:
**[lag.doruk.ch](https://lag.doruk.ch)**

---

© 2025 Doruk | FHNW Linear Algebra Module
