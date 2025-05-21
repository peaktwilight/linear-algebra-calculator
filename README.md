# Doruk's Linear Algebra Calculator [![Version](https://img.shields.io/badge/version-1.6.5-blue.svg)](https://github.com/peaktwilight/linear-algebra-calculator/releases/tag/v1.6.5)

## Features
-   Calculators for vectors, matrices and more
-   Interactive visualizations for intuitive understanding
-   Step-by-step solution explanations
-   Practice quizzes and learning resources
-   Includes a CLI, TUI and a Web App

## CLI Demo
![CLI Demo](public/Doruks_Algebra_Calculator_CLI.gif)

A comprehensive toolkit for learning and solving linear algebra problems. This project offers three main interfaces:
-   **CLI (Command-Line Interface)**: For direct command execution and scripting.
-   **TUI (Text-based User Interface)**: An interactive, terminal-based experience (default).
-   **Web App**: A graphical interface with interactive visualizations, hosted at [lag.doruk.ch](https://lag.doruk.ch).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/peaktwilight/linear-algebra-calculator.git
cd linear-algebra-calculator

# Install dependencies based on your needs (see Installation Options below)
pip install -r requirements.txt

# NOW YOU HAVE 2 OPTIONS

# 1) Run the interactive UI (recommended for first-time users)
python linalg.py
# OR
# 2) run the web interface
streamlit run streamlit_app.py
```

## For advanced users, there's also a CLI
```bash
# Show available commands
python linalg.py --cli --help

# Example: Normalize a vector
python linalg.py --cli --command "normalize_vector --vector '3,4'"

# Example: Solve a system of equations
python linalg.py --cli --command "solve_gauss --matrix '1,2,3; 4,5,6; 7,8,9'"
```

## File Structure

-   `linalg.py`: Main entry point (recommended starting point)
-   `linalg_cli.py`: Command-line interface implementation
-   `linalg_ui.py`: Rich terminal UI implementation
-   `streamlit_app.py`: Web interface implementation
-   `st_components/`: Streamlit interface components
-   `given_reference/`: Reference code and core algorithms
-   `help_guides.py`: Learning resources and tutorials

## Online Version

A hosted version of the web interface is available at:
**[lag.doruk.ch](https://lag.doruk.ch)**

---

© 2025 Doruk | FHNW Linear Algebra Module