# Doruk's Linear Algebra Calculator [![Version](https://img.shields.io/badge/version-1.7.0-blue.svg)](https://github.com/peaktwilight/linear-algebra-calculator/releases/tag/v1.7.0)
![Web App Demo](public/linear-algebra-calculator.gif)

## Features

-   🔢 **Calculators:** Vectors, matrices, and many more linear algebra operations.
-   🔍 **Linear Mapping Analysis:** Automatically test if mappings are linear and generate matrix representations.
-   🎨 **Interactive Visualizations:** Understand complex concepts intuitively with dynamic graphical representations in a user-friendly Web App.
-   📚 **Step-by-Step Explanations:** Understand the "how" and "why" with detailed walkthroughs for solutions.
-   🧠 **Practice & Learn:** Test yourself with engaging quizzes and explore comprehensive learning resources.
-   💻 **Multiple Interfaces:** Choose your preferred way to work:
    -   🌐 **Sleek Web App:** A rich, graphical experience with interactive visualizations.
    -   ⌨️ **Efficient CLI:** For quick calculations and scripting power.
    -   💡 **Intuitive TUI:** A user-friendly, terminal-based interactive mode.

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