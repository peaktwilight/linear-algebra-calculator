# Doruk's Linear Algebra Calculator [![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/peaktwilight/python_25fs/releases/tag/v1.1.0)

```
╭───────────────────────────────────────────────────────────────────────╮
 ___               _   _      _    _                   
|   \ ___ _ _ _  _| |_( )___ | |  (_)_ _  ___ __ _ _ _ 
| |) / _ \ '_| || | / //(_-< | |__| | ' \/ -_) _` | '_|
|___/\___/_|  \_,_|_\_\ /__/ |____|_|_||_\___\__,_|_|  
                                                       
   _   _          _                ___      _         _      _           
  /_\ | |__ _ ___| |__ _ _ __ _   / __|__ _| |__ _  _| |__ _| |_ ___ _ _ 
 / _ \| / _` / -_) '_ \ '_/ _` | | (__/ _` | / _| || | / _` |  _/ _ \ '_|
/_/ \_\_\__, \___|_.__/_| \__,_|  \___\__,_|_\__|\_,_|_\__,_|\__\___/_|  
        |___/
╰───────────────────────────────────────────────────────────────────────╯
```

A comprehensive toolkit for learning and solving linear algebra problems, developed for the LAG Fachmodul at FHNW.

## Features

- Interactive UI with detailed explanations and guidance
- Vector operations (normalization, projections, angles, cross products)
- Matrix operations (addition, multiplication, transformation)
- Linear equation systems (Gaussian elimination, parametric solutions)
- Geometric calculations (distances, plane intersections)
- Enhanced search functionality with interactive filtering and clear results organization
- Comprehensive help system for learning linear algebra concepts
- Example exercises with step-by-step solutions
- Problem recognition quiz to improve problem-solving skills

## File Structure

- `linalg.py` - Main entry point script with unified interface and ASCII banner
- `linalg_cli.py` - Core linear algebra functionality with command-line interface
- `linalg_ui.py` - Rich user interface with ASCII banner (directly runnable)
- `help_guides.py` - Help content and learning resources for the framework
- `given_reference/` - Directory with reference code including core functionality

## Installation

```bash
# Clone the repository
git clone https://github.com/peaktwilight/python_25fs.git
cd python_25fs

# Install dependencies
pip install numpy sympy scipy rich questionary
```

> **Note**: The `core.py` module is located in the `given_reference` directory and imported from there.

## Usage

The framework offers multiple interfaces through the unified `linalg.py` launcher:

### Unified Launcher (Recommended)

```bash
# Interactive UI mode (default)
python linalg.py

# CLI mode
python linalg.py --cli

# Direct command execution
python linalg.py --command "solve_gauss --matrix '1, 2, 3; 4, 5, 6'"

# Skip the ASCII banner
python linalg.py --no-banner
```

### Interactive UI

```bash
# Using the unified launcher
python linalg.py

# Or running the UI module directly
python linalg_ui.py

# Optional UI flags
python linalg_ui.py --no-banner   # Skip the ASCII banner
python linalg_ui.py --no-color    # Disable colored output
```

Features:
- Interactive menus with explanations
- Advanced search functionality with real-time filtering and organized results
- Comprehensive help system
- Example exercises with solutions
- Problem recognition guidance
- Quizzes to test understanding

### Command Line Interface

```bash
# Using the unified launcher
python linalg.py --cli <command> [options]

# Or using the dedicated CLI
python linalg_cli.py <command> [options]
```

For help on available commands:
```bash
python linalg.py --cli --help
# Or
python linalg_cli.py --help
```

## Learning Resources

The Interactive UI includes a "Learning Resources" section with:

1. **Browse help topics** - Detailed explanations of linear algebra concepts
2. **Example exercises** - Practice problems with step-by-step solutions
3. **Linear algebra guide** - Structured learning path for all concepts
4. **Common homework patterns** - Strategies for typical homework problems
5. **Problem recognition guide** - Learn to identify problem types

## Examples

### Search Functionality

Search for specific concepts or operations:
```
Select: Search
Enter search term: projection
```

Filter search results interactively:
```
Select: Search
Enter search term: matrix
Filter by category: [Use arrow keys to select categories]
```

### Vector Operations

Normalize a vector:
```
Select category: Vector Operations
Select operation: Normalize a vector
Enter vector: 3, 4
```

Calculate vector projection:
```
Select category: Vector Operations
Select operation: Calculate vector shadow (projection)
Enter vector to project onto: 3, -4
Enter vector to be projected: 12, -1
```

### Matrix Operations

Solve a system of linear equations:
```
Select category: Matrix Operations
Select operation: Solve system with Gaussian elimination
Enter augmented matrix: 1, -4, -2, -25; 0, -3, 6, -18; 7, -13, -4, -85
```

Calculate matrix product:
```
Select category: Matrix Operations
Select operation: Calculate matrix product
Enter first matrix: 0, 4, 2; 2, -1, 5; 6, 0, -3
Enter second matrix: 4, 1, 9; 5, 4, 2; -3, 5, 2
```

## For Students

The framework is designed to help you learn linear algebra concepts while solving exercises:

1. **If you're new to linear algebra**: Start with the "Learning Resources" section in the Interactive UI to understand the fundamentals.

2. **If you're struggling with homework**: Use the "Common homework patterns" guide to identify solution strategies for your specific problem type.

3. **If you need to understand concepts**: Each operation includes detailed explanations and formula derivations.

4. **If you want to practice**: Try the example exercises and problem recognition quiz to test your understanding.

## About the Author

This calculator was created by Doruk as a personal project for the Linear Algebra (LAG) module at the University of Applied Sciences and Arts Northwestern Switzerland (FHNW). It was designed to help students understand and solve complex linear algebra problems through an intuitive interface with detailed explanations.

## License

MIT

---

© 2025 Doruk | FHNW Linear Algebra Module