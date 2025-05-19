# Linear Algebra Exercise Framework

A comprehensive CLI tool for solving linear algebra problems and exercises.

## Features

- Vector operations (polar coordinates, normalization, shadows, cross products)
- Matrix operations (addition, scalar multiplication, matrix product)
- Linear equation systems (Gaussian elimination, pivots, free variables)
- Geometric calculations (triangles, distances, intersections of planes)
- Series calculations and symbolic math

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/linear-algebra-framework.git
cd linear-algebra-framework

# Install dependencies
pip install numpy sympy scipy
```

## Usage

```bash
python linalg_cli.py <command> [options]
```

Run with `--help` to see available commands:

```bash
python linalg_cli.py --help
```

## Examples

### Vector Operations

Convert polar coordinates to Cartesian:
```bash
python linalg_cli.py polar_to_cartesian --r 5 --phi 45 --degrees
```

Normalize a vector:
```bash
python linalg_cli.py normalize_vector --vector "3, 4"
```

Calculate the vector projection (shadow):
```bash
python linalg_cli.py vector_shadow --vector_a "3, -4" --vector_b "12, -1"
```

### Matrix Operations

Solve a system of linear equations:
```bash
python linalg_cli.py solve_gauss --matrix "1, -4, -2, -25; 0, -3, 6, -18; 7, -13, -4, -85"
```

Find intersections of planes:
```bash
python linalg_cli.py intersection_planes --matrix "2, 1, -2, -1; 1, 8, -4, 10; 6, -1, 18, 81"
```

Calculate matrix product:
```bash
python linalg_cli.py matrix_product --matrix_a "0, 4, 2; 2, -1, 5; 6, 0, -3" --matrix_b "4, 1, 9; 5, 4, 2; -3, 5, 2"
```

### Other Operations

Calculate the sum of a series:
```bash
python linalg_cli.py sum_series --start 0 --end 15 --formula "5+3*i"
```

Check if vectors are orthogonal:
```bash
python linalg_cli.py check_orthogonal --vector "1, 5, 2" --check_vectors "263, -35, -44" "71, 5, -48"
```

Calculate the distance from a point to a plane:
```bash
python linalg_cli.py point_plane_distance --normal "4, 0, -3" --plane_point "5, 1, -2" --point "10, 4, -12"
```

## Input Formats

- Vectors: `"x1, x2, x3"` or `"x1 x2 x3"`
- Matrices: `"a, b, c; d, e, f"` (semicolons separate rows, commas separate elements)

## License

MIT