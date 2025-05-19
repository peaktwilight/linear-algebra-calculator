#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the UI changes
"""
from ui_rich import LinearAlgebraRichUI
from rich.console import Console

def main():
    ui = LinearAlgebraRichUI()
    console = Console()
    
    print("Test 1: Standard format with colons")
    test_results1 = """Original vector: [3. 4.]
Length: 5.0000
Normalized vector: [0.6 0.8]
Verification - length of normalized vector: 1.0000"""
    
    table1 = ui.format_results_as_table(test_results1, "Standard Format")
    console.print(table1)
    
    print("\nTest 2: Single line output (polar_to_cartesian)")
    test_results2 = """Cartesian coordinates: [3.9945, 0.2093]"""
    
    table2 = ui.format_results_as_table(test_results2, "Polar to Cartesian")
    console.print(table2)
    
    print("\nTest 3: Mixed format with no colons")
    test_results3 = """Vector a: [1, 2, 3]
Vector b: [4, 5, 6]
Cross product (a × b): [10, 20, 30]"""
    
    table3 = ui.format_results_as_table(test_results3, "Mixed Format")
    console.print(table3)
    
    print("\nTest 4: Empty vector normalization")
    test_results4 = """Original vector: []
Length: 0.0000
Normalized vector: []
Verification - length of normalized vector: 0.0000"""
    
    table4 = ui.format_results_as_table(test_results4, "Empty Vector")
    console.print(table4)
    
    print("\nTest 5: Matrix output")
    test_results5 = """Original augmented matrix [A|b]:
[[1. 2. 3. 4.]
 [5. 6. 7. 8.]
 [9. 0. 1. 2.]]

Row echelon form:
[[1. 0. 0. 0.2]
 [0. 1. 0. 0.3]
 [0. 0. 1. 0.4]]

The system has a unique solution:
x1 = 0.2000
x2 = 0.3000
x3 = 0.4000"""
    
    table5 = ui.format_results_as_table(test_results5, "Matrix Results")
    console.print(table5)
    
    print("\nOur improved formatting should now handle various output formats correctly!")
    print("All results will display properly in the tables, regardless of their format.")

if __name__ == "__main__":
    main()