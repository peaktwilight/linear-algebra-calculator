#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for the Linear Algebra Quiz
"""

import streamlit as st
import numpy as np # Used for checking answers, e.g. vector normalization

# Import self-sufficient utilities
from .st_math_utils import MathUtils
import random

class QuizComponent:
    def __init__(self):
        """Initialize the quiz component."""
        self.available_quiz_types = [
            "Vector Normalization",
            "Matrix Multiplication", 
            "Dot Product",
            "Matrix Determinant",
            "Linear Systems",
            "Matrix Algebra",
            "Summations",
            "Linear Mappings"
        ]
    
    def _generate_simple_quiz(self, quiz_type, difficulty, num_questions):
        """Generate simple quiz questions"""
        questions = []
        
        for i in range(num_questions):
            # For random quiz, pick a random question type
            current_quiz_type = quiz_type
            if quiz_type == "random":
                available_types = [t for t in self.available_quiz_types if t != "random"]
                current_quiz_type = random.choice(available_types)
            
            if current_quiz_type == "Vector Normalization":
                # Generate vector normalization question
                if difficulty == "easy":
                    vector = [random.randint(1, 5), random.randint(1, 5)]
                elif difficulty == "medium": 
                    vector = [random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)]
                else:
                    vector = [random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)]
                
                magnitude = sum(x**2 for x in vector)**0.5
                normalized = [x/magnitude for x in vector]
                
                questions.append({
                    "title": "Vector Normalization",
                    "question": f"Normalize the vector {vector}",
                    "type": "Vector Normalization",
                    "answer": normalized,
                    "explanation": f"Magnitude = √({' + '.join([f'{x}²' for x in vector])}) = {magnitude:.3f}\nNormalized = {vector} / {magnitude:.3f} = {[round(x, 3) for x in normalized]}"
                })
            
            elif current_quiz_type == "Dot Product":
                # Generate dot product question
                if difficulty == "easy":
                    v1 = [random.randint(1, 5), random.randint(1, 5)]
                    v2 = [random.randint(1, 5), random.randint(1, 5)]
                else:
                    v1 = [random.randint(-5, 5) for _ in range(3)]
                    v2 = [random.randint(-5, 5) for _ in range(3)]
                
                dot_product = sum(a*b for a, b in zip(v1, v2))
                
                questions.append({
                    "title": "Dot Product",
                    "question": f"Calculate the dot product of {v1} and {v2}",
                    "type": "Dot Product", 
                    "answer": dot_product,
                    "explanation": f"Dot product = {' + '.join([f'{a}×{b}' for a, b in zip(v1, v2)])} = {dot_product}"
                })
            
            elif current_quiz_type == "Matrix Multiplication":
                # Generate matrix multiplication question
                if difficulty == "easy":
                    A = [[random.randint(1, 3) for _ in range(2)] for _ in range(2)]
                    B = [[random.randint(1, 3) for _ in range(2)] for _ in range(2)]
                else:
                    rows_A, cols_A = (2, 3) if difficulty == "medium" else (3, 3)
                    rows_B, cols_B = (3, 2) if difficulty == "medium" else (3, 3)
                    A = [[random.randint(-2, 2) for _ in range(cols_A)] for _ in range(rows_A)]
                    B = [[random.randint(-2, 2) for _ in range(cols_B)] for _ in range(rows_B)]
                
                # Calculate result
                result = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
                
                questions.append({
                    "title": "Matrix Multiplication",
                    "question": f"Calculate A × B where A = {A} and B = {B}",
                    "type": "Matrix Multiplication",
                    "answer": result,
                    "explanation": f"Matrix multiplication: Result[i][j] = Σ(A[i][k] × B[k][j])"
                })
            
            elif current_quiz_type == "Matrix Determinant":
                # Generate determinant question
                if difficulty == "easy":
                    matrix = [[random.randint(1, 5), random.randint(1, 5)], 
                             [random.randint(1, 5), random.randint(1, 5)]]
                    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
                    explanation = f"det = {matrix[0][0]}×{matrix[1][1]} - {matrix[0][1]}×{matrix[1][0]} = {det}"
                else:
                    matrix = [[random.randint(-3, 3) for _ in range(3)] for _ in range(3)]
                    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
                    explanation = "Using cofactor expansion along first row"
                
                questions.append({
                    "title": "Matrix Determinant",
                    "question": f"Calculate the determinant of {matrix}",
                    "type": "Matrix Determinant",
                    "answer": det,
                    "explanation": explanation
                })
            
            elif current_quiz_type == "Linear Systems":
                # Generate linear system question
                if difficulty == "easy":
                    # 2x2 system
                    a, b, c = random.randint(1, 3), random.randint(1, 3), random.randint(1, 5)
                    d, e, f = random.randint(1, 3), random.randint(1, 3), random.randint(1, 5)
                    system = f"{a}x + {b}y = {c}\n{d}x + {e}y = {f}"
                    det = a*e - b*d
                    if det != 0:
                        x = (c*e - b*f) / det
                        y = (a*f - c*d) / det
                        answer = {"x": round(x, 3), "y": round(y, 3)}
                    else:
                        answer = "infinite" if (c*e == b*f and a*f == c*d) else "no solution"
                else:
                    # 3x3 system with parameters
                    system = "x₁ + x₂ + 2x₃ = 4\nx₂ - 4x₃ = 1\n2x₂ - 12x₃ = 3"
                    answer = "inconsistent"
                
                questions.append({
                    "title": "Linear System",
                    "question": f"Solve the system:\n{system}",
                    "type": "Linear Systems",
                    "answer": answer,
                    "explanation": "Use Gaussian elimination or Cramer's rule"
                })
            
            elif current_quiz_type == "Summations":
                # Generate summation question
                if difficulty == "easy":
                    # Geometric series
                    r = 1/5
                    n = 4
                    terms = [f"(1/5)^{i}" for i in range(n+1)]
                    result = sum(r**i for i in range(n+1))
                    
                    questions.append({
                        "title": "Geometric Series",
                        "question": f"Calculate: Σ(1/5)ⁱ for i=0 to {n}",
                        "type": "Summations",
                        "answer": round(result, 4),
                        "explanation": f"Geometric series: a(1-rⁿ⁺¹)/(1-r) = 1(1-(1/5)⁵)/(1-1/5) = {result:.4f}"
                    })
                else:
                    # Arithmetic series
                    n = random.randint(10, 25)
                    result = sum((i + 5)**2 for i in range(1, n+1)) - 350
                    
                    questions.append({
                        "title": "Arithmetic Series",
                        "question": f"Calculate: Σ(i+5)² for i=1 to {n}, then subtract 350",
                        "type": "Summations", 
                        "answer": result,
                        "explanation": f"Use formula for sum of squares: Σk² = n(n+1)(2n+1)/6"
                    })
            
            elif current_quiz_type == "Linear Mappings":
                # Generate linear mapping question
                if difficulty == "easy":
                    # Simple 2D transformation
                    mapping = "L(x,y) = (3y, 2y)"
                    is_linear = True
                    matrix = [[0, 3], [0, 2]]
                else:
                    # Non-linear mapping
                    mapping = "L(v) = (1 - v₂)"
                    is_linear = False
                    matrix = None
                
                questions.append({
                    "title": "Linear Mapping",
                    "question": f"Is the mapping {mapping} linear? If yes, provide its matrix representation.",
                    "type": "Linear Mappings",
                    "answer": {"is_linear": is_linear, "matrix": matrix},
                    "explanation": "Check additivity L(u+v)=L(u)+L(v) and homogeneity L(cu)=cL(u)"
                })
        
        return questions
        
    def render(self):
        """Render the quiz component in Streamlit."""
        st.title("Linear Algebra Quiz")
        st.write("Test your linear algebra knowledge with interactive questions!")
        
        # Quiz options
        with st.expander("Quiz Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                quiz_type = st.selectbox(
                    "Select Quiz Type",
                    ["random"] + self.available_quiz_types,
                    format_func=lambda x: "Random Mix" if x == "random" else x.replace("_", " ").title()
                )
                
                difficulty = st.select_slider(
                    "Difficulty Level",
                    options=["easy", "medium", "hard"],
                    value="medium"
                )
            
            with col2:
                num_questions = st.slider(
                    "Number of Questions",
                    min_value=1,
                    max_value=10,
                    value=3
                )
                
                st.markdown("### Quiz Instructions")
                st.markdown(
                    "Test your linear algebra knowledge with these interactive quizzes! "
                    "Answer the questions and check your solutions against the step-by-step explanations."
                )
        
        # Generate quiz button
        if st.button("Generate New Quiz", key="generate_quiz"):
            questions = self._generate_simple_quiz(quiz_type, difficulty, num_questions)
            
            # Store questions in session state
            st.session_state.quiz_questions = questions
            st.session_state.quiz_responses = [{"answered": False, "correct": None} for _ in range(len(questions))]
        
        # Display quiz questions if they exist in session state
        if "quiz_questions" in st.session_state and st.session_state.quiz_questions:
            self._display_quiz()
    
    def _display_quiz(self):
        """Display the quiz questions stored in session state."""
        questions = st.session_state.quiz_questions
        responses = st.session_state.quiz_responses
        
        # Quiz progress
        progress = sum(1 for r in responses if r["answered"]) / len(questions)
        st.progress(progress, text=f"Completed {int(progress * 100)}% of the quiz")
        
        # Score if all questions are answered
        if all(r["answered"] for r in responses):
            correct_count = sum(1 for r in responses if r["correct"])
            st.success(f"Quiz completed! Your score: {correct_count}/{len(questions)}")
        
        # Display each question
        for i, question in enumerate(questions):
            with st.expander(f"Question {i+1}: {question['title']}", expanded=not responses[i]["answered"]):                
                # Question content
                st.markdown(f"**{question['question']}**")
                
                # Custom input based on question type
                user_answer = None # Initialize user_answer
                if question["type"] == "Dot Product":
                    user_answer = st.number_input("Enter the dot product:", key=f"q{i}_dot")
                
                elif question["type"] == "Matrix Determinant":
                    user_answer = st.number_input("Enter the determinant:", key=f"q{i}_det", format="%.3f")
                
                elif question["type"] == "Summations":
                    user_answer = st.number_input("Enter the sum:", key=f"q{i}_sum", format="%.4f")
                
                elif question["type"] == "Linear Systems":
                    if isinstance(question["answer"], dict):
                        col1, col2 = st.columns(2)
                        with col1:
                            x_val = st.number_input("x =", key=f"q{i}_x", format="%.3f")
                        with col2:
                            y_val = st.number_input("y =", key=f"q{i}_y", format="%.3f")
                        user_answer = {"x": x_val, "y": y_val}
                    else:
                        solution_type = st.selectbox(
                            "Solution type:",
                            ["unique", "infinite", "no solution", "inconsistent"],
                            key=f"q{i}_solution_type"
                        )
                        user_answer = solution_type
                
                elif question["type"] == "Linear Mappings":
                    linear_check = st.radio("Is the mapping linear?", ["Yes", "No"], key=f"q{i}_linear")
                    is_linear = linear_check == "Yes"
                    
                    matrix_input = None
                    if is_linear:
                        st.write("Enter the matrix representation:")
                        if "2x2" in str(question["answer"].get("matrix", [])):
                            cols = st.columns(2)
                            row1 = []
                            row2 = []
                            for j in range(2):
                                with cols[j]:
                                    row1.append(st.number_input(f"a1{j+1}:", key=f"q{i}_m1{j}"))
                                    row2.append(st.number_input(f"a2{j+1}:", key=f"q{i}_m2{j}"))
                            matrix_input = [row1, row2]
                    
                    user_answer = {"is_linear": is_linear, "matrix": matrix_input}
                
                elif question["type"] == "Matrix Multiplication":
                    st.write("Enter the result matrix:")
                    result_shape = len(question["answer"]), len(question["answer"][0])
                    matrix_result = []
                    
                    for row_idx in range(result_shape[0]):
                        cols = st.columns(result_shape[1])
                        row = []
                        for col_idx in range(result_shape[1]):
                            with cols[col_idx]:
                                val = st.number_input(f"[{row_idx+1}][{col_idx+1}]:", key=f"q{i}_r{row_idx}{col_idx}")
                                row.append(val)
                        matrix_result.append(row)
                    user_answer = matrix_result
                
                elif question["type"] == "polar_to_cartesian":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_response = st.number_input("x coordinate:", key=f"q{i}_x", format="%.4f")
                    with col2:
                        y_response = st.number_input("y coordinate:", key=f"q{i}_y", format="%.4f")
                    user_answer = {"x": x_response, "y": y_response}
                
                elif question["type"] == "vector_normalization":
                    st.markdown("Enter the normalized vector components:")
                    dims = len(question["parameters"]["vector"])
                    cols = st.columns(dims)
                    user_answer_list = [] # Changed name to avoid conflict
                    for j in range(dims):
                        with cols[j]:
                            val = st.number_input(f"Component {j+1}:", key=f"q{i}_comp{j}", format="%.6f")
                            user_answer_list.append(val)
                    user_answer = user_answer_list # Assign to user_answer
                
                elif question["type"] == "orthogonal_vectors":
                    radio_answer = st.radio( # Renamed to avoid conflict
                        "Are the vectors orthogonal?",
                        ["Yes", "No"],
                        key=f"q{i}_orthogonal"
                    )
                    user_answer = radio_answer == "Yes"
                
                # Add more input types for other question types as needed
                
                # Default fallback for other question types (if no specific input is defined)
                supported_types = [
                    "Dot Product", "Matrix Determinant", "Summations", "Linear Systems", 
                    "Linear Mappings", "Matrix Multiplication", "Vector Normalization",
                    "polar_to_cartesian", "vector_normalization", "orthogonal_vectors"
                ]
                
                if user_answer is None and question["type"] not in supported_types:
                    st.write("This question type has no interactive input in the UI yet. You can view the solution.")

                # Check answer button
                check_col, reset_col = st.columns([3, 1])
                with check_col:
                    # Disable check if no answer can be provided by UI yet
                    disable_check = user_answer is None and question["type"] not in supported_types
                    check_answer_button = st.button("Check Answer", key=f"check_q{i}", disabled=disable_check)
                with reset_col:
                    reset_answer = st.button("Reset", key=f"reset_q{i}")
                
                # Process answer check
                if check_answer_button and user_answer is not None:
                    correct = self._check_answer(question, user_answer)
                    if correct:
                        st.success("Correct! 🎉")
                    else:
                        st.error("Not quite right. Try again or see the solution.")
                    
                    responses[i]["answered"] = True
                    responses[i]["correct"] = correct
                
                # Reset answer
                if reset_answer:
                    responses[i]["answered"] = False
                    responses[i]["correct"] = None
                
                # Solution section
                if responses[i]["answered"] or st.checkbox("Show Solution", key=f"show_solution_{i}"):
                    st.markdown("### Solution")
                    if "solution_steps" in question:
                        for step in question["solution_steps"]:
                            st.markdown(step)
                    elif "explanation" in question:
                        st.markdown(question["explanation"])
                    
                    st.markdown(f"**Answer:** {question['answer']}")
    
    def _check_answer(self, question, user_answer):
        """Check if the user's answer is correct."""
        answer_type = question["type"]
        correct_answer = question["answer"]
        
        if answer_type == "Dot Product":
            return abs(user_answer - correct_answer) < 0.001
        
        elif answer_type == "Matrix Determinant":
            return abs(user_answer - correct_answer) < 0.001
        
        elif answer_type == "Summations":
            return abs(user_answer - correct_answer) < 0.001
        
        elif answer_type == "Matrix Multiplication":
            if not isinstance(user_answer, list) or not isinstance(correct_answer, list):
                return False
            if len(user_answer) != len(correct_answer):
                return False
            for i in range(len(user_answer)):
                if len(user_answer[i]) != len(correct_answer[i]):
                    return False
                for j in range(len(user_answer[i])):
                    if abs(user_answer[i][j] - correct_answer[i][j]) > 0.001:
                        return False
            return True
        
        elif answer_type == "Linear Systems":
            if isinstance(correct_answer, dict) and isinstance(user_answer, dict):
                x_correct = abs(user_answer["x"] - correct_answer["x"]) < 0.01
                y_correct = abs(user_answer["y"] - correct_answer["y"]) < 0.01
                return x_correct and y_correct
            else:
                return user_answer == correct_answer
        
        elif answer_type == "Linear Mappings":
            if isinstance(correct_answer, dict) and isinstance(user_answer, dict):
                linear_correct = user_answer["is_linear"] == correct_answer["is_linear"]
                if not linear_correct:
                    return False
                
                if correct_answer["is_linear"] and correct_answer["matrix"] is not None:
                    if user_answer["matrix"] is None:
                        return False
                    # Check matrix equality
                    for i in range(len(correct_answer["matrix"])):
                        for j in range(len(correct_answer["matrix"][i])):
                            if abs(user_answer["matrix"][i][j] - correct_answer["matrix"][i][j]) > 0.01:
                                return False
                return True
            return False
        
        elif answer_type == "Vector Normalization":
            # Check if normalized vector components are within acceptable tolerance
            if isinstance(user_answer, list) and isinstance(correct_answer, list):
                if len(user_answer) == len(correct_answer):
                    return all(abs(u - c) < 0.01 for u, c in zip(user_answer, correct_answer))
            return False
        
        elif answer_type == "polar_to_cartesian":
            # Check if x and y coordinates are within acceptable tolerance
            x_correct = abs(user_answer["x"] - correct_answer["x"]) < 0.01
            y_correct = abs(user_answer["y"] - correct_answer["y"]) < 0.01
            return x_correct and y_correct
        
        elif answer_type == "vector_normalization":
            # Check if normalized vector components are within acceptable tolerance
            # Ensure user_answer is a list/array of numbers
            if isinstance(user_answer, (list, np.ndarray)) and isinstance(correct_answer["normalized_vector"], (list, np.ndarray)):
                 if len(user_answer) == len(correct_answer["normalized_vector"]):
                    return all(abs(u - c) < 0.01 for u, c in zip(user_answer, correct_answer["normalized_vector"]))
            return False # If types or lengths don't match
        
        elif answer_type == "orthogonal_vectors":
            # Check if the orthogonality assessment is correct
            return user_answer == correct_answer["are_orthogonal"]
        
        # Default fallback
        st.warning(f"Answer checking for quiz type '{answer_type}' is not fully implemented in the UI yet.")
        return False 