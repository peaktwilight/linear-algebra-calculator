#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for the Linear Algebra Quiz
"""

import streamlit as st
import numpy as np # Used for checking answers, e.g. vector normalization
# import pandas as pd # Not used in QuizComponent
# import matplotlib.pyplot as plt # Not used in QuizComponent
# import plotly.express as px # Not used in QuizComponent
# import plotly.graph_objects as go # Not used in QuizComponent

# Import Quiz generator
from linear_algebra_quiz import LinearAlgebraQuiz

class QuizComponent:
    def __init__(self):
        """Initialize the quiz component."""
        self.quiz_generator = LinearAlgebraQuiz()
        self.available_quiz_types = self.quiz_generator.get_quiz_types()
        
    def render(self):
        """Render the quiz component in Streamlit."""
        st.title("Linear Algebra Quiz")
        
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
            if quiz_type == "random":
                questions = self.quiz_generator.generate_random_quiz(
                    count=num_questions,
                    difficulty=difficulty
                )
            else:
                questions = self.quiz_generator.generate_quiz(
                    quiz_type=quiz_type,
                    difficulty=difficulty,
                    count=num_questions
                )
            
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
                if question["type"] == "polar_to_cartesian":
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
                # This case should ideally be handled or avoided by ensuring all quiz types have UI
                if user_answer is None and not question["type"] in ["polar_to_cartesian", "vector_normalization", "orthogonal_vectors"]:
                    st.write("This question type has no interactive input in the UI yet. You can view the solution.")

                # Check answer button
                check_col, reset_col = st.columns([3, 1])
                with check_col:
                    # Disable check if no answer can be provided by UI yet
                    disable_check = user_answer is None and not question["type"] in ["polar_to_cartesian", "vector_normalization", "orthogonal_vectors"]
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
                    for step in question["solution_steps"]:
                        st.markdown(step)
    
    def _check_answer(self, question, user_answer):
        """Check if the user's answer is correct."""
        answer_type = question["type"]
        correct_answer = question["answer"]
        
        if answer_type == "polar_to_cartesian":
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
        
        # Add more checks for other question types as needed
        
        # Default fallback
        st.warning(f"Answer checking for quiz type '{answer_type}' is not fully implemented in the UI yet.")
        return False 