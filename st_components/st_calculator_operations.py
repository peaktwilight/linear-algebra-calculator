#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Component for Linear Algebra Calculator Operations
"""

import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Not directly used in LinAlgCalculator, but good to keep if visualizations expand
import plotly.express as px
import plotly.graph_objects as go
from given_reference.core import mrref # mnull, eliminate are used by linalg_cli, not directly here
from .st_visualization_utils import display_vector_visualization, display_matrix_heatmap

# Import CLI functionality to reuse functions
from linalg_cli import LinearAlgebraExerciseFramework

# Import Utilities
from .st_utils import StreamOutput

# Import Mixins
from .st_vector_operations_mixin import VectorOperationsMixin
from .st_matrix_operations_mixin import MatrixOperationsMixin

class LinAlgCalculator(VectorOperationsMixin, MatrixOperationsMixin):
    def __init__(self):
        super().__init__() # Initialize mixins if they have their own __init__ (optional)
        self.framework = LinearAlgebraExerciseFramework()
    
    # All vector and matrix methods are now inherited from mixins.
    # The main calculator class can be kept lean, or include 
    # additional high-level orchestration logic if needed in the future.