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

class LinAlgCalculator:
    def __init__(self):
        self.framework = LinearAlgebraExerciseFramework()
    
    # All vector methods previously here have been moved to VectorOperationsMixin