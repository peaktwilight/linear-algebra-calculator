import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def display_vector_visualization(vectors, names=None, origin=None, construction_lines=None):
    """Create a visualization for vectors, optionally with construction lines."""
    if names is None:
        names = [f"Vector {i+1}" for i in range(len(vectors))]
    
    # Determine dimension
    dim = 0
    if vectors and len(vectors[0]) > 0:
        dim = len(vectors[0])
    elif construction_lines and construction_lines[0][0] is not None and len(construction_lines[0][0]) > 0:
        dim = len(construction_lines[0][0])
    elif construction_lines and construction_lines[0][1] is not None and len(construction_lines[0][1]) > 0:
        dim = len(construction_lines[0][1])
    else:
        st.warning("Cannot determine dimension for visualization.")
        return

    if not vectors: # If primary vectors list is empty, but we have lines
        st.info("Visualizing construction lines only.")
    else: # Validate primary vectors
        for vec in vectors:
            if len(vec) != dim:
                st.warning(f"Cannot visualize. Primary vectors have mixed dimensions or don't match construction lines: expected {dim} vs {len(vec)}")
                return
    
    if origin is None:
        origin = np.zeros(dim)
    
    fig = go.Figure()
    max_val = 1.0 # Start with a default to avoid issues with all-zero vectors/lines
        
    if dim == 2:
        # Add primary vectors as arrows
        if vectors:
            for i, vec in enumerate(vectors):
                current_max_coord = max(abs(v_comp) for v_comp in vec[:2]) if len(vec) >= 2 else 0
                max_val = max(max_val, current_max_coord)
                fig.add_trace(go.Scatter(
                    x=[origin[0], origin[0] + vec[0]],
                    y=[origin[1], origin[1] + vec[1]],
                    mode='lines+markers',
                    name=names[i] if names and i < len(names) else f"Vector {i+1}",
                    line=dict(width=3),
                    marker=dict(size=[0, 10])
                ))
        
        # Add construction lines if any
        if construction_lines:
            for p1, p2, name, style_dict in construction_lines:
                p1_2d = p1[:2] if len(p1) >= 2 else np.array([0.0, 0.0])
                p2_2d = p2[:2] if len(p2) >= 2 else np.array([0.0, 0.0])
                current_max_coord = max(abs(p1_2d[0]), abs(p1_2d[1]), abs(p2_2d[0]), abs(p2_2d[1]))
                max_val = max(max_val, current_max_coord)
                fig.add_trace(go.Scatter(
                    x=[p1_2d[0], p2_2d[0]],
                    y=[p1_2d[1], p2_2d[1]],
                    mode='lines',
                    name=name,
                    line=style_dict
                ))
        
        fig.update_layout(
            xaxis=dict(range=[-max_val * 1.2, max_val * 1.2], zeroline=True, zerolinewidth=2, zerolinecolor='white', showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)', color='white'),
            yaxis=dict(range=[-max_val * 1.2, max_val * 1.2], zeroline=True, zerolinewidth=2, zerolinecolor='white', showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)', color='white'),
            title="Vector Visualization", title_font_color="white", showlegend=True, width=600, height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.1)', legend=dict(font=dict(color="white")),
        )
        st.plotly_chart(fig)
    
    elif dim == 3:
        if vectors:
            for i, vec in enumerate(vectors):
                current_max_coord = max(abs(v_comp) for v_comp in vec[:3]) if len(vec) >= 3 else 0
                max_val = max(max_val, current_max_coord)
                fig.add_trace(go.Scatter3d(
                    x=[origin[0], origin[0] + vec[0]],
                    y=[origin[1], origin[1] + vec[1]],
                    z=[origin[2], origin[2] + vec[2]],
                    mode='lines+markers',
                    name=names[i] if names and i < len(names) else f"Vector {i+1}",
                    line=dict(width=6),
                    marker=dict(size=[0, 8])
                ))
        
        if construction_lines:
            for p1, p2, name, style_dict in construction_lines:
                p1_3d = p1[:3] if len(p1) >= 3 else np.array([0.0, 0.0, 0.0])
                p2_3d = p2[:3] if len(p2) >= 3 else np.array([0.0, 0.0, 0.0])
                current_max_coord = max(abs(p1_3d[0]), abs(p1_3d[1]), abs(p1_3d[2]), abs(p2_3d[0]), abs(p2_3d[1]), abs(p2_3d[2]))
                max_val = max(max_val, current_max_coord)
                fig.add_trace(go.Scatter3d(
                    x=[p1_3d[0], p2_3d[0]],
                    y=[p1_3d[1], p2_3d[1]],
                    z=[p1_3d[2], p2_3d[2]],
                    mode='lines',
                    name=name,
                    line=style_dict
                ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-max_val * 1.2, max_val * 1.2], title="X", color="white", gridcolor='rgba(255, 255, 255, 0.2)', backgroundcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(range=[-max_val * 1.2, max_val * 1.2], title="Y", color="white", gridcolor='rgba(255, 255, 255, 0.2)', backgroundcolor='rgba(0,0,0,0.1)'),
                zaxis=dict(range=[-max_val * 1.2, max_val * 1.2], title="Z", color="white", gridcolor='rgba(255, 255, 255, 0.2)', backgroundcolor='rgba(0,0,0,0.1)'),
                aspectmode='cube'),
            title="3D Vector Visualization", title_font_color="white", width=700, height=700, paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color="white")),
        )
        st.plotly_chart(fig)
    else:
        st.info(f"Cannot visualize {dim}-dimensional vectors directly. Using tabular representation instead.")
        if vectors: # Only show table if there are primary vectors
            vector_df = pd.DataFrame(vectors, index=names if names else [f"Vector {i+1}" for i in range(len(vectors))])
            st.table(vector_df)

def display_matrix_heatmap(matrix, title="Matrix Visualization", center_scale=False, color_scale='Viridis'):
    """
    Create a heatmap visualization for matrices.
    
    Parameters:
    -----------
    matrix : array-like
        The matrix to visualize
    title : str
        The title for the visualization
    center_scale : bool
        Whether to center the color scale at zero. Useful for matrices with both positive and negative values.
    color_scale : str
        The colorscale to use. Options include 'Viridis', 'RdBu', 'Greys', etc.
    """
    matrix = np.array(matrix)
    
    # Set up color scale options
    imshow_args = {
        'labels': dict(x="Column", y="Row", color="Value"),
        'x': [f"Col {i+1}" for i in range(matrix.shape[1])],
        'y': [f"Row {i+1}" for i in range(matrix.shape[0])],
        'text_auto': True,
        'color_continuous_scale': color_scale,
        'title': title
    }
    
    # Center the color scale at zero if requested
    if center_scale:
        max_abs = np.max(np.abs(matrix))
        imshow_args['zmin'] = -max_abs
        imshow_args['zmax'] = max_abs
        if color_scale == 'Viridis':  # Use a diverging color scale for centered data
            imshow_args['color_continuous_scale'] = 'RdBu_r'
    
    # Create a heatmap using plotly
    fig = px.imshow(matrix, **imshow_args)
    
    # Update layout for dark theme compatibility
    fig.update_layout(
        width=800, 
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        title_font_color="white",
        font=dict(color="white"),
        xaxis=dict(color="white", gridcolor='rgba(255, 255, 255, 0.2)'),
        yaxis=dict(color="white", gridcolor='rgba(255, 255, 255, 0.2)'),
    )
    st.plotly_chart(fig) 