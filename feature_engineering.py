"""
Feature Engineering and Data Visualization Module
Comprehensive analysis and visualization of ML pipeline
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline:
    """ML Feature Engineering and Visualization Pipeline"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
    def get_categorical_statistics(self):
        """
        Calculate categorical statistics: frequency and class imbalance
        
        Returns:
            Dictionary with categorical statistics
        """
        stats_dict = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            total = len(self.df)
            
            stats_dict[col] = {
                'frequencies': value_counts,
                'percentages': (value_counts / total * 100).round(2),
                'total_unique': len(value_counts),
                'imbalance_ratio': value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1
            }
        
        return stats_dict
    
    def get_numeric_statistics(self):
        """
        Get descriptive statistics for numeric attributes
        
        Returns:
            DataFrame with comprehensive statistics
        """
        stats_df = self.df[self.numeric_cols].describe().T
        
        # Add additional statistics
        for col in self.numeric_cols:
            stats_df.loc[col, 'skewness'] = self.df[col].skew()
            stats_df.loc[col, 'kurtosis'] = self.df[col].kurtosis()
            stats_df.loc[col, 'variance'] = self.df[col].var()
        
        return stats_df
    
    def plot_categorical_statistics(self):
        """
        Plot frequency and class imbalance for categorical variables
        """
        fig = make_subplots(
            rows=len(self.categorical_cols),
            cols=2,
            subplot_titles=[f"{col} - Frequency" for col in self.categorical_cols] +
                           [f"{col} - Percentage" for col in self.categorical_cols],
            specs=[[{"type": "bar"}, {"type": "pie"}] for _ in self.categorical_cols]
        )
        
        for idx, col in enumerate(self.categorical_cols[:5]):  # Limit to first 5
            value_counts = self.df[col].value_counts()
            
            # Bar chart for frequency
            fig.add_trace(
                go.Bar(x=value_counts.index.astype(str), y=value_counts.values,
                       name=col, showlegend=False),
                row=idx+1, col=1
            )
            
            # Pie chart for percentage
            fig.add_trace(
                go.Pie(labels=value_counts.index.astype(str), 
                       values=value_counts.values,
                       name=col, showlegend=False),
                row=idx+1, col=2
            )
        
        fig.update_layout(height=300*min(len(self.categorical_cols), 5), 
                         title_text="Estadísticas Categóricas - Frecuencia y Desequilibrio de Clases",
                         showlegend=False)
        
        return fig
    
    def plot_attribute_statistics(self):
        """
        Plot statistics of individual numeric attributes
        """
        fig = make_subplots(
            rows=len(self.numeric_cols),
            cols=2,
            subplot_titles=[f"{col} - Distribución" for col in self.numeric_cols] +
                           [f"{col} - Box Plot" for col in self.numeric_cols],
            specs=[[{"type": "histogram"}, {"type": "box"}] for _ in self.numeric_cols]
        )
        
        for idx, col in enumerate(self.numeric_cols[:8]):  # Limit to first 8
            # Histogram
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, nbinsx=30, showlegend=False),
                row=idx+1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=self.df[col], name=col, showlegend=False),
                row=idx+1, col=2
            )
        
        fig.update_layout(height=250*min(len(self.numeric_cols), 8),
                         title_text="Trazado de Estadísticas de Atributos Individuales",
                         showlegend=False)
        
        return fig
    
    def plot_multivariate_statistics(self):
        """
        Plot statistics for multiple variables relationships
        """
        # Select top numeric columns by variance
        top_cols = self.df[self.numeric_cols].std().nlargest(4).index.tolist()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{col}" for col in top_cols],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        for idx, col in enumerate(top_cols):
            row = idx // 2 + 1
            col_pos = idx % 2 + 1
            
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, nbinsx=40, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=600,
                         title_text="Trazado de Estadísticas de Variables Múltiples")
        
        return fig
    
    def plot_scatter_matrix(self):
        """
        Create scatter plot matrix with identification
        """
        # Select top numeric columns
        cols_to_plot = self.df[self.numeric_cols].std().nlargest(5).index.tolist()
        
        fig = px.scatter_matrix(
            self.df[cols_to_plot],
            dimensions=cols_to_plot,
            title="Matriz de Dispersión - Identificación de Relaciones",
            labels={col: col for col in cols_to_plot},
            hover_data=self.df.columns.tolist()
        )
        
        fig.update_traces(diagonal_visible=True)
        fig.update_layout(height=900, width=1000)
        
        return fig
    
    def plot_correlation_matrix(self):
        """
        Plot correlation matrix as heatmap
        """
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlación")
        ))
        
        fig.update_layout(
            title="Matriz de Correlación - Heatmap",
            height=800,
            width=900,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def get_correlation_analysis(self):
        """
        Detailed correlation analysis
        """
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strongest correlations (excluding diagonal)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlación': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.reindex(corr_df['Correlación'].abs().sort_values(ascending=False).index)
        
        return corr_matrix, corr_df.head(15)
    
    def get_class_imbalance_info(self):
        """
        Analyze class imbalance for target variable
        """
        # Try to identify target column
        potential_targets = ['Status', 'Status_Label', 'Target', 'outcome', 'class']
        target_col = None
        
        for col in potential_targets:
            if col in self.df.columns:
                target_col = col
                break
        
        if target_col is None:
            return None
        
        value_counts = self.df[target_col].value_counts()
        total = len(self.df)
        
        imbalance_info = {
            'target_column': target_col,
            'class_distribution': value_counts,
            'class_percentages': (value_counts / total * 100).round(2),
            'imbalance_ratio': value_counts.max() / value_counts.min(),
            'most_common_class': value_counts.idxmax(),
            'least_common_class': value_counts.idxmin()
        }
        
        return imbalance_info
    
    def plot_class_imbalance(self):
        """
        Visualize class imbalance
        """
        imbalance_info = self.get_class_imbalance_info()
        
        if imbalance_info is None:
            return None
        
        target_col = imbalance_info['target_column']
        value_counts = imbalance_info['class_distribution']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Distribución Absoluta", "Distribución Relativa (%)"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=value_counts.index.astype(str), y=value_counts.values,
                   marker_color='lightblue', showlegend=False),
            row=1, col=1
        )
        
        # Pie chart
        percentages = (value_counts / value_counts.sum() * 100).round(1)
        fig.add_trace(
            go.Pie(labels=value_counts.index.astype(str),
                   values=value_counts.values,
                   textinfo='label+percent',
                   showlegend=True),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"Desequilibrio de Clases - {target_col}",
            height=500
        )
        
        return fig


if __name__ == "__main__":
    print("Feature Engineering Pipeline Module Loaded")
