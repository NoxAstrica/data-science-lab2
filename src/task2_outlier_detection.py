"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}

    def _row_mask(self, method: str) -> pd.Series:
        """True for every row that is an outlier under the given method."""
        if method not in self.outlier_masks:
            detectors = {
                "zscore": self.detect_outliers_zscore,
                "iqr": self.detect_outliers_iqr,
                "isolation_forest": self.detect_outliers_isolation_forest,
                "lof": self.detect_outliers_lof,
            }
            if method not in detectors:
                raise ValueError(f"Unknown method: {method}")
            detectors[method]()
        return self.outlier_masks[method].any(axis=1)
        
    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        num = self.data.select_dtypes(include=[np.number])
        z = (num - num.mean()) / num.std()
        mask = z.abs() > threshold
        self.outlier_masks["zscore"] = mask
        return mask

    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        num = self.data.select_dtypes(include=[np.number])
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        mask = (num < q1 - multiplier * iqr) | (num > q3 + multiplier * iqr)
        self.outlier_masks["iqr"] = mask
        return mask

    def detect_outliers_isolation_forest(self, contamination: float = 0.1,
                                        random_state: int = 42) -> pd.DataFrame:
        num = self.data.select_dtypes(include=[np.number])
        model = IsolationForest(contamination=contamination, random_state=random_state)
        pred = model.fit_predict(num.fillna(num.median()))   # -1 means outlier
        mask = pd.DataFrame({"is_outlier": pred == -1}, index=self.data.index)
        self.outlier_masks["isolation_forest"] = mask
        return mask

    def detect_outliers_lof(self, n_neighbors: int = 20,
                            contamination: float = 0.1) -> pd.DataFrame:
        num = self.data.select_dtypes(include=[np.number])
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        pred = model.fit_predict(num.fillna(num.median()))
        mask = pd.DataFrame({"is_outlier": pred == -1}, index=self.data.index)
        self.outlier_masks["lof"] = mask
        return mask

    def get_outlier_summary(self) -> pd.DataFrame:
        rows = []
        for method in ["zscore", "iqr", "isolation_forest", "lof"]:
            flagged = self._row_mask(method)
            rows.append({
                "method": method,
                "outlier_count": int(flagged.sum()),
                "outlier_percentage": flagged.mean() * 100,
            })
        return pd.DataFrame(rows).set_index("method").T

    def remove_outliers(self, method: str = "iqr") -> pd.DataFrame:
        df = self.data[~self._row_mask(method)].copy()
        self.treated_datasets[f"removed_{method}"] = df
        return df

    def cap_outliers(self, method: str = "iqr", multiplier: float = 1.5) -> pd.DataFrame:
        df = self.data.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
        iqr = q3 - q1
        df[num_cols] = df[num_cols].clip(lower=q1 - multiplier * iqr,
                                        upper=q3 + multiplier * iqr, axis=1)
        self.treated_datasets["capped"] = df
        return df

    def transform_outliers_log(self) -> pd.DataFrame:
        df = self.data.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            shift = -df[col].min() if df[col].min() < 0 else 0   # make everything >= 0
            df[col] = np.log1p(df[col] + shift)
        self.treated_datasets["log_transform"] = df
        return df

    def transform_outliers_boxcox(self) -> pd.DataFrame:
        df = self.data.copy()
        cols = df.select_dtypes(include=[np.number]).columns
        df[cols] = PowerTransformer(method="yeo-johnson").fit_transform(df[cols])
        self.treated_datasets["boxcox_transform"] = df
        return df
    
    def evaluate_model_performance(self, target_col: str,
                               dataset_name: str = "original") -> Dict[str, float]:
        if dataset_name == "original":
            df = self.original_data
        elif dataset_name in self.treated_datasets:
            df = self.treated_datasets[dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        df = df.select_dtypes(include=[np.number]).dropna()
        X, y = df.drop(columns=[target_col]), df[target_col]
        
        scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
        model = LinearRegression().fit(X, y)
        return {
            "r2_score": float(model.score(X, y)),
            "r2_std": float(scores.std()),
            "mean_cv_score": float(scores.mean()),
        }
        
    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        cols = self.original_data.select_dtypes(include=[np.number]).columns[:6]
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        for ax, col in zip(axes, cols):
            sns.boxplot(y=self.original_data[col], ax=ax)
            ax.set_title(col)
        for ax in axes[len(cols):]:
            ax.axis("off")
        plt.tight_layout()
        
        return fig
    
    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
    
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        for ax, col in zip(axes, numeric_cols):
            sns.violinplot(y=self.data[col], ax=ax)
            ax.axhline(self.data[col].mean(), color="red", linestyle="--", label="mean")
            ax.axhline(self.data[col].median(), color="green", linestyle=":", label="median")
            ax.set_title(col)
            ax.legend()
        for ax in axes[len(numeric_cols):]:
            ax.axis("off")
        plt.tight_layout()
        
        return fig
    
    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Column not found: {x_col} or {y_col}")
            return None
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, method in zip(axes, ["zscore", "iqr", "isolation_forest"]):
            self._row_mask(method)                      # make sure the mask exists
            mask = self.outlier_masks[method]
            flagged = mask[[x_col, y_col]].any(axis=1) if method != "isolation_forest" else mask["is_outlier"]
            ax.scatter(self.data[x_col], self.data[y_col],
                    c=np.where(flagged, "red", "steelblue"), alpha=0.6)
            ax.set_title(method)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        plt.tight_layout()
                                       
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        from mpl_toolkits.mplot3d import Axes3D

        if method not in self.outlier_masks:
            print(f"Method '{method}' not found. Run the detection first.")
            return None

        flagged = self._row_mask(method)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.data[x_col], self.data[y_col], self.data[z_col],
                c=np.where(flagged, "red", "steelblue"), alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"Outliers ({method})")
                                 
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
