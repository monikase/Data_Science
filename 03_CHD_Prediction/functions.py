import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import Counter
from scipy.stats import chi2_contingency

# F1. Summary Table

def summarize_chd(df, column_name, chd_column="TenYearCHD"):
    """
    Summarize counts, percentages, CHD percentages, 
    and absolute CHD percentages for a categorical column.
    """
    summary = df[column_name].value_counts().rename("Count")
    
    # Percentage of each group
    percent = (summary / summary.sum() * 100).round(1).astype(str) + "%"
    
    # CHD percentage within each group
    chd_percent = (
        df[df[chd_column] == 1][column_name].value_counts() / summary * 100
    ).round(1).astype(str) + "%"
    
    # Absolute CHD percentage relative to the total dataset
    chd_absolute_percent = (
        df[df[chd_column] == 1][column_name].value_counts() / len(df) * 100
    ).round(1).astype(str) + "%"
    
    df_summary = pd.concat([summary, percent, chd_percent, chd_absolute_percent], axis=1).fillna("0.0%")
    df_summary.columns = ["Count", "Percentage", "CHD_Percentage", "CHD_Absolute_Percentage"]
    
    return df_summary

# F2. Plotting Categorical values with CHD status.

def plot_chd_categorical_distribution(df, column_name, title, legend_labels):
    """
    Plots the distribution of a categorical variable against the TenYearCHD status.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the categorical column.
        title (str): The title of the plot.
        legend_labels (list): The labels for the legend (e.g., ["No", "Yes']).
    """
    ax = sns.countplot(x=column_name, hue="TenYearCHD", data=df)

    # Calculate the total counts for each category
    category_totals = df.groupby(column_name, observed=False)['TenYearCHD'].count()
    
    # Iterate through the bars to add the percentage labels
    for p in ax.patches:
        category_index = int(p.get_x() + 0.5) 
        category = sorted(df[column_name].unique())[category_index]

        height = p.get_height()
        
        # Get the total count for the current category
        total_count = category_totals.loc[category]
        
        # Calculate the relative percentage
        percentage = (height / total_count) * 100
        
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12
            )

    plt.xlabel(f"{column_name}")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(title="TenYearCHD", labels=legend_labels)
    plt.show()

# F3.  Segment the TenYearCHD=1 population by age bins and a categorical variable, then plot the counts and percentages.

def plot_age_distribution_by_category(df, category_col, title):
    """
    Generates a bar plot showing the age distribution of individuals with TenYearCHD=1,
    segmented by a specified categorical variable, with percentage annotations.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        category_col (str): The name of the categorical column to group by (e.g., 'sex', 'is_smoking').
        title (str): The title for the plot.
    """
    # Define age bins and filter for TenYearCHD=1
    bins = range(35, 76, 5)
    df_chd = df[df["TenYearCHD"] == 1].copy()
    df_chd["age_bin"] = pd.cut(df_chd["age"], bins=bins, right=False)

    # Calculate counts and percentages
    counts = df_chd.groupby(["age_bin", category_col], observed=False).size().unstack(fill_value=0)
    total_count = counts.values.sum()
    percents = (counts / total_count) * 100

    ax = counts.plot(kind="bar", rot=0, title=title)
    plt.ylabel("Counts")
    plt.xlabel("Age Bins")

    for container, cat in zip(ax.containers, counts.columns):
        for i, bar in enumerate(container):
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{percents.iloc[i][cat]:.1f}%",
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=12
                )
    plt.show()

# F4. Bins Plotting

def plot_bins(df, col, bins, labels, title, xlabel):
    """
    Create a bar plot with counts and percentage annotations for a binned variable vs. TenYearCHD.    
    """
    # Create a temporary Series for bins instead of modifying df
    col_bins = pd.cut(df[col], bins=bins, labels=labels, right=False)

    # Counts + percentages
    counts = df.groupby([col_bins, "TenYearCHD"], observed=False).size().unstack(fill_value=0)
    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    # Plot
    ax = counts.plot(kind="bar", rot=0, title=title)
    plt.ylabel("Counts")
    plt.xlabel(xlabel)

    # Annotate bars with percentages
    for (i, j), val in counts.stack().items():
        perc = percents.loc[i, j]
        ax.annotate(
            f"{perc:.1f}%",
            (list(counts.index).index(i) + (list(counts.columns).index(j)-0.5)*0.2, val),
            ha="center", va="bottom", fontsize=12
        )

    plt.legend(title="TenYearCHD")
    plt.show()

# F5. Boxplot Function

def plot_variable_by_chd(df, x_col, title):
    """
    Generates a box plot showing the distribution of a variable
    by TenYearCHD status, with median values annotated.    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column to plot on the x-axis.
        title (str): The title for the plot.
    """
    # Calculate the median for each TenYearCHD group
    median_values = df.groupby("TenYearCHD", observed=False)[x_col].median()

    # Create the horizontal box plot
    ax = sns.boxplot(x=x_col, y="TenYearCHD", data=df, orient="h")

    # Annotate the plot with the median values
    for idx, val in enumerate(median_values):
        ax.text(val, idx, f"{val:.1f}", va="center", ha="left",
                fontsize=12, color="white", weight="bold")

    ax.set_xlabel(x_col)
    ax.set_ylabel("TenYearCHD (0 = No, 1 = Yes)")
    ax.set_title(title)
    
    plt.show()

# F6. Blood Pressure Graph

def plot_bp_categories(df, sys_col="sysBP", dia_col="diaBP"):
    """
    Plots blood pressure categories for a DataFrame with systolic and diastolic BP.
    Args:
        df (pd.DataFrame): DataFrame with BP data
        sys_col (str): Column name for systolic BP
        dia_col (str): Column name for diastolic BP
    """
    # Categories function
    def bp_category(sys, dia):
        if (sys >= 160) or (dia >= 100):
            return 4  # Hypertension Stage 2
        elif (140 <= sys < 160) or (90 <= dia < 100):
            return 3  # Hypertension Stage 1
        elif (120 <= sys < 140) or (80 <= dia < 90):
            return 2  # Prehypertension
        elif (90 <= sys < 120) and (60 <= dia < 80):
            return 1  # Normal
        elif (sys < 90) and (dia < 60):
            return 0  # Low
        else:
            return 1  # fallback Normal

    # Background grid
    x_min, x_max = 40, 150
    y_min, y_max = 70, 260
    nx, ny = 400, 400
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(x, y)
    
    vec_cat = np.vectorize(bp_category)
    cat_grid = vec_cat(YY, XX)
    
    colors = ["#c7d7f9", "#a8e6a1", "#ffd966", "#f7a6a6", "#d54e5a"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5, 1), cmap.N)
    
    fig, ax = plt.subplots(figsize=(11,8))
    
    # Background
    ax.pcolormesh(x, y, cat_grid, cmap=cmap, norm=norm, shading="auto", alpha=0.45, zorder=0)
    
    # classify points
    point_cats = df.apply(lambda r: bp_category(r[sys_col], r[dia_col]), axis=1)
    
    # Scatter points
    ax.scatter(df[dia_col], df[sys_col],
               c=point_cats, cmap=cmap, norm=norm,
               s=28, edgecolor="k", linewidth=0.3, alpha=0.9, zorder=5)
    
    # Percentages
    counts = Counter(point_cats)
    total = len(point_cats)
    labels = ["Low", "Normal", "Prehypertension", "Hypertension Stage 1", "Hypertension Stage 2"]
    percentages = {i: (counts.get(i, 0)/total)*100 for i in range(len(labels))}
    
    # Legend
    handles = [
        Patch(facecolor=colors[i], edgecolor="k",
              label=f"{labels[i]} ({percentages[i]:.1f}%)")
        for i in range(len(labels))
    ]
    ax.legend(handles=handles, title="BP Category", loc="upper left", fontsize=10, title_fontsize=10)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Diastolic BP (Lower Number)", fontsize=10)
    ax.set_ylabel("Systolic BP (Upper Number)", fontsize=10)
    ax.set_title("Blood Pressure Categories Distribution", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)   
    plt.show()

# F7. Histogram Plot

def plot_hist(df, column, bins=30, kde=True, xlabel=None, ylabel="Frequency", title=None):
    sns.histplot(data=df, x=column, bins=bins, kde=kde)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.title(title if title else f"Distribution of {column}")
    plt.show()

# F8. Correlation Function

def calculate_phi_correlation(df, column1, column2):
    """    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column1 (str): The name of the first binary column.
        column2 (str): The name of the second binary column.        
    Returns:
        tuple: A tuple containing the Phi coefficient and a string interpretation.
    """
    # Create a contingency table
    contingency_table = pd.crosstab(df[column1], df[column2])
    
    # Calculate the chi-squared statistic
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Total number of observations
    n = contingency_table.sum().sum()
    
    # Phi coefficient
    phi = np.sqrt(chi2 / n)
    
    # Determine the sign of the correlation. A positive sign indicates that
    # as one variable increases (from 0 to 1), the other also tends to increase.
    if contingency_table.iloc[0, 0] * contingency_table.iloc[1, 1] < \
       contingency_table.iloc[0, 1] * contingency_table.iloc[1, 0]:
        phi = -phi

    if abs(phi) < 0.1:
        interpretation = "Very weak correlation"
    elif abs(phi) < 0.3:
        interpretation = "Weak correlation"
    elif abs(phi) < 0.5:
        interpretation = "Moderate correlation"
    else:
        interpretation = "Strong correlation"
        
    return phi, interpretation

