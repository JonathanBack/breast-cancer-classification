import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer(as_frame=True)
df = breast_cancer_data.frame
total_samples = len(df)

# Maping names to the targets
df['target_name'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

# Features that we will explore
main_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area'] 

# Numerical summary
df[main_features].describe()

# DataViz: histograma
sns.histplot(
    data=df, x = "mean radius", hue="target_name", kde=True, element="step"
)
plt.title("Mean radius distribution")

# DataViz: boxplot
sns.boxplot(data=df, x="target_name", y="mean area")
plt.title('Mean area boxplot')

# DataViz: PairPlot
sns.pairplot(
    data=df[main_features + ['target_name']], hue="target_name"
)

# DataViz: parallel coords
paralel_features = ['mean radius', 'mean texture', 'mean perimeter']

plt.figure(figsize=(10,5))
pd.plotting.parallel_coordinates(
    df[paralel_features + ['target_name']],
    class_column = "target_name"
)
plt.title('Parallel coordinates plot')
plt.grid(True, linestyle = "--", alpha=0.5)
plt.show()

# DataViz

# Prepation for probabilities calculation
df_mal = df[df['target_name'] == 'Malignant']
df_ben = df[df['target_name'] == 'Benign']

mal_samples = len(df_mal)
ben_samples = len(df_ben)

# Numerical summary per class
df_mal[main_features].describe()
df_ben[main_features].describe()

# Prior probabilities
prior_mal = mal_samples / total_samples
prior_ben = ben_samples / total_samples

# Calculating thresholds for defining events and their probabilities 
area_threshold = df['mean area'].mean()
texture_threshold = df['mean texture'].mean()
radius_threshold = df['mean radius'].mean()

# Event A: mean area > area threshold
eventA = len(df[df['mean area'] > area_threshold])
probA = eventA / total_samples

# Event B:  mean texture > texture threshold
eventB = len(df[df['mean texture'] > texture_threshold])
probB = eventB / total_samples

# Event C: mean radius > radius threshold
eventC = len(df[df['mean radius'] > radius_threshold])
probC = eventC / total_samples

# Event D: mean area < area threshold AND mean texture > texture threshold
eventD = len(df[(df['mean area'] > area_threshold) & (df['mean texture'] > texture_threshold)])
probD = eventD / total_samples

# EventE: mean area > area threshold OR mean texture < texture threshold
eventE = len(df[(df['mean area'] > area_threshold) | (df['mean texture'] < texture_threshold)])
probE = eventE / total_samples

# Conditional probability
# EventF: IF A = mean area > area threshold AND B is Malignant

EventF = len(df[(df['mean area'] > area_threshold) & (df['target_name'] == 'Malignant')])  
probAB = EventF / total_samples

probcondAB = probAB / prior_mal

# Bayesian decision

def calculate_and_plot_bayes(feature_name):
    mu_mal, std_mal = df_mal[feature_name].mean(), df_mal[feature_name].std()
    mu_ben, std_ben = df_ben[feature_name].mean(), df_ben[feature_name].std()
    
    x_range = np.linspace(df[feature_name].min(), df[feature_name].max(), 500 )

    # PDFs conditional gaussians
    pdf_mal = stats.norm.pdf(x_range, mu_mal, std_mal)
    pdf_ben = stats.norm.pdf(x_range, mu_ben, std_ben)

    # Posterior by bayes rule
    evidence = (pdf_mal * prior_mal) + (pdf_ben * prior_ben)
    posterior_mal = (pdf_mal * prior_mal) / evidence
    posterior_ben = (pdf_ben * prior_ben) / evidence
    
    # Associated plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    
    # Histograms and PDF curves
    sns.histplot(
        df,
        x=feature_name,
        hue="target_name",
        stat='density',
        common_norm=False,
        alpha=0.3,
        bins=15,
        ax=ax1
    )
    ax1.plot(x_range, pdf_mal, color='red', lw=2, label='p(x | Malignant)')
    ax1.plot(x_range, pdf_ben, color='blue', lw=2, label='p(x | Benign)')
    ax1.set_title(f"Conditional PDF and Histogram: {feature_name}")
    ax1.legend()

    # Posterior probability curves
    ax2.plot(x_range, posterior_mal, color='red', lw=2, label='P(Malignant | x)')
    ax2.plot(x_range, posterior_ben, color='blue', lw=2, label='P(Benign | x)')
    ax2.axhline(0.5, color='gray', linestyle='--', label='Decision threshold')
    ax2.set_title(f"Posterior Probability: P(Class | x) for {feature_name}")
    ax2.set_ylabel("Probability")
    ax2.legend()

calculate_and_plot_bayes("mean area")
calculate_and_plot_bayes("mean texture")