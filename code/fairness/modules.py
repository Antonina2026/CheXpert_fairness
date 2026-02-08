import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def data_representation_fairness(dataset, sensitive_attribute):
    counts = dataset[sensitive_attribute].value_counts().sort_index()
    proportions = counts / counts.sum()
    representation_rate = round(counts / counts.sum()*100,4)

    # Multi-Group Disparate Impact Ratio (Relative to Best Group)
    disparate_impact_ratio = round(proportions / proportions.max(),4)

    # Expected uniform proportion
    expected = 1 / len(proportions)
    
    # Generalised Disparate Impact (Relative to Total Population
    # AIF360 Similar to Statistical Parity (dataset-level variant)
    gdi = round(proportions / expected,4)
    
    df = pd.DataFrame({
    sensitive_attribute: counts.index,
    'Samples': counts.values,
    'Representation rate, %': representation_rate.values,
    'Disparate impact ratio': disparate_impact_ratio,
    'Generalised DI ratio': gdi,    
    }).reset_index(drop=True)

    conditions = [
    df['Disparate impact ratio'] == 1,   
    (df['Disparate impact ratio'] >= 0.8) & (df['Disparate impact ratio'] < 1),
    (df['Disparate impact ratio'] >= 0.6) & (df['Disparate impact ratio'] < 0.8),
    (df['Disparate impact ratio'] >= 0.4) & (df['Disparate impact ratio'] < 0.6),
    df['Disparate impact ratio'] < 0.4
    ]
    
    choices = [
        "Reference",
        "Acceptable",
        "Moderate Imbalance", 
        "Severe Underrepresentation", 
        "Critical Underrepresentation" 
    ]

    df['DI interpretation'] = np.select(conditions, choices)
    # df['Interpretation'] = df['Disparate impact ratio'].apply(ratio_interpretation)

    conditions = [
    (df['Generalised DI ratio'] >= 0.95) & (df['Generalised DI ratio'] <= 1.05),    
    (df['Generalised DI ratio'] >= 0.8) & (df['Generalised DI ratio'] < 0.95),    
    (df['Generalised DI ratio'] < 0.8),
    (df['Generalised DI ratio'] > 1.05) & (df['Generalised DI ratio'] <= 1.25),
     df['Generalised DI ratio'] > 1.25
    ]
    
    choices = [
        "Well Represented",
        "Moderate Underrepresentation",
        "Severe Underrepresentation",
        "Moderate Overrepresentation",
        "Severe Overrepresentation"
    ]

    df['GDI interpretation'] = np.select(conditions, choices)
    
    df = df[[sensitive_attribute, 'Samples', 'Representation rate, %', 'Disparate impact ratio', 'DI interpretation', 
             'Generalised DI ratio','GDI interpretation']]

    return df

def risk_ratio_multilabel(dataset, sensitive_attribute, labels):

    records = []

    for label in labels:
        prevalence = dataset.groupby(sensitive_attribute)[label].mean()
        ref_group = prevalence.idxmax()
        ref_rate = prevalence.loc[ref_group]

        for group, rate in prevalence.items():
            risk_ration = round(rate / ref_rate,2)
            records.append({
                "Label": label,
                "Attribute": group,
                "Prevalence": round(rate,4),
                "Risk ratio": risk_ration,
                "Reference group": ref_group
            })


    df = pd.DataFrame(records)
    
    conditions = [
    (df['Risk ratio'] == 1) ,    
    (df['Risk ratio'] >= 0.8) & (df['Risk ratio'] <= 1.25),    
    (df['Risk ratio'] < 0.8),
     df['Risk ratio'] > 1.25
    ]
    
    choices = [
        "Equal prevalence",
        "Acceptable",
        "Underrepresentation",
        "Overrepresentation"
    ]

    df['Interpretation'] = np.select(conditions, choices)

    return df

def coverage_gap_intersectional(dataset, group1_col, group2_col):
 
    # Create intersectional group
    df = dataset.copy()
    df["intersection_group"] = ( df[group1_col].astype(str) + " | " + df[group2_col].astype(str) )

    # Count samples per group
    counts = df["intersection_group"].value_counts().sort_index()
    total = counts.sum()

    # Empirical proportions
    proportions = counts / total
    proportions_perc = round(counts / total*100,4)

    # Ideal proportion
    num_groups = len(proportions)
    ideal_proportion = 1 / num_groups
    ideal_proportion_perc = round(1 / num_groups*100,4)

    # Coverage gap
    coverage_gap = ideal_proportion - proportions
    coverage_gap_samples = round((ideal_proportion- proportions)*total,0)
    

    result = pd.DataFrame({
        "Intersection group": proportions.index,
        "Samples": counts.values,
        "Proportion, %": proportions_perc,
        "Ideal proportion, %": ideal_proportion_perc,
        "Coverage Gap, samples": coverage_gap_samples
    })

    return result.sort_values("Intersection group")

def risk_ratio_multilabel_intersectional_group(dataset, group1_col, group2_col, labels):

    # Create intersectional group
    df = dataset.copy()
    df["intersection_group"] = ( df[group1_col].astype(str) + " | " + df[group2_col].astype(str) )

    records = []

    for label in labels:
        prevalence = df.groupby("intersection_group")[label].mean()
        
        ref_group = prevalence.idxmax()
        ref_rate = prevalence.loc[ref_group]

        for group, rate in prevalence.items():
            risk_ration = round(rate / ref_rate,2)
            records.append({
                "Label": label,
                "Attribute": group,
                "Prevalence": round(rate,4),
                "Risk ratio": risk_ration,
                "Reference group": ref_group
            })


    df_result = pd.DataFrame(records)
    
    conditions = [
    (df_result['Risk ratio'] == 1) ,    
    (df_result['Risk ratio'] >= 0.8) & (df_result['Risk ratio'] <= 1.25),    
    (df_result['Risk ratio'] < 0.8),
     df_result['Risk ratio'] > 1.25
    ]
    
    choices = [
        "Equal prevalence",
        "Acceptable",
        "Underrepresentation",
        "Overrepresentation"
    ]

    df_result['Interpretation'] = np.select(conditions, choices)

    return df_result

def js_divergence_by_group(dataset, sensitive_attribute, labels):
    """
    Compute Jensen–Shannon divergence between each group's label distribution
    and the total population distribution.

    """

    # ----- Population distribution -----
    pop_dist = dataset[labels].mean().values
    pop_dist = pop_dist / pop_dist.sum()

    results = []

    for group, group_df in dataset.groupby(sensitive_attribute):
        group_dist = group_df[labels].mean().values
        group_dist = group_dist / group_dist.sum()

        js_value = jensenshannon(group_dist, pop_dist, base=2)

        # ----- Interpretation -----
        if js_value < 0.05:
            interpretation = "Very similar to population"
        elif js_value < 0.15:
            interpretation = "Moderately different"
        elif js_value < 0.30:
            interpretation = "Substantially different"
        else:
            interpretation = "Highly divergent"

        results.append({
            sensitive_attribute: group,
            "JS_value": round(js_value, 4),
            "Interpretation": interpretation
        })

    return pd.DataFrame(results)
    
def js_divergence_by_label_and_group(dataset, sensitive_attribute, labels):
    """
    Compute JS divergence per label and per group
    comparing group label distribution to population.

    Returns a tidy DataFrame.
    """

    results = []

    for label in labels:
        # population distribution for this label (Yes / No)
        pop_dist = dataset[label].value_counts(normalize=True).reindex([0,1], fill_value=0).values

        for group, group_df in dataset.groupby(sensitive_attribute):
            group_dist = group_df[label].value_counts(normalize=True).reindex([0,1], fill_value=0).values

            js_value = jensenshannon(group_dist, pop_dist, base=2)

            if js_value < 0.05:
                interpretation = "Very similar to population"
            elif js_value < 0.15:
                interpretation = "Moderately different"
            elif js_value < 0.30:
                interpretation = "Substantially different"
            else:
                interpretation = "Highly divergent"

            results.append({
                "Label": label,
                sensitive_attribute: group,
                "JS_value": round(js_value, 4),
                "Interpretation": interpretation
            })

    return pd.DataFrame(results)