import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

# Set style for better plots
sns.set_style("white")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def plot_km_age_dichotomy(df, event_col, time_col, age_cutoff=85, title_suffix=""):
    """
    Plot KM curves comparing patients above and below age cutoff with log-rank test
    
    Parameters:
    -----------
    df : DataFrame
        Patient data
    event_col : str
        Column name for event indicator
    time_col : str
        Column name for time to event
    age_cutoff : int
        Age cutoff for comparison (default 85)
    title_suffix : str
        Additional text for plot title
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Prepare data
    df_km = df.dropna(subset=[event_col, time_col, 'age'])
    df_km[time_col] = pd.to_numeric(df_km[time_col], errors='coerce')
    df_km[event_col] = (pd.to_numeric(df_km[event_col], errors='coerce') > 0).astype(int)
    df_km['age'] = pd.to_numeric(df_km['age'], errors='coerce')
    df_km = df_km.dropna(subset=[time_col, 'age'])
    
    # Create age groups
    df_km['age_group'] = (df_km['age'] >= age_cutoff).astype(int)
    
    # Define groups
    groups = {
        0: f'Age < {age_cutoff} years',
        1: f'Age ≥ {age_cutoff} years'
    }
    
    colors = ['#2E86AB', '#A23B72']
    kmf_objects = []
    
    # Plot KM curves for each group
    for group_val, group_label in groups.items():
        mask = df_km['age_group'] == group_val
        group_data = df_km[mask]
        
        if len(group_data) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_data[time_col],
            event_observed=group_data[event_col],
            label=f'{group_label}'
        )
        
        kmf.plot_survival_function(ax=ax, color=colors[group_val], linewidth=2.5, ci_show=True, ci_alpha=0.2)
        kmf_objects.append(kmf)
    
    # Perform log-rank test
    group0_data = df_km[df_km['age_group'] == 0]
    group1_data = df_km[df_km['age_group'] == 1]
    
    if len(group0_data) > 0 and len(group1_data) > 0:
        results = logrank_test(
            durations_A=group0_data[time_col],
            durations_B=group1_data[time_col],
            event_observed_A=group0_data[event_col],
            event_observed_B=group1_data[event_col]
        )
        
        # Add p-value to plot
        p_value_text = f'Log-rank p = {results.p_value:.4f}'
        if results.p_value < 0.001:
            p_value_text = 'Log-rank p < 0.001'
        
        ax.text(
            0.98, 0.02, p_value_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right'
        )
    
    # Add at-risk counts
    if kmf_objects:
        add_at_risk_counts(*kmf_objects, ax=ax, rows_to_show=['At risk'])
    
    # Formatting
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Survival probability', fontsize=12)
    ax.set_title(f'Survival by age group (cutoff: {age_cutoff} years){title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', alpha=1)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig, results if 'results' in locals() else None

def plot_km_four_age_groups(df, event_col, time_col, title_suffix=""):
    """
    Plot KM curves comparing four age groups: <75, 75-79, 80-84, ≥85.
    """
    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))

    # Prepare data
    df_km = df.dropna(subset=[event_col, time_col, 'age'])
    df_km[time_col] = pd.to_numeric(df_km[time_col], errors='coerce')
    df_km[event_col] = (pd.to_numeric(df_km[event_col], errors='coerce') > 0).astype(int)
    df_km['age'] = pd.to_numeric(df_km['age'], errors='coerce')
    df_km = df_km.dropna(subset=[time_col, 'age'])

    # Create detailed age groups
    conditions = [
        df_km['age'] < 75,
        (df_km['age'] >= 75) & (df_km['age'] < 80),
        (df_km['age'] >= 80) & (df_km['age'] < 85),
        df_km['age'] >= 85
    ]
    choices = ['<75', '75-79', '80-84', '≥85']
    df_km['age_group_detailed'] = np.select(conditions, choices, default='')

    # Colors for each group
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    group_labels = ['<75 years', '75-79 years', '80-84 years', '≥85 years']

    kmf_objects = []

    # Plot KM curves
    for i, (group, label) in enumerate(zip(choices, group_labels)):
        group_data = df_km[df_km['age_group_detailed'] == group]
        if len(group_data) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_data[time_col],
            event_observed=group_data[event_col],
            label=f'{label}'
        )
        kmf.plot_survival_function(ax=ax1, color=colors[i], linewidth=2.5, ci_show=False)
        kmf_objects.append(kmf)
        median_surv = kmf.median_survival_time_
        n_events = group_data[event_col].sum()
        print(f"{label}: n={len(group_data)}, events={n_events}, median survival={median_surv:.0f} days")

    # Add at-risk counts only
    if kmf_objects:
        add_at_risk_counts(*kmf_objects, ax=ax1, rows_to_show=['At risk'])

    # Overall log-rank test across all four groups
    if len(df_km) > 0:
        mv_result = multivariate_logrank_test(
            event_durations=df_km[time_col],
            groups=df_km['age_group_detailed'],
            event_observed=df_km[event_col]
        )
        overall_p = float(mv_result.p_value)
        p_text = f'Overall log-rank p = {overall_p:.4f}' if overall_p >= 0.001 else 'Overall log-rank p < 0.001'
        ax1.text(
            0.98, 0.02, p_text,
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment='bottom',
            horizontalalignment='right'
        )

    # Formatting
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.set_ylabel('Survivalprobability', fontsize=12)
    ax1.set_title(f'Survival by detailed age groups{title_suffix}', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig, None

def create_all_km_plots(df, endpoints):
    """
    Create KM plots for each endpoint:
    - Dichotomy: Age < 85 vs ≥ 85 (with CI)
    - Four groups: <75, 75-79, 80-84, ≥85 (no CI, no pairwise p-values)
    """
    os.makedirs('stats/survival/km_age_comparisons', exist_ok=True)

    for event_col, time_col, endpoint_name in endpoints:
        if event_col not in df.columns or time_col not in df.columns:
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {endpoint_name}")
        print('='*60)

        # Plot 1: Age < 85 vs ≥ 85
        fig1, lr_result1 = plot_km_age_dichotomy(
            df, event_col, time_col,
            age_cutoff=85,
            title_suffix=f"\n{endpoint_name}"
        )

        if lr_result1:
            print(f"\nAge <85 vs ≥85 comparison:")
            print(f"  Log-rank chi-square: {lr_result1.test_statistic:.3f}")
            print(f"  P-value: {lr_result1.p_value:.4f}")
            print(f"  Significant: {'Yes' if lr_result1.p_value < 0.05 else 'No'}")

        plt.savefig(f"stats/survival/km_age_comparisons/km_85cutoff_{endpoint_name.replace(' ', '_')}.png",
                    dpi=300, bbox_inches='tight')
        

        # Plot 2: Four age groups (<75, 75-79, 80-84, ≥85)
        fig2, _ = plot_km_four_age_groups(
            df, event_col, time_col,
            title_suffix=f"\n{endpoint_name}"
        )

        plt.savefig(f"stats/survival/km_age_comparisons/km_4groups_{endpoint_name.replace(' ', '_')}.png",
                    dpi=300, bbox_inches='tight')
        

# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    # Load data from PostgreSQL
    db_name = 'evalfast'
    db_user = os.getenv('DB_USER', 'your_username')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')

    df = None
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        query = """
        SELECT 
            age,
            is_male,
            bmi,
            dyslip,
            hta,
            smoker,
            is_diabetic,
            fup_30d_all_death, fup_30d_all_death_time,
            fup_1y_all_death,  fup_1y_all_death_time,
            fup_5y_all_death,  fup_5y_all_death_time,
            fup_1y_mace4,      fup_1y_mace4_time,
            fup_5y_mace4,      fup_5y_mace4_time
        FROM psyfast_fup
        WHERE 1=1
        """
        df = pd.read_sql_query(query, conn)
        df = df.loc[:, ~df.columns.duplicated()]

        # Coerce numeric types
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
        time_cols = [c for c in df.columns if c.endswith('_time')]
        for c in time_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        event_cols = [c for c in df.columns if c.startswith('fup_') and not c.endswith('_time')]
        for c in event_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(0)
            df[c] = (df[c] > 0).astype(int)
        for c in ['is_male', 'is_diabetic', 'hta', 'smoker', 'dyslip']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] = (df[c] > 0).astype(int)
    except Exception as e:
        print(f"Database load error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    # Define endpoints to analyze
    endpoints = [
        ('fup_30d_all_death', 'fup_30d_all_death_time', '30-day mortality'),
        ('fup_1y_all_death', 'fup_1y_all_death_time', '1-year mortality'),
        ('fup_5y_all_death', 'fup_5y_all_death_time', '5-year mortality'),
        ('fup_1y_mace4', 'fup_1y_mace4_time', '1-year MACE'),
        ('fup_5y_mace4', 'fup_5y_mace4_time', '5-year MACE')
    ]
    
    print("\n" + "="*80)
    print("KAPLAN-MEIER SURVIVAL ANALYSIS BY AGE GROUPS")
    print("="*80)
    
    # Create all plots
    create_all_km_plots(df, endpoints)
    
    # Summary statistics by age group
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY AGE GROUP")
    print("="*80)
    
    # For the dichotomous comparison (< 85 vs ≥ 85)
    df['age_85_group'] = (df['age'] >= 85).map({False: '<85', True: '≥85'})
    
    summary_stats = []
    for group in ['<85', '≥85']:
        group_data = df[df['age_85_group'] == group]
        stats = {
            'Age Group': group,
            'N': len(group_data),
            'Mean Age': group_data['age'].mean(),
            'Male %': (group_data['is_male'].mean() * 100) if 'is_male' in group_data else np.nan,
            'Diabetic %': (group_data['is_diabetic'].mean() * 100) if 'is_diabetic' in group_data else np.nan,
            'HTN %': (group_data['hta'].mean() * 100) if 'hta' in group_data else np.nan
        }
        
        # Add mortality rates
        for event_col, _, endpoint_name in endpoints:
            if event_col in group_data.columns:
                mortality_rate = group_data[event_col].mean() * 100
                stats[f'{endpoint_name} (%)'] = mortality_rate
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    print("\nComparison: Age < 85 vs ≥ 85")
    print(summary_df.round(1).to_string())
    
    # For the four-group comparison
    conditions = [
        df['age'] < 75,
        (df['age'] >= 75) & (df['age'] < 80),
        (df['age'] >= 80) & (df['age'] < 85),
        df['age'] >= 85
    ]
    choices = ['<75', '75-79', '80-84', '≥85']
    df['age_4groups'] = np.select(conditions, choices, default='')
    
    summary_stats_4 = []
    for group in choices:
        group_data = df[df['age_4groups'] == group]
        stats = {
            'Age Group': group,
            'N': len(group_data),
            'Mean Age': group_data['age'].mean() if len(group_data) > 0 else np.nan
        }
        
        # Add mortality rates
        for event_col, _, endpoint_name in endpoints[:3]:  # Focus on main mortality endpoints
            if event_col in group_data.columns and len(group_data) > 0:
                mortality_rate = group_data[event_col].mean() * 100
                stats[f'{endpoint_name} (%)'] = mortality_rate
        
        summary_stats_4.append(stats)
    
    summary_df_4 = pd.DataFrame(summary_stats_4)
    print("\n\nDetailed Age Groups: <75, 75-79, 80-84, ≥85")
    print(summary_df_4.round(1).to_string())
    
    # Save summary tables
    os.makedirs('stats/survival/km_age_comparisons', exist_ok=True)
    summary_df.to_csv('stats/survival/km_age_comparisons/summary_85cutoff.csv', index=False)
    summary_df_4.to_csv('stats/survival/km_age_comparisons/summary_4groups.csv', index=False)
    
    print("\n" + "="*80)
    print("Analysis complete! All plots and summaries saved to:")
    print("  stats/survival/km_age_comparisons/")
    print("="*80)