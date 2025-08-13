import psycopg2
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.exceptions import ConvergenceError
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts
import seaborn as sns
from scipy import stats
from datetime import datetime
import io

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Load environment variables
load_dotenv()

# Database connection details
db_name = 'evalfast'
db_user = os.getenv('DB_USER', 'your_username')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')

# ========================================
# DATABASE CONNECTION AND DATA RETRIEVAL
# ========================================

try:
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    print("Successfully connected to the database.")

    # SQL query - focusing on STEMI patients who were revascularized
    query = """
    SELECT 
        age,
        is_male,
        bmi,
        dyslip,
        hta,
        smoker,
        is_diabetic,
        atcd_mi,
        atcd_cabg,
        atcd_pci,
        atcd_stroke,
        atcd_bleeding,
        -- Mortality endpoints
        fup_30d_all_death, fup_30d_all_death_time,
        fup_1y_all_death, fup_1y_all_death_time,
        fup_5y_all_death, fup_5y_all_death_time,
        -- CV mortality
        fup_30d_cvdeath, fup_30d_cvdeath_time,
        fup_1y_cvdeath, fup_1y_cvdeath_time,
        fup_5y_cvdeath, fup_5y_cvdeath_time,
        -- Other outcomes
        fup_1y_mace4, fup_1y_mace4_time,
        fup_5y_mace4, fup_5y_mace4_time
    FROM 
        psyfast_fup
    WHERE 
        1=1
        -- Add condition for STEMI patients who underwent revascularization
        -- Adjust this WHERE clause based on your database schema
        -- AND diagnosis = 'STEMI' 
        -- AND revascularization = 1
    """

    df = pd.read_sql_query(query, conn)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Coerce numeric types for analysis
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    
    # Time columns (days)
    time_cols = [c for c in df.columns if c.endswith('_time')]
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Event indicator columns (0/1)
    event_cols = [c for c in df.columns if c.startswith('fup_') and not c.endswith('_time')]
    for c in event_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(0)
        df[c] = (df[c] > 0).astype(int)
    
    # Binary covariates
    for c in ['is_male', 'is_diabetic', 'hta', 'smoker', 'dyslip', 'atcd_mi', 'atcd_cabg', 'atcd_pci', 'atcd_stroke', 'atcd_bleeding']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = (df[c] > 0).astype(int)
    
    print(f"Data loaded: {len(df)} patients")

except Exception as e:
    print(f"Database connection error: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
        print("Database connection closed.")

# ========================================
# LOAD ACTUAL SWISS LIFE TABLE DATA (2020-2023)
# ========================================

def load_swiss_life_table_from_hmd():
    """
    Load actual Swiss life table data from HMD format (2020-2023)
    Using the real data you provided
    """
    
    # Full Swiss life table data for 2020-2023
    # This is the actual data from the Human Mortality Database
    life_table_text = """
Year,Age,mx,qx,ax,lx,dx,Lx,Tx,ex
2020-2023,0,0.00356,0.00355,0.14,100000,355,99715,8419090,84.19
2020-2023,1-4,0.00021,0.00084,1.39,99645,84,398215,8319375,83.50
2020-2023,5-9,0.00010,0.00050,2.69,99561,50,497680,7921160,79.57
2020-2023,10-14,0.00010,0.00050,2.83,99511,50,497430,7423480,74.60
2020-2023,15-19,0.00028,0.00140,2.82,99461,139,496957,6926051,69.63
2020-2023,20-24,0.00041,0.00205,2.73,99322,204,495973,6429094,64.73
2020-2023,25-29,0.00041,0.00205,2.68,99118,203,494829,5933120,59.85
2020-2023,30-34,0.00052,0.00260,2.76,98915,257,493931,5438291,54.97
2020-2023,35-39,0.00065,0.00324,2.67,98658,320,492460,4944360,50.11
2020-2023,40-44,0.00091,0.00453,2.68,98338,446,490682,4451900,45.27
2020-2023,45-49,0.00122,0.00608,2.71,97892,595,488022,3961218,40.47
2020-2023,50-54,0.00201,0.01000,2.68,97297,973,484053,3473196,35.70
2020-2023,55-59,0.00333,0.01655,2.68,96324,1594,477634,2989143,31.03
2020-2023,60-64,0.00558,0.02753,2.71,94730,2607,467131,2511509,26.52
2020-2023,65-69,0.00920,0.04503,2.66,92123,4147,450496,2044378,22.20
2020-2023,70-74,0.01498,0.07240,2.68,87976,6369,424958,1593882,18.11
2020-2023,75-79,0.02579,0.12165,2.68,81607,9931,384707,1168924,14.32
2020-2023,80-84,0.04846,0.21781,2.68,71676,15611,322353,784217,10.94
2020-2023,85-89,0.09841,0.39756,2.58,56065,22297,226082,461864,8.24
2020-2023,90-94,0.18991,0.63135,2.35,33768,21318,112270,235782,6.98
2020-2023,95-99,0.32620,0.82893,2.03,12450,10320,31451,123512,9.92
2020-2023,100-104,0.50352,0.93788,1.65,2130,1998,3966,92061,43.23
2020-2023,105-109,0.68468,0.97753,1.35,132,129,189,88095,667.39
2020-2023,110+,0.81261,1.00000,1.23,3,3,4,87906,29302.00
"""
    
    # Parse the life table
    df_lt = pd.read_csv(io.StringIO(life_table_text))
    
    # Process age column to get numeric age values
    df_lt['age_start'] = 0
    df_lt['age_end'] = 0
    
    for idx, row in df_lt.iterrows():
        age_str = row['Age']
        if age_str == '0':
            df_lt.loc[idx, 'age_start'] = 0
            df_lt.loc[idx, 'age_end'] = 1
        elif '-' in age_str:
            parts = age_str.split('-')
            df_lt.loc[idx, 'age_start'] = int(parts[0])
            df_lt.loc[idx, 'age_end'] = int(parts[1]) + 1
        elif '+' in age_str:
            df_lt.loc[idx, 'age_start'] = int(age_str.replace('+', ''))
            df_lt.loc[idx, 'age_end'] = 120
        else:
            df_lt.loc[idx, 'age_start'] = int(age_str)
            df_lt.loc[idx, 'age_end'] = int(age_str) + 1
    
    df_lt['age_mid'] = (df_lt['age_start'] + df_lt['age_end']) / 2
    
    return df_lt

# Load the actual Swiss life table
swiss_lt_hmd = load_swiss_life_table_from_hmd()
print("Loaded actual Swiss life table data (2020-2023) from HMD")
print(f"Life expectancy at birth: {swiss_lt_hmd.loc[0, 'ex']:.2f} years")

# ========================================
# CREATE INTERPOLATED LIFE TABLE FOR ALL AGES
# ========================================

def create_complete_life_table(df_lt):
    """
    Create a complete life table with values for every single age
    by interpolating from the HMD age groups
    """
    
    # Create dataframe for all ages 0-110
    all_ages = pd.DataFrame({'age': range(111)})
    
    # Key columns to interpolate
    columns_to_interpolate = ['qx', 'ex', 'mx']
    
    for col in columns_to_interpolate:
        # Create mapping from age to value
        age_values = []
        for _, row in df_lt.iterrows():
            if row['age_start'] == row['age_end'] - 1:
                # Single year age
                age_values.append((row['age_start'], row[col]))
            else:
                # Age group - use midpoint or distribute
                if row['age_start'] < 100:  # Not the open-ended group
                    # Use the midpoint for age groups
                    age_values.append((row['age_mid'], row[col]))
        
        # Create interpolation dataframe
        interp_df = pd.DataFrame(age_values, columns=['age', col])
        
        # Merge and interpolate
        all_ages = pd.merge(all_ages, interp_df, on='age', how='left')
        all_ages[col] = all_ages[col].interpolate(method='linear')
        all_ages[col] = all_ages[col].fillna(method='bfill').fillna(method='ffill')
    
    # Calculate survival probabilities for different horizons
    all_ages['px'] = 1 - all_ages['qx']  # 1-year survival probability
    
    # Calculate cumulative survival for different time horizons
    for horizon in [1, 2, 3, 5, 10]:
        all_ages[f'surv_{horizon}y'] = np.nan
        
        for age in range(111):
            if age + horizon <= 110:
                # Product of annual survival probabilities
                surv = 1.0
                for y in range(horizon):
                    if age + y <= 110:
                        surv *= all_ages.loc[age + y, 'px']
                all_ages.loc[age, f'surv_{horizon}y'] = surv
            else:
                # For very old ages, use exponential extrapolation
                if age < 110:
                    base_surv = all_ages.loc[age, 'px']
                    all_ages.loc[age, f'surv_{horizon}y'] = base_surv ** horizon
    
    return all_ages

# Create complete life table
swiss_lt_complete = create_complete_life_table(swiss_lt_hmd)
print("Created complete life table with interpolated values for all ages")

# ========================================
# CREATE AGE GROUPS
# ========================================

def create_age_groups(df):
    """
    Create 5 age groups for analysis
    """
    age_groups = [
        (0, 55, '<55'),
        (55, 65, '55-64'),
        (65, 75, '65-74'),
        (75, 85, '75-84'),
        (85, 120, '≥85')
    ]
    
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[g[0] for g in age_groups] + [120],
        labels=[g[2] for g in age_groups],
        right=False
    )
    
    # Calculate mean age per group for matching with life tables
    df['age_group_mean'] = df.groupby('age_group')['age'].transform('mean')
    
    return df, age_groups

df, age_groups = create_age_groups(df)
print("\nAge group distribution:")
print(df['age_group'].value_counts().sort_index())

# ========================================
# CALCULATE EXPECTED SURVIVAL USING ACTUAL LIFE TABLE
# ========================================

def calculate_expected_survival_hmd(patient_data, swiss_lt, time_horizon_days):
    """
    Calculate expected survival for patients using actual HMD life table data
    Properly accounts for aging during follow-up
    """
    time_horizon_years = time_horizon_days / 365.25
    expected_survivals = []
    
    for _, patient in patient_data.iterrows():
        baseline_age = int(min(110, patient['age']))
        
        # Calculate survival probability accounting for aging
        survival_prob = 1.0
        
        # Break down into yearly intervals
        full_years = int(time_horizon_years)
        remaining_fraction = time_horizon_years - full_years
        
        # For each full year, apply the age-specific annual survival
        for year in range(full_years):
            current_age = min(baseline_age + year, 110)
            annual_survival = swiss_lt.loc[swiss_lt['age'] == current_age, 'px'].values[0]
            survival_prob *= annual_survival
        
        # For the remaining fraction of a year
        if remaining_fraction > 0 and baseline_age + full_years <= 110:
            current_age = min(baseline_age + full_years, 110)
            annual_survival = swiss_lt.loc[swiss_lt['age'] == current_age, 'px'].values[0]
            # Use exponential model for partial year
            survival_prob *= annual_survival ** remaining_fraction
        
        expected_survivals.append(survival_prob)
    
    return np.array(expected_survivals)

# ========================================
# STANDARDIZED MORTALITY RATIO BY AGE GROUP
# ========================================

def calculate_smr_by_age_group(df, endpoints, swiss_lt):
    """
    Calculate SMR for each age group using actual Swiss life table data
    """
    smr_results = []
    
    for event_col, time_col, endpoint_name in endpoints:
        if event_col not in df.columns or time_col not in df.columns:
            continue
            
        for age_group in df['age_group'].unique():
            if pd.isna(age_group):
                continue
            
            group_data = df[df['age_group'] == age_group].dropna(subset=[event_col, time_col, 'age'])
            group_data[time_col] = pd.to_numeric(group_data[time_col], errors='coerce')
            group_data['age'] = pd.to_numeric(group_data['age'], errors='coerce')
            group_data = group_data.dropna(subset=[time_col, 'age'])
            
            if len(group_data) == 0:
                continue
            
            # Observed deaths
            observed_deaths = group_data[event_col].sum()
            n_patients = len(group_data)
            
            # Calculate expected deaths using actual life table
            median_followup = group_data[time_col].median()
            expected_survivals = calculate_expected_survival_hmd(group_data, swiss_lt, median_followup)
            expected_deaths = n_patients - expected_survivals.sum()
            
            # Calculate SMR
            if expected_deaths > 0:
                smr = observed_deaths / expected_deaths
                
                # 95% CI using exact Poisson method
                if observed_deaths > 0:
                    ci_lower = stats.chi2.ppf(0.025, 2 * observed_deaths) / (2 * expected_deaths)
                    ci_upper = stats.chi2.ppf(0.975, 2 * (observed_deaths + 1)) / (2 * expected_deaths)
                else:
                    ci_lower, ci_upper = 0, stats.chi2.ppf(0.975, 2) / (2 * expected_deaths)
                
                # P-value for SMR = 1
                if observed_deaths >= expected_deaths:
                    p_value = 2 * (1 - stats.poisson.cdf(observed_deaths - 1, expected_deaths))
                else:
                    p_value = 2 * stats.poisson.cdf(observed_deaths, expected_deaths)
                p_value = min(1.0, p_value)
                
                # Get life expectancy at mean age from actual data
                mean_age = int(group_data['age'].mean())
                life_exp = swiss_lt.loc[swiss_lt['age'] == min(mean_age, 110), 'ex'].values[0]
                
                smr_results.append({
                    'Endpoint': endpoint_name,
                    'Age Group': age_group,
                    'N Patients': n_patients,
                    'Mean Age': round(group_data['age'].mean(), 1),
                    'Life Expectancy (Swiss)': round(life_exp, 1),
                    'Median Follow-up (days)': round(median_followup, 0),
                    'Observed Deaths': int(observed_deaths),
                    'Expected Deaths': round(expected_deaths, 1),
                    'SMR': round(smr, 2),
                    'SMR 95% CI': f"({round(ci_lower, 2)}-{round(ci_upper, 2)})",
                    'P-value': round(p_value, 4),
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Interpretation': 'Higher mortality' if smr > 1.2 else ('Similar mortality' if 0.8 <= smr <= 1.2 else 'Lower mortality')
                })
    
    return pd.DataFrame(smr_results)

# ========================================
# ENHANCED KAPLAN-MEIER PLOTS
# ========================================

def plot_km_with_swiss_expected(df, event_col, time_col, endpoint_name, swiss_lt):
    """
    Create enhanced KM plots using actual Swiss life table data
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    age_groups_sorted = ['<55', '55-64', '65-74', '75-84', '≥85']
    colors = plt.cm.Set1(np.linspace(0, 1, 9))
    
    for idx, age_group in enumerate(age_groups_sorted):
        ax = axes[idx]
        
        # Filter data
        group_data = df[df['age_group'] == age_group].dropna(subset=[event_col, time_col, 'age'])
        group_data[time_col] = pd.to_numeric(group_data[time_col], errors='coerce')
        group_data[event_col] = (pd.to_numeric(group_data[event_col], errors='coerce') > 0).astype(int)
        group_data = group_data.dropna(subset=[time_col])
        
        if len(group_data) == 0:
            ax.text(0.5, 0.5, f'No data for age {age_group}', ha='center', va='center')
            ax.set_title(f'Age {age_group}')
            continue
        
        # Observed Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_data[time_col],
            event_observed=group_data[event_col],
            label='Observed (Post-STEMI)'
        )
        kmf.plot_survival_function(ax=ax, color=colors[idx], linewidth=2.5, ci_show=True, ci_alpha=0.2)
        
        # Calculate expected survival from Swiss life table
        mean_age = int(group_data['age'].mean())
        max_time = group_data[time_col].max()
        time_points = np.linspace(0, min(max_time, 365.25 * 10), 100)  # Up to 10 years
        
        expected_survival = []
        for t in time_points:
            days_elapsed = t
            years_elapsed = t / 365.25
            
            # Calculate survival probability from age at baseline to age at time t
            # This accounts for aging during follow-up
            survival_prob = 1.0
            
            # Break down the time into yearly intervals
            full_years = int(years_elapsed)
            remaining_fraction = years_elapsed - full_years
            
            # For each full year, multiply by the annual survival probability at that age
            for year in range(full_years):
                current_age = min(mean_age + year, 110)
                annual_survival = swiss_lt.loc[swiss_lt['age'] == current_age, 'px'].values[0]
                survival_prob *= annual_survival
            
            # For the remaining fraction of a year
            if remaining_fraction > 0 and mean_age + full_years <= 110:
                current_age = min(mean_age + full_years, 110)
                annual_survival = swiss_lt.loc[swiss_lt['age'] == current_age, 'px'].values[0]
                # Use exponential model for partial year
                survival_prob *= annual_survival ** remaining_fraction
            
            expected_survival.append(survival_prob)
        
        # Plot expected survival
        ax.plot(time_points, expected_survival, 
                color='gray', linestyle='--', linewidth=2, alpha=0.7,
                label='Expected (Swiss Population)')
        
        # Add life expectancy annotation
        life_exp = swiss_lt.loc[swiss_lt['age'] == min(mean_age, 110), 'ex'].values[0]
        
        # Calculate SMR for annotation
        observed_deaths = group_data[event_col].sum()
        expected_survivals = calculate_expected_survival_hmd(group_data, swiss_lt, group_data[time_col].median())
        expected_deaths = len(group_data) - expected_survivals.sum()
        
        if expected_deaths > 0:
            smr = observed_deaths / expected_deaths
            # Calculate p-value
            if observed_deaths >= expected_deaths:
                p_value = 2 * (1 - stats.poisson.cdf(observed_deaths - 1, expected_deaths))
            else:
                p_value = 2 * stats.poisson.cdf(observed_deaths, expected_deaths)
            p_value = min(1.0, p_value)
            
            # Add text box with statistics
            textstr = f'n = {len(group_data)}\nMean age: {mean_age:.1f}\nSwiss LE: {life_exp:.1f} years\nSMR: {smr:.2f}\np = {p_value:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.35, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Add at-risk counts
        add_at_risk_counts(kmf, ax=ax, rows_to_show=['At risk'])
        
        # Formatting
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel('Survival Probability', fontsize=11)
        ax.set_title(f'Age {age_group}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, min(max_time * 1.05, 365.25 * 10)])
    
    # Remove empty subplot if needed
    if len(age_groups_sorted) < 6:
        fig.delaxes(axes[5])
    
    plt.suptitle(f'Post-STEMI Survival vs Swiss Population Expected Survival\n{endpoint_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('stats/survival/stemi_swiss_comparison', exist_ok=True)
    filename = f'stats/survival/stemi_swiss_comparison/km_{endpoint_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.show()
    
    return fig

# ========================================
# MAIN ANALYSIS
# ========================================

# Define endpoints
endpoints = [
    ('fup_30d_all_death', 'fup_30d_all_death_time', '30-day all-cause mortality'),
    ('fup_1y_all_death', 'fup_1y_all_death_time', '1-year all-cause mortality'),
    ('fup_5y_all_death', 'fup_5y_all_death_time', '5-year all-cause mortality'),
    ('fup_1y_cvdeath', 'fup_1y_cvdeath_time', '1-year CV mortality'),
    ('fup_5y_cvdeath', 'fup_5y_cvdeath_time', '5-year CV mortality')
]

print("\n" + "="*80)
print("ANALYSIS: POST-STEMI SURVIVAL VS SWISS EXPECTED (HMD 2020-2023)")
print("="*80)

# Display key life expectancy values from actual data
print("\nSwiss Life Expectancy at Key Ages (2020-2023 HMD Data):")
print("-" * 50)
for age in [0, 55, 65, 75, 85]:
    le = swiss_lt_complete.loc[swiss_lt_complete['age'] == age, 'ex'].values[0]
    print(f"Age {age:2d}: {le:.1f} years remaining life expectancy")

# 1. Calculate SMR by age group
print("\n1. Calculating Standardized Mortality Ratios using actual Swiss life table...")
smr_df = calculate_smr_by_age_group(df, endpoints, swiss_lt_complete)

if not smr_df.empty:
    # Save results
    os.makedirs('stats/survival', exist_ok=True)
    smr_df.to_csv('stats/survival/stemi_smr_actual_life_table.csv', index=False)
    print("\nSMR Results (Using Actual Swiss Life Table 2020-2023):")
    print(smr_df[['Endpoint', 'Age Group', 'N Patients', 'Mean Age', 'Life Expectancy (Swiss)', 
                  'Observed Deaths', 'Expected Deaths', 'SMR', 'P-value', 'Interpretation']].to_string())
    
    # Create pivot table for better visualization
    pivot_smr = smr_df.pivot_table(
        values='SMR',
        index='Age Group',
        columns='Endpoint',
        aggfunc='first'
    )
    
    print("\n" + "="*80)
    print("SMR SUMMARY TABLE (SMR > 1 = Higher mortality than Swiss population)")
    print("="*80)
    print(pivot_smr.to_string())
    
    # Identify significant findings
    print("\n" + "="*80)
    print("STATISTICALLY SIGNIFICANT DIFFERENCES FROM EXPECTED MORTALITY")
    print("="*80)
    sig_results = smr_df[smr_df['Significant'] == 'Yes']
    
    if not sig_results.empty:
        for _, row in sig_results.iterrows():
            print(f"\n{row['Endpoint']} - Age {row['Age Group']}:")
            print(f"  SMR = {row['SMR']} {row['SMR 95% CI']} (p = {row['P-value']})")
            print(f"  → {row['Interpretation']} compared to Swiss population")
            print(f"  (Observed: {row['Observed Deaths']} deaths, Expected: {row['Expected Deaths']} deaths)")
    else:
        print("No significant differences from expected mortality found.")

# 2. Create Kaplan-Meier plots
print("\n2. Creating Kaplan-Meier plots with Swiss expected survival curves...")
for event_col, time_col, endpoint_name in endpoints[:3]:  # Focus on main mortality endpoints
    if event_col in df.columns and time_col in df.columns:
        plot_km_with_swiss_expected(df, event_col, time_col, endpoint_name, swiss_lt_complete)

# 3. Generate Clinical Summary Report
print("\n" + "="*80)
print("CLINICAL SUMMARY: POST-STEMI LIFE EXPECTANCY RESTORATION")
print("="*80)

if not smr_df.empty:
    # Analyze by time period
    for time_period in ['30-day', '1-year', '5-year']:
        period_data = smr_df[smr_df['Endpoint'].str.contains(time_period)]
        
        if not period_data.empty:
            print(f"\n{time_period.upper()} OUTCOMES:")
            print("-" * 40)
            
            for _, row in period_data.iterrows():
                if 'all-cause' in row['Endpoint'].lower() or 'mortality' in row['Endpoint'].lower():
                    age_group = row['Age Group']
                    smr = row['SMR']
                    
                    # Interpret SMR
                    if smr < 0.8:
                        status = "✓ Better than expected"
                    elif 0.8 <= smr <= 1.2:
                        status = "≈ Normal life expectancy restored"
                    elif 1.2 < smr <= 2.0:
                        status = "⚠ Moderately increased mortality"
                    else:
                        status = "⚠⚠ Substantially increased mortality"
                    
                    print(f"  Age {age_group}: SMR = {smr:.2f} - {status}")
                    
                    if row['P-value'] < 0.05:
                        print(f"    (Statistically significant, p = {row['P-value']:.3f})")

print("\n" + "="*80)
print("KEY FINDINGS AND CLINICAL IMPLICATIONS")
print("="*80)

print("""
INTERPRETATION GUIDE:
- SMR = 1.0: Mortality equals Swiss general population (normal life expectancy)
- SMR < 1.0: Lower mortality than expected (better outcomes)
- SMR > 1.0: Higher mortality than expected (excess risk persists)

CLINICAL IMPLICATIONS:
1. If SMR approaches 1.0 over time → Successful restoration of life expectancy
2. Persistent SMR > 1.5 → Need for intensified secondary prevention
3. Age-specific patterns → Tailored interventions by age group

SWISS REFERENCE (2020-2023):
- Life expectancy at birth: 84.2 years
- At age 65: 22.2 additional years
- At age 75: 14.3 additional years
- At age 85: 8.2 additional years

This analysis uses official Swiss life table data from the Human Mortality Database,
providing the most accurate comparison to expected survival in the Swiss population.
""")

print("\nAnalysis complete. Results saved to stats/survival/stemi_swiss_comparison/")
print("Generated files:")
print("  - stemi_smr_actual_life_table.csv: SMR analysis with actual Swiss data")
print("  - KM plots: Visual comparison using actual expected survival curves")