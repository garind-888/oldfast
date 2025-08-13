import psycopg2
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Database connection details
db_name = 'evalfast'
db_user = os.getenv('DB_USER', 'your_username')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')

try:
    # Establish connection to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    print("Successfully connected to the database.")
 
    # SQL query to select the required variables
    query = """
    SELECT pain_to_fmc, fmc_to_diagnosis, diagnosis_to_cathlab, fmc_to_cardio_called_old,
           cathlab_to_balloon, pain_to_balloon, fmc_to_balloon, diagnosis_to_balloon, 
           anxious AS mental_health, psymed, is_psy, is_male, culprit, culprit_field, 
           is_cx, is_lad, is_lm, is_rca, age, is_surpoid, is_obese, is_old, is_very_old, 
           is_very_very_old, bmi, oh, drugs, is_diabetic, dyslip, hta, smoker,
           atcd_mi, atcd_cabg, atcd_pci, is_canicule, way_fmc, is_cancer, is_cirrhosis, 
           atcd_hf, atcd_stroke, atcd_bleeding, family_history, is_ambulance, is_cold,
           pic_ck, pic_ckmb, lvef, fmc_to_balloon_test, is_vulnerable
    FROM psyfast_fup
    WHERE 1=1
    AND is_acr = 0
    """

    # Read the data into a pandas DataFrame
    df = pd.read_sql_query(query, conn)
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # List of time interval variables to analyze
    time_vars = [
        'pain_to_fmc', 
        'fmc_to_diagnosis',
        'diagnosis_to_cathlab',
        'cathlab_to_balloon',
        'diagnosis_to_balloon',
        'fmc_to_balloon',
        'pain_to_balloon'
    ]

    # Continuous variables
    continuous_vars = ['pic_ck', 'pic_ckmb', 'lvef']

    # Main grouping variable (exposure)
    group_vars = ['is_very_old']

    # Confounder sets for sensitivity analysis
    no_confounders = []
    minimal_confounders = ['is_male', 'way_fmc']
    risk_factors_confounders = ['is_diabetic', 'hta', 'smoker', 'dyslip']
    revasc_confounders = ['atcd_mi', 'atcd_pci']
    full_confounders = [
        'is_male', 'bmi', 'way_fmc',
        'is_diabetic', 'hta', 'smoker', 'dyslip',
        'atcd_mi', 'atcd_pci'
    ]

    confounder_sets = {
        'No confounders': no_confounders,
        'Minimal': minimal_confounders,
        'CV risk factors': risk_factors_confounders,
        'CV history': revasc_confounders,
        'Full model': full_confounders
    }

    # Convert time interval variables to minutes
    for var in time_vars:
        if not pd.api.types.is_numeric_dtype(df[var]):
            df[var] = pd.to_timedelta(df[var], errors='coerce')
            df[var] = df[var].dt.total_seconds() / 60.0

    # Convert categorical variables
    categorical_vars = [
        'is_cx', 'is_psy', 'mental_health', 'psymed', 'is_male', 'drugs', 
        'is_diabetic','is_canicule', 'is_cancer', 'is_cirrhosis', 'atcd_hf', 
        'atcd_stroke', 'atcd_bleeding', 'is_ambulance', 'dyslip', 'hta', 
        'smoker', 'atcd_mi', 'atcd_cabg', 'atcd_pci', 'is_surpoid', 'is_obese', 
        'is_old', 'is_very_old', 'is_very_very_old', 'way_fmc', 'is_cold'
    ]
    
    for var in categorical_vars:
        if var in df.columns:
            df[var] = df[var].astype('category')

    # --------------------------
    # Results containers
    # --------------------------
    descriptive_stats = []
    summary_results = []
    sensitivity_results = []
    glm_full_results = []

    # --------------------------
    # Main analysis loop
    # --------------------------
    print("\n=== Starting Analysis with Drop NA Approach ===")

    for group_var in group_vars:
        print(f"\nProcessing group variable: '{group_var}'")

        # Get unique groups
        if df[group_var].dtype.name == 'category':
            groups = df[group_var].cat.categories
        else:
            groups = df[group_var].unique()

        if len(groups) < 2:
            print(f"Skipping '{group_var}' - insufficient unique categories.")
            continue

        # Process each outcome variable
        for outcome_var in time_vars + continuous_vars:
            print(f"  Analyzing outcome: '{outcome_var}'")

            # Prepare columns needed for analysis
            cols_needed = list(set([outcome_var, group_var] + full_confounders))
            
            # Drop NA for required columns
            df_model = df[cols_needed].dropna()

            # Check if both groups still have data
            if df_model[group_var].nunique() < 2:
                print(f"    Insufficient data after dropping NA.")
                continue

            # Calculate descriptive statistics by group
            df_temp = df_model.copy()
            df_temp[group_var] = df_temp[group_var].astype(str)
            
            group_stats = df_temp.groupby(group_var)[outcome_var].agg([
                ('n', 'count'),
                ('mean', 'mean'),
                ('sd', 'std'),
                ('median', 'median'),
                ('q1', lambda x: x.quantile(0.25)),
                ('q3', lambda x: x.quantile(0.75)),
                ('min', 'min'),
                ('max', 'max')
            ])

            if len(group_stats) < 2:
                print(f"    Not enough groups to compare.")
                continue

            # Extract group labels
            g0, g1 = str(groups[0]), str(groups[1])
            
            if g0 not in group_stats.index or g1 not in group_stats.index:
                print(f"    Group labels missing in data.")
                continue

            # Calculate 95% CIs for means
            for g in [g0, g1]:
                n = group_stats.loc[g, 'n']
                mean = group_stats.loc[g, 'mean']
                sd = group_stats.loc[g, 'sd']
                sem = sd / np.sqrt(n) if n > 0 else np.nan
                
                # Store descriptive statistics
                descriptive_stats.append({
                    'Group_Variable': group_var,
                    'Outcome': outcome_var,
                    'Group': f"{group_var}={g}",
                    'N': int(n),
                    'Mean': round(mean, 2),
                    'SD': round(sd, 2),
                    'SEM': round(sem, 3),
                    '95%_CI_Lower': round(mean - 1.96 * sem, 2) if not np.isnan(sem) else np.nan,
                    '95%_CI_Upper': round(mean + 1.96 * sem, 2) if not np.isnan(sem) else np.nan,
                    'Median': round(group_stats.loc[g, 'median'], 2),
                    'Q1': round(group_stats.loc[g, 'q1'], 2),
                    'Q3': round(group_stats.loc[g, 'q3'], 2),
                    'Min': round(group_stats.loc[g, 'min'], 2),
                    'Max': round(group_stats.loc[g, 'max'], 2)
                })

            # Variables to store unadjusted and adjusted results
            unadjusted_p = None
            unadjusted_coef = None
            unadjusted_ci_lower = None
            unadjusted_ci_upper = None
            
            adjusted_p = None
            adjusted_coef = None
            adjusted_ci_lower = None
            adjusted_ci_upper = None

            # Run GLM models with different confounder sets
            for conf_label, confounders in confounder_sets.items():
                # Build formula
                if confounders:
                    formula = f"{outcome_var} ~ C({group_var}) + " + " + ".join(confounders)
                else:
                    formula = f"{outcome_var} ~ C({group_var})"

                try:
                    model = smf.glm(formula=formula, data=df_model, family=sm.families.Gaussian())
                    result = model.fit()
                    
                    # Extract coefficient for group difference
                    param_names = [name for name in result.params.index if f"C({group_var})" in name]
                    
                    if not param_names:
                        print(f"    Parameter for '{group_var}' not found in {conf_label} model.")
                        continue

                    param_name = param_names[0]
                    coef = result.params[param_name]
                    conf_int = result.conf_int().loc[param_name]
                    p_value = result.pvalues[param_name]

                    # Store unadjusted (no confounders) results
                    if conf_label == 'No confounders':
                        unadjusted_p = p_value
                        unadjusted_coef = coef
                        unadjusted_ci_lower = conf_int[0]
                        unadjusted_ci_upper = conf_int[1]
                    
                    # Store fully adjusted results
                    if conf_label == 'Full model':
                        adjusted_p = p_value
                        adjusted_coef = coef
                        adjusted_ci_lower = conf_int[0]
                        adjusted_ci_upper = conf_int[1]

                    # Store sensitivity analysis results
                    sensitivity_results.append({
                        'Group_Variable': group_var,
                        'Outcome': outcome_var,
                        'Model': conf_label,
                        'N_Total': len(df_model),
                        'Coefficient': round(coef, 3),
                        '95%_CI_Lower': round(conf_int[0], 3),
                        '95%_CI_Upper': round(conf_int[1], 3),
                        'SE': round(result.bse[param_name], 3),
                        'P_value': round(p_value, 4) if p_value >= 0.0001 else '<0.0001',
                        'AIC': round(result.aic, 1),
                        'BIC': round(result.bic, 1)
                    })

                    # Store full GLM output for full model only
                    if conf_label == 'Full model':
                        glm_full_results.append({
                            'Group_Variable': group_var,
                            'Outcome': outcome_var,
                            'Formula': formula,
                            'Summary': result.summary().as_text()
                        })

                except Exception as e:
                    print(f"    GLM failed for {conf_label}: {e}")
                    continue

            # Store summary results with both unadjusted and adjusted p-values
            if unadjusted_p is not None and adjusted_p is not None:
                # Get means for both groups
                mean_g0 = group_stats.loc[g0, 'mean']
                sd_g0 = group_stats.loc[g0, 'sd']
                mean_g1 = group_stats.loc[g1, 'mean']
                sd_g1 = group_stats.loc[g1, 'sd']
                
                summary_results.append({
                    'Group_Variable': group_var,
                    'Outcome': outcome_var,
                    f'Mean_{g0}': round(mean_g0, 2),
                    f'SD_{g0}': round(sd_g0, 2),
                    f'Mean_{g1}': round(mean_g1, 2),
                    f'SD_{g1}': round(sd_g1, 2),
                    'Unadj_Diff': round(unadjusted_coef, 3),
                    'Unadj_95CI_Lower': round(unadjusted_ci_lower, 3),
                    'Unadj_95CI_Upper': round(unadjusted_ci_upper, 3),
                    'Unadj_P_value': round(unadjusted_p, 4) if unadjusted_p >= 0.0001 else '<0.0001',
                    'Adj_Diff': round(adjusted_coef, 3),
                    'Adj_95CI_Lower': round(adjusted_ci_lower, 3),
                    'Adj_95CI_Upper': round(adjusted_ci_upper, 3),
                    'Adj_P_value': round(adjusted_p, 4) if adjusted_p >= 0.0001 else '<0.0001'
                })

    # --------------------------
    # Create organized output DataFrames
    # --------------------------
    
    # 1. Descriptive Statistics Table
    if descriptive_stats:
        df_descriptive = pd.DataFrame(descriptive_stats)
        # Sort for better readability
        df_descriptive = df_descriptive.sort_values(['Group_Variable', 'Outcome', 'Group'])
        
        # Save with proper formatting
        df_descriptive.to_csv('stats/glm/descriptive_statistics.csv', index=False, float_format='%.3f')
        print("\n✓ Saved descriptive_statistics.csv")

    # 2. Results Summary Table (with unadjusted and adjusted p-values)
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        # Sort for better readability
        df_summary = df_summary.sort_values(['Group_Variable', 'Outcome'])
        
        # Save with proper formatting
        df_summary.to_csv('stats/glm/results_summary.csv', index=False, float_format='%.4f')
        print("✓ Saved results_summary.csv")

    # 3. Sensitivity Analysis Results (all models)
    if sensitivity_results:
        df_sensitivity = pd.DataFrame(sensitivity_results)
        # Sort by group variable, outcome, and model order
        model_order = {'No confounders': 0, 'Minimal': 1, 'CV risk factors': 2, 'CV history': 3, 'Full model': 4}
        df_sensitivity['Model_Order'] = df_sensitivity['Model'].map(model_order)
        df_sensitivity = df_sensitivity.sort_values(['Group_Variable', 'Outcome', 'Model_Order'])
        df_sensitivity = df_sensitivity.drop('Model_Order', axis=1)
        
        # Save
        df_sensitivity.to_csv('stats/glm/sensitivity_analysis.csv', index=False, float_format='%.4f')
        print("✓ Saved sensitivity_analysis.csv")

    # 4. Full GLM outputs
    if glm_full_results:
        with open('stats/glm/glm_full_model_outputs.txt', 'w') as f:
            for res in glm_full_results:
                f.write("="*80 + "\n")
                f.write(f"Group Variable: {res['Group_Variable']}\n")
                f.write(f"Outcome: {res['Outcome']}\n")
                f.write(f"Formula: {res['Formula']}\n")
                f.write("-"*80 + "\n")
                f.write(res['Summary'])
                f.write("\n\n")
        print("✓ Saved glm_full_model_outputs.txt")

    print(f"\n=== Analysis Complete ===")
    print(f"Total models run: {len(sensitivity_results)}")
    print(f"Outcomes analyzed: {len(set(s['Outcome'] for s in sensitivity_results))}")

except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'conn' in locals() and conn:
        conn.close()
        print("\nDatabase connection closed.")