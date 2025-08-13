import psycopg2
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, chi2_contingency, fisher_exact

load_dotenv()

# Database connection details
db_name = 'evalfast'
db_user = os.getenv('DB_USER', 'your_username')  # Replace with your PostgreSQL username
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')

def is_categorical(series):
    unique_vals = series.dropna().unique()
    if len(unique_vals) < 5:
        return True
    return False

def format_continuous(series, normal=True):
    series = series.dropna()
    if normal:
        return f"{series.mean():.2f} ± {series.std():.2f}"
    else:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        return f"{series.median():.2f} [{q1:.2f}-{q3:.2f}]"

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

    # SQL query
    query = """
    SELECT 
        is_very_old,
        is_male, 
        is_cx, 
        is_lad, 
        is_rca,
        age, 
        bmi, 
        diabete, 
        dyslip, 
        hta, 
        smoker,
        atcd_mi, 
        atcd_cabg, 
        atcd_pci, 
        way_fmc, 
        is_cancer, 
        atcd_hf, 
        atcd_stroke, 
        atcd_bleeding, 
        pic_ck, 
        pic_ckmb, 
        lvef,
        is_rcv
    FROM 
        psyfast_fup
    WHERE
        1=1
    """

    df = pd.read_sql_query(query, conn)

except Exception as e:
    print("An error occurred:")
    print(e)
finally:
    if 'conn' in locals() and conn:
        conn.close()
        print("\nDatabase connection closed.")

group_vars = ['is_very_old']

# Check which group_vars are valid (have at least 2 categories)
valid_group_vars = []
for gv in group_vars:
    if gv in df.columns:
        if df[gv].dtype.name != 'category':
            df[gv] = df[gv].astype('category')
        if len(df[gv].cat.categories) >= 2:
            valid_group_vars.append(gv)

if len(valid_group_vars) == 0:
    # No valid grouping variable found; produce baseline without p-value and groups
    results = []
    vars_to_analyze = df.columns.tolist()  # Analyze all columns
    
    for var in vars_to_analyze:
        # Skip if var is one of the original grouping vars since they are invalid anyway
        if var in group_vars:
            continue

        series = df[var].dropna()
        if series.empty:
            continue

        if is_categorical(df[var]):
            # Categorical
            contingency = df[var].value_counts(dropna=True)
            # Filter categories ≥ 1 only if numeric
            for cat_val, count in contingency.items():
                try:
                    cat_val_num = float(cat_val)
                    if cat_val_num < 1:
                        continue
                except:
                    # Non-numeric category - skip
                    continue

                n = df[var].notna().sum()
                perc = (count / n) * 100 if n > 0 else np.nan
                results.append({
                    'Variable': f"{var}={cat_val}",
                    'Value': f"{count} ({perc:.1f}%)"
                })
        else:
            # Continuous
            combined = series
            stat, p_shapiro = shapiro(combined)
            if p_shapiro > 0.05:
                desc = format_continuous(series, normal=True)
            else:
                desc = format_continuous(series, normal=False)

            results.append({
                'Variable': var,
                'Value': desc
            })

    no_group_df = pd.DataFrame(results, columns=['Variable', 'Value'])
    no_group_filename = "stats/baseline_characteristics_no_group.csv"
    no_group_df.to_csv(no_group_filename, index=False)
    print(f"Baseline characteristics table (no group) saved as {no_group_filename}")

else:
    # We have at least one valid grouping variable, produce separate files as before
    for gv in valid_group_vars:
        cat0, cat1 = df[gv].cat.categories[0], df[gv].cat.categories[1]
        df_g0 = df[df[gv] == cat0]
        df_g1 = df[df[gv] == cat1]

        results = []
        vars_to_analyze = [c for c in df.columns if c != gv]

        for var in vars_to_analyze:
            data0 = df_g0[var]
            data1 = df_g1[var]

            # Skip if no data in either group
            if data0.dropna().empty or data1.dropna().empty:
                continue

            if is_categorical(df[var]):
                # Categorical variable
                contingency = pd.crosstab(df[gv], df[var])

                # Determine test
                if contingency.shape == (2, 2):
                    if (contingency.values < 5).any():
                        stat, p_value = fisher_exact(contingency)
                    else:
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                else:
                    chi2, p_value, dof, expected = chi2_contingency(contingency)

                # Filter and output only categories with numeric value ≥ 1
                for cat_val in contingency.columns:
                    try:
                        cat_val_num = float(cat_val)
                        if cat_val_num < 1:
                            continue
                    except:
                        # Non-numeric category - skip
                        continue

                    count0 = contingency.loc[cat0, cat_val] if cat0 in contingency.index else 0
                    n0 = df_g0.shape[0]
                    perc0 = (count0 / n0 * 100) if n0 > 0 else np.nan

                    count1 = contingency.loc[cat1, cat_val] if cat1 in contingency.index else 0
                    n1 = df_g1.shape[0]
                    perc1 = (count1 / n1 * 100) if n1 > 0 else np.nan

                    results.append({
                        'Variable': f"{var}={cat_val}",
                        f"{cat0}": f"{count0} ({perc0:.1f}%)",
                        f"{cat1}": f"{count1} ({perc1:.1f}%)",
                        'p-value': f"{p_value:.3g}"
                    })

            else:
                # Continuous variable
                combined = pd.concat([data0.dropna(), data1.dropna()])
                stat, p_shapiro = shapiro(combined)
                if p_shapiro > 0.05:
                    # Normal
                    desc0 = format_continuous(data0, normal=True)
                    desc1 = format_continuous(data1, normal=True)
                    tstat, p_val = ttest_ind(data0.dropna(), data1.dropna(), nan_policy='omit')
                else:
                    # Non-normal
                    desc0 = format_continuous(data0, normal=False)
                    desc1 = format_continuous(data1, normal=False)
                    u_stat, p_val = mannwhitneyu(data0.dropna(), data1.dropna(), alternative='two-sided')

                results.append({
                    'Variable': var,
                    f"{cat0}": desc0,
                    f"{cat1}": desc1,
                    'p-value': f"{p_val:.3g}"
                })

        baseline_df = pd.DataFrame(results, columns=['Variable', f"{cat0}", f"{cat1}", 'p-value'])
        baseline_filename = f"stats/baseline_characteristics_{gv}.csv"
        baseline_df.to_csv(baseline_filename, index=False)
        print(f"Baseline characteristics table saved as {baseline_filename}")