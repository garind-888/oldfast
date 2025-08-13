import psycopg2
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from dotenv import load_dotenv

import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test, logrank_test

# Load environment variables
load_dotenv()

# Database connection details
db_name = 'evalfast'
db_user = os.getenv('DB_USER', 'your_username')  # Remplacez par votre identifiant PostgreSQL
db_password = os.getenv('DB_PASSWORD')           # Assurez-vous que cette variable est définie dans votre .env
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')

try:
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    print("Successfully connected to the database.")

    query = """
        SELECT 
             anxious, 
             is_male, 
             fup_30d_cvdeath, fup_30d_cvdeath_time,
            fup_90d_cvdeath, fup_90d_cvdeath_time,
            fup_1y_cvdeath,  fup_1y_cvdeath_time,

            
            
            is_acr,
            
            nb_vessels_disease,
            did_vf_occur,
            
            is_rcv,
            thrombectomy,
            fmc_to_diagnosis,
            kilip,
             culprit, 
             culprit_field, 
             is_cx, 
             is_lad, 
             is_lm, 
             is_rca,
             age, 
             bmi, 
             oh, 
             drugs, 
             diabete, 
             dyslip, 
             hta, 
             smoker,
             atcd_mi, 
             atcd_cabg, 
             atcd_pci, 
             is_diabetic, 
             way_fmc, 
             atcd_hf, 
             atcd_stroke, 
             
             atcd_bleeding, 
             family_history, 
             is_ambulance, 
             pic_ck, 
             pic_ckmb, 
             lvef,
             fup_1y_mace4,
             fup_1y_mace4_time,
             fup_5y_mace4,
             fup_5y_mace4_time,
             fup_1y_all_death, fup_1y_all_death_time, 
             fup_5y_all_death, fup_5y_all_death_time, 
             fup_1y_bleed, fup_1y_bleed_time, 
             fup_5y_bleed, fup_5y_bleed_time, 
             fup_1y_stroke, fup_1y_stroke_time, 
             fup_5y_stroke, fup_5y_stroke_time, 
             fup_1y_revasc, fup_1y_revasc_time, 
             fup_5y_revasc, fup_5y_revasc_time, 
             fup_1y_mi, fup_1y_mi_time, 
             fup_5y_mi, fup_5y_mi_time, 
             fup_1y_cvdeath, fup_1y_cvdeath_time, 
             fup_5y_cvdeath, fup_5y_cvdeath_time, 
             fup_1y_other_death, fup_1y_other_death_time, 
             fup_5y_other_death, fup_5y_other_death_time
        FROM 
            psyfast_fup
        WHERE 
            1=1
            AND culprit_field in (1, 2, 3, 4)
            AND atcd_cabg = 0 OR atcd_cabg IS NULL
            AND fup_30d_cvdeath = 0
    """

    df = pd.read_sql_query(query, conn)

except Exception as e:
    print("An error occurred:")
    print(e)
finally:
    if 'conn' in locals() and conn:
        conn.close()
        print("\nDatabase connection closed.")


##########################################
# Conversion des variables catégorielles
##########################################
categorical_vars = [
    'is_cx', 'is_lad', 'is_lm', 'is_rca',
    'anxious', 'is_male', 'oh', 'drugs', 'is_diabetic', 'atcd_hf',
    'atcd_stroke', 'atcd_bleeding', 'family_history', 'is_ambulance', 
    'diabete', 'dyslip', 'hta', 'smoker', 
    'atcd_mi', 'atcd_cabg', 'atcd_pci',
    'way_fmc',
    
]
for var in categorical_vars:
    if var in df.columns:
        df[var] = df[var].astype('category')


########################
# Définition des issues
########################
survival_endpoints = [
    ('fup_5y_cvdeath', 'fup_5y_cvdeath_time', '5-year cardiovascular death'),
    #('fup_5y_other_death', 'fup_5y_other_death_time', '5-year non-cardiovascular death'),
    #('fup_5y_mace4', 'fup_5y_mace4_time', '5-year MACCE'),
    #('fup_5y_stroke', 'fup_5y_stroke_time', '5-year stroke'),
    #('fup_5y_revasc', 'fup_5y_revasc_time', '5-year any revascularisation'),
    #('fup_5y_mi', 'fup_5y_mi_time', '5-year myocardial infarction'),
]

group_vars = ['is_cx']

############################
# Définition des confondeurs
############################
full_confounders = [
   'is_acr',
   'dyslip',
   'nb_vessels_disease',
   'lvef',
   'smoker',
   'atcd_mi',
   'bmi',
   
   'atcd_pci',
   
   'is_diabetic',

   'is_rcv',
  # 'thrombectomy',
   #'fmc_to_diagnosis',
   #'kilip'
]

no_confounders = ['lvef']


confounder_sets = {
    'No Confounders': no_confounders,
    'Full Confounders': full_confounders
}

# Ajout des analyses de sensibilité individuelles
individual_confounders = [
    'is_acr',
    'dyslip',
    'nb_vessels_disease',
    'lvef',
    'smoker',
    'atcd_stroke',
    'atcd_mi',
    'bmi',
    'atcd_pci',
    'is_diabetic',
    'is_rcv',
    'thrombectomy',
    'kilip'
]

# Création des sets de confounders individuels
for conf in individual_confounders:
    confounder_sets[f'is_cx + {conf}'] = [conf]

results_list = []

###############################################################################
#           5-Year Survival Sensitivity Analysis
###############################################################################
for group_var in group_vars:
    for event_col, time_col, endpoint_name in survival_endpoints:
        # 1) On dropna sur group_var, event_col et time_col puis copy()
        df_surv = df.dropna(subset=[group_var, event_col, time_col]).copy()
        
        # 2) Pour l'analyse, on va créer time/event ici :
        df_surv['time'] = df_surv[time_col]
        df_surv['event'] = df_surv[event_col]
        
        # Vérifions qu'il y a bien au moins 2 groupes différents
        if df_surv[group_var].nunique() < 2:
            print(f"Not enough groups for {group_var} in endpoint {endpoint_name}.")
            continue
        
        # 3) On parcourt les différents sets de confounders
        for label, confs in confounder_sets.items():
            covariates = [group_var] + confs
            
            # Vérifions que toutes ces covariates existent dans df_surv
            missing_covars = [c for c in covariates if c not in df_surv.columns]
            if missing_covars:
                print(f"Missing covariates {missing_covars} in the data, skipping {label} model for {endpoint_name} (5-year).")
                continue
            
            # 4) Construction de df_model : on ne garde que les colonnes nécessaires
            relevant_columns = ['time', 'event'] + covariates
            df_model = df_surv[relevant_columns].copy()
            
            # On supprime toute ligne qui contient un NaN parmi ces colonnes
            df_model.dropna(inplace=True)
            
            # Après suppression des NaNs, vérifions à nouveau le nombre de groupes
            if df_model[group_var].nunique() < 2:
                print(f"Not enough data for both {group_var} groups after dropna in {label} model for {endpoint_name} (5-year).")
                continue
            
            cph = CoxPHFitter()
            try:
                cph.fit(df_model, duration_col='time', event_col='event')
            except ConvergenceError:
                results_list.append({
                    'Endpoint': endpoint_name,
                    'Model': label,
                    'Exposure': group_var,
                    'HR': 'convergence halted',
                    'HR Lower 95%CI': 'convergence halted',
                    'HR Upper 95%CI': 'convergence halted',
                    'P-value': 'convergence halted',
                    'Timeframe': '5-year'
                })
                print(f"Convergence halted for {group_var} in {endpoint_name} with {label} model (5-year).")
                continue
            
            # --- Test de l'hypothèse des hasards proportionnels ---
            try:
                ph_test_results = proportional_hazard_test(cph, df_model, time_transform='rank')
                print(f"\n[PH Test Results: {endpoint_name}, Model: {label}, Exposure: {group_var}, 5-year]")
                print(ph_test_results.summary)
            except Exception as ex:
                print(f"PH test error for {endpoint_name}, {label}, {group_var}:\n{ex}")
            
            # Extraction des résultats de la Cox
            if group_var in cph.summary.index:
                hr = cph.summary.loc[group_var, "exp(coef)"]
                lower_ci = cph.summary.loc[group_var, "exp(coef) lower 95%"]
                upper_ci = cph.summary.loc[group_var, "exp(coef) upper 95%"]
                p_value = cph.summary.loc[group_var, "p"]
            else:
                # Si jamais la variable prend la forme "anxious=1" dans le summary
                param_names = [idx for idx in cph.summary.index if group_var in idx]
                if len(param_names) == 0:
                    print(f"No parameter found for {group_var} in {label} model for {endpoint_name} (5-year).")
                    continue
                param_name = param_names[0]
                hr = cph.summary.loc[param_name, "exp(coef)"]
                lower_ci = cph.summary.loc[param_name, "exp(coef) lower 95%"]
                upper_ci = cph.summary.loc[param_name, "exp(coef) upper 95%"]
                p_value = cph.summary.loc[param_name, "p"]
            
            # Stockage des résultats
            results_list.append({
                'Endpoint': endpoint_name,
                'Model': label,
                'Exposure': group_var,
                'HR': hr,
                'HR Lower 95%CI': lower_ci,
                'HR Upper 95%CI': upper_ci,
                'P-value': p_value,
                'Timeframe': '5-year'
            })
            
            ######################################################
            #  Production des KM plots pour le modèle Full Confounders
            ######################################################
            if label == "Full Confounders":
                # KM sur df_model
                kmf_list = []
                groups_in_data = df_model[group_var].unique()

                # 1) Récupération de la KM pour chaque groupe
                for grp in groups_in_data:
                    mask = (df_model[group_var] == grp)
                    kmf = KaplanMeierFitter()
                    legend_label = "LCx" if grp == 1 else "LAD or RCA"
                    kmf.fit(
                        durations=df_model.loc[mask, 'time'],
                        event_observed=df_model.loc[mask, 'event'],
                        label=legend_label
                    )
                    kmf_list.append((kmf, legend_label))

                #=============================================================
                # PLOT 1: Y-axis (0 => 1) + table "at risk"
                #=============================================================
                fig, ax = plt.subplots(figsize=(10, 8))
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)

                ax.set_xlabel("Time (days)", fontsize=12)
                ax.set_ylabel("Cumulative probability of event", fontsize=12)
                ax.set_ylim(0, 1)
                add_at_risk_counts(*[kmf for kmf, _ in kmf_list], ax=ax)
                plt.tight_layout()
                ax.legend().remove()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/{endpoint_name}_full_confounders.png", dpi=300)

                #=============================================================
                # PLOT 2: Zoom (0 => 10% ou max) + annotation du HR
                #=============================================================
                fig, ax = plt.subplots(figsize=(7, 5))
                max_prob = 0
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)
                    group_max = kmf.cumulative_density_.iloc[:, 0].max()
                    max_prob = max(max_prob, group_max)

                upper_limit = max(0.1, max_prob) * 1.1
                ax.set_ylim(0, upper_limit)

                # Calcul du test de log-rank
                results = logrank_test(
                    df_model.loc[df_model[group_var] == 1, 'time'],
                    df_model.loc[df_model[group_var] == 0, 'time'],
                    df_model.loc[df_model[group_var] == 1, 'event'],
                    df_model.loc[df_model[group_var] == 0, 'event']
                )
                logrank_p = results.p_value

                hr_text = (
                    f"Adjusted hazard ratio: {hr:.2f}\n"
                    f"95% CI {lower_ci:.2f} to {upper_ci:.2f}; p{'<0.001' if p_value < 0.001 else f'={p_value:.3f}'}\n"
                    f"Log-rank p{'<0.001' if logrank_p < 0.001 else f'={logrank_p:.3f}'}"
                )
                ax.text(
                    0.5, 0.30, hr_text,
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6)
                )

                ax.legend().remove()
                plt.tight_layout()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/{endpoint_name}_full_confounders_zoomed.png", dpi=300)


###############################################################################
#           30-Day Truncated Survival Sensitivity Analysis
###############################################################################
for group_var in group_vars:
    for event_col, time_col, endpoint_name in survival_endpoints:
        # On part de df, on dropna sur group_var, event_col et time_col
        df_surv = df.dropna(subset=[group_var, event_col, time_col]).copy()
        
        # Vérifions qu'il y a au moins 2 groupes
        if df_surv[group_var].nunique() < 2:
            print(f"Not enough groups for {group_var} in endpoint {endpoint_name} (30-day).")
            continue
        
        # On crée des colonnes "time" et "event" tronquées à 30 jours
        df_surv['time'] = np.where(df_surv[time_col] > 30, 30, df_surv[time_col])
        df_surv['event'] = np.where((df_surv[time_col] <= 30) & (df_surv[event_col] == 1), 1, 0)

        for label, confs in confounder_sets.items():
            covariates = [group_var] + confs
            
            missing_covars = [c for c in covariates if c not in df_surv.columns]
            if missing_covars:
                print(f"Missing covariates {missing_covars} in {label} model for 30-day {endpoint_name}.")
                continue
            
            relevant_columns = ['time', 'event'] + covariates
            df_model = df_surv[relevant_columns].copy()
            df_model.dropna(inplace=True)

            if df_model[group_var].nunique() < 2:
                print(f"Not enough data for both {group_var} groups after dropna in {label} model for {endpoint_name} (30-day).")
                continue

            cph_30d = CoxPHFitter()
            try:
                cph_30d.fit(df_model, duration_col='time', event_col='event')
            except ConvergenceError:
                results_list.append({
                    'Endpoint': endpoint_name,
                    'Model': label,
                    'Exposure': group_var,
                    'HR': 'convergence halted',
                    'HR Lower 95%CI': 'convergence halted',
                    'HR Upper 95%CI': 'convergence halted',
                    'P-value': 'convergence halted',
                    'Timeframe': '30-day'
                })
                print(f"Convergence halted for {group_var} in {endpoint_name} with {label} model (30-day).")
                continue

            # Test PH
            try:
                ph_test_results_30d = proportional_hazard_test(cph_30d, df_model, time_transform='rank')
                print(f"\n[PH Test Results: {endpoint_name}, Model: {label}, Exposure: {group_var}, 30-day]")
                print(ph_test_results_30d.summary)
            except Exception as ex:
                print(f"PH test error for {endpoint_name}, {label}, {group_var} (30-day):\n{ex}")

            # Extraction des résultats
            if group_var in cph_30d.summary.index:
                hr = cph_30d.summary.loc[group_var, "exp(coef)"]
                lower_ci = cph_30d.summary.loc[group_var, "exp(coef) lower 95%"]
                upper_ci = cph_30d.summary.loc[group_var, "exp(coef) upper 95%"]
                p_value = cph_30d.summary.loc[group_var, "p"]
            else:
                param_names = [idx for idx in cph_30d.summary.index if group_var in idx]
                if len(param_names) == 0:
                    print(f"No parameter found for {group_var} in {label} model for {endpoint_name} (30-day).")
                    continue
                param_name = param_names[0]
                hr = cph_30d.summary.loc[param_name, "exp(coef)"]
                lower_ci = cph_30d.summary.loc[param_name, "exp(coef) lower 95%"]
                upper_ci = cph_30d.summary.loc[param_name, "exp(coef) upper 95%"]
                p_value = cph_30d.summary.loc[param_name, "p"]

            results_list.append({
                'Endpoint': endpoint_name,
                'Model': label,
                'Exposure': group_var,
                'HR': hr,
                'HR Lower 95%CI': lower_ci,
                'HR Upper 95%CI': upper_ci,
                'P-value': p_value,
                'Timeframe': '30-day'
            })

            ######################################################
            #  KM plots (Full Confounders) 
            ######################################################
            if label == "Full Confounders":
                kmf_list = []
                groups_in_data = df_model[group_var].unique()

                for grp in groups_in_data:
                    mask = (df_model[group_var] == grp)
                    kmf = KaplanMeierFitter()
                    legend_label = "LCx" if grp == 1 else "LAD or RCA"
                    kmf.fit(
                        durations=df_model.loc[mask, 'time'],
                        event_observed=df_model.loc[mask, 'event'],
                        label=legend_label
                    )
                    kmf_list.append((kmf, legend_label))

                #=============================================================
                # PLOT 1: Y-axis (0 => 1) + at_risk
                #=============================================================
                fig, ax = plt.subplots(figsize=(10, 8))
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)

                ax.set_xlabel("Time (days)", fontsize=12)
                ax.set_ylabel("Cumulative probability of event", fontsize=12)
                ax.set_ylim(0, 1)
                add_at_risk_counts(*[kmf for kmf, _ in kmf_list], ax=ax)
                plt.tight_layout()
                ax.legend().remove()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/30_{endpoint_name}_30day_full_confounders.png", dpi=300)

                #=============================================================
                # PLOT 2: Zoom (0 => 10% ou max) + annotation HR
                #=============================================================
                fig, ax = plt.subplots(figsize=(7, 5))
                max_prob = 0
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)
                    group_max = kmf.cumulative_density_.iloc[:, 0].max()
                    max_prob = max(max_prob, group_max)

                upper_limit = max(0.1, max_prob) * 1.1
                ax.set_ylim(0, upper_limit)

                # Calcul du test de log-rank
                results = logrank_test(
                    df_model.loc[df_model[group_var] == 1, 'time'],
                    df_model.loc[df_model[group_var] == 0, 'time'],
                    df_model.loc[df_model[group_var] == 1, 'event'],
                    df_model.loc[df_model[group_var] == 0, 'event']
                )
                logrank_p = results.p_value

                hr_text = (
                    f"Adjusted hazard ratio: {hr:.2f}\n"
                    f"95% CI {lower_ci:.2f} to {upper_ci:.2f}; p{'<0.001' if p_value < 0.001 else f'={p_value:.3g}'}\n"
                    f"Log-rank p{'<0.001' if logrank_p < 0.001 else f'={logrank_p:.3f}'}"
                )
                ax.text(
                    0.05, 0.90, hr_text,
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6)
                )

                ax.legend().remove()
                plt.tight_layout()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/30_d{endpoint_name}_30day_full_confounders_zoomed.png", dpi=300)


###############################################################################
#           90-Day Truncated Survival Sensitivity Analysis
###############################################################################
for group_var in group_vars:
    for event_col, time_col, endpoint_name in survival_endpoints:
        # On part de df, on dropna sur group_var, event_col et time_col
        df_surv = df.dropna(subset=[group_var, event_col, time_col]).copy()
        
        # Vérifions qu'il y a au moins 2 groupes
        if df_surv[group_var].nunique() < 2:
            print(f"Not enough groups for {group_var} in endpoint {endpoint_name} (90-day).")
            continue
        
        # On crée des colonnes "time" et "event" tronquées à 90 jours
        df_surv['time'] = np.where(df_surv[time_col] > 90, 90, df_surv[time_col])
        df_surv['event'] = np.where((df_surv[time_col] <= 90) & (df_surv[event_col] == 1), 1, 0)

        for label, confs in confounder_sets.items():
            covariates = [group_var] + confs
            
            missing_covars = [c for c in covariates if c not in df_surv.columns]
            if missing_covars:
                print(f"Missing covariates {missing_covars} in {label} model for 90-day {endpoint_name}.")
                continue
            
            relevant_columns = ['time', 'event'] + covariates
            df_model = df_surv[relevant_columns].copy()
            df_model.dropna(inplace=True)

            if df_model[group_var].nunique() < 2:
                print(f"Not enough data for both {group_var} groups after dropna in {label} model for {endpoint_name} (90-day).")
                continue

            cph_90d = CoxPHFitter()
            try:
                cph_90d.fit(df_model, duration_col='time', event_col='event')
            except ConvergenceError:
                results_list.append({
                    'Endpoint': endpoint_name,
                    'Model': label,
                    'Exposure': group_var,
                    'HR': 'convergence halted',
                    'HR Lower 95%CI': 'convergence halted',
                    'HR Upper 95%CI': 'convergence halted',
                    'P-value': 'convergence halted',
                    'Timeframe': '90-day'
                })
                print(f"Convergence halted for {group_var} in {endpoint_name} with {label} model (90-day).")
                continue

            # Test PH
            try:
                ph_test_results_90d = proportional_hazard_test(cph_90d, df_model, time_transform='rank')
                print(f"\n[PH Test Results: {endpoint_name}, Model: {label}, Exposure: {group_var}, 90-day]")
                print(ph_test_results_90d.summary)
            except Exception as ex:
                print(f"PH test error for {endpoint_name}, {label}, {group_var} (90-day):\n{ex}")

            # Extraction des résultats
            if group_var in cph_90d.summary.index:
                hr = cph_90d.summary.loc[group_var, "exp(coef)"]
                lower_ci = cph_90d.summary.loc[group_var, "exp(coef) lower 95%"]
                upper_ci = cph_90d.summary.loc[group_var, "exp(coef) upper 95%"]
                p_value = cph_90d.summary.loc[group_var, "p"]
            else:
                param_names = [idx for idx in cph_90d.summary.index if group_var in idx]
                if len(param_names) == 0:
                    print(f"No parameter found for {group_var} in {label} model for {endpoint_name} (90-day).")
                    continue
                param_name = param_names[0]
                hr = cph_90d.summary.loc[param_name, "exp(coef)"]
                lower_ci = cph_90d.summary.loc[param_name, "exp(coef) lower 95%"]
                upper_ci = cph_90d.summary.loc[param_name, "exp(coef) upper 95%"]
                p_value = cph_90d.summary.loc[param_name, "p"]

            results_list.append({
                'Endpoint': endpoint_name,
                'Model': label,
                'Exposure': group_var,
                'HR': hr,
                'HR Lower 95%CI': lower_ci,
                'HR Upper 95%CI': upper_ci,
                'P-value': p_value,
                'Timeframe': '90-day'
            })

            ######################################################
            #  KM plots (Full Confounders) 
            ######################################################
            if label == "Full Confounders":
                kmf_list = []
                groups_in_data = df_model[group_var].unique()

                for grp in groups_in_data:
                    mask = (df_model[group_var] == grp)
                    kmf = KaplanMeierFitter()
                    legend_label = "LCx" if grp == 1 else "LAD or RCA"
                    kmf.fit(
                        durations=df_model.loc[mask, 'time'],
                        event_observed=df_model.loc[mask, 'event'],
                        label=legend_label
                    )
                    kmf_list.append((kmf, legend_label))

                #=============================================================
                # PLOT 1: Y-axis (0 => 1) + at_risk
                #=============================================================
                fig, ax = plt.subplots(figsize=(10, 8))
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)

                ax.set_xlabel("Time (days)", fontsize=12)
                ax.set_ylabel("Cumulative probability of event", fontsize=12)
                ax.set_ylim(0, 1)
                add_at_risk_counts(*[kmf for kmf, _ in kmf_list], ax=ax)
                plt.tight_layout()
                ax.legend().remove()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/90_{endpoint_name}_90day_full_confounders.png", dpi=300)

                #=============================================================
                # PLOT 2: Zoom (0 => 10% ou max) + annotation HR
                #=============================================================
                fig, ax = plt.subplots(figsize=(7, 5))
                max_prob = 0
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)
                    group_max = kmf.cumulative_density_.iloc[:, 0].max()
                    max_prob = max(max_prob, group_max)

                upper_limit = max(0.1, max_prob) * 1.1
                ax.set_ylim(0, upper_limit)

                # Calcul du test de log-rank
                results = logrank_test(
                    df_model.loc[df_model[group_var] == 1, 'time'],
                    df_model.loc[df_model[group_var] == 0, 'time'],
                    df_model.loc[df_model[group_var] == 1, 'event'],
                    df_model.loc[df_model[group_var] == 0, 'event']
                )
                logrank_p = results.p_value

                hr_text = (
                    f"Adjusted hazard ratio: {hr:.2f}\n"
                    f"95% CI {lower_ci:.2f} to {upper_ci:.2f}; p{'<0.001' if p_value < 0.001 else f'={p_value:.3f}'}\n"
                    f"Log-rank p{'<0.001' if logrank_p < 0.001 else f'={logrank_p:.3f}'}"
                )
                ax.text(
                    0.5, 0.20, hr_text,
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6)
                )

                ax.legend().remove()
                plt.tight_layout()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/90_{endpoint_name}_90day_full_confounders_zoomed.png", dpi=300)

###############################################################################
#           365-Day Truncated Survival Sensitivity Analysis
###############################################################################
for group_var in group_vars:
    for event_col, time_col, endpoint_name in survival_endpoints:
        # On part de df, on dropna sur group_var, event_col et time_col
        df_surv = df.dropna(subset=[group_var, event_col, time_col]).copy()
        
        # Vérifions qu'il y a au moins 2 groupes
        if df_surv[group_var].nunique() < 2:
            print(f"Not enough groups for {group_var} in endpoint {endpoint_name} (365-day).")
            continue
        
        # On crée des colonnes "time" et "event" tronquées à 365 jours
        df_surv['time'] = np.where(df_surv[time_col] > 365, 365, df_surv[time_col])
        df_surv['event'] = np.where((df_surv[time_col] <= 365) & (df_surv[event_col] == 1), 1, 0)

        for label, confs in confounder_sets.items():
            covariates = [group_var] + confs
            
            missing_covars = [c for c in covariates if c not in df_surv.columns]
            if missing_covars:
                print(f"Missing covariates {missing_covars} in {label} model for 365-day {endpoint_name}.")
                continue
            
            relevant_columns = ['time', 'event'] + covariates
            df_model = df_surv[relevant_columns].copy()
            df_model.dropna(inplace=True)

            if df_model[group_var].nunique() < 2:
                print(f"Not enough data for both {group_var} groups after dropna in {label} model for {endpoint_name} (365-day).")
                continue

            cph_365d = CoxPHFitter()
            try:
                cph_365d.fit(df_model, duration_col='time', event_col='event')
            except ConvergenceError:
                results_list.append({
                    'Endpoint': endpoint_name,
                    'Model': label,
                    'Exposure': group_var,
                    'HR': 'convergence halted',
                    'HR Lower 95%CI': 'convergence halted',
                    'HR Upper 95%CI': 'convergence halted',
                    'P-value': 'convergence halted',
                    'Timeframe': '365-day'
                })
                print(f"Convergence halted for {group_var} in {endpoint_name} with {label} model (365-day).")
                continue

            # Test PH
            try:
                ph_test_results_365d = proportional_hazard_test(cph_365d, df_model, time_transform='rank')
                print(f"\n[PH Test Results: {endpoint_name}, Model: {label}, Exposure: {group_var}, 365-day]")
                print(ph_test_results_365d.summary)
            except Exception as ex:
                print(f"PH test error for {endpoint_name}, {label}, {group_var} (365-day):\n{ex}")

            # Extraction des résultats
            if group_var in cph_365d.summary.index:
                hr = cph_365d.summary.loc[group_var, "exp(coef)"]
                lower_ci = cph_365d.summary.loc[group_var, "exp(coef) lower 95%"]
                upper_ci = cph_365d.summary.loc[group_var, "exp(coef) upper 95%"]
                p_value = cph_365d.summary.loc[group_var, "p"]
            else:
                param_names = [idx for idx in cph_365d.summary.index if group_var in idx]
                if len(param_names) == 0:
                    print(f"No parameter found for {group_var} in {label} model for {endpoint_name} (365-day).")
                    continue
                param_name = param_names[0]
                hr = cph_365d.summary.loc[param_name, "exp(coef)"]
                lower_ci = cph_365d.summary.loc[param_name, "exp(coef) lower 95%"]
                upper_ci = cph_365d.summary.loc[param_name, "exp(coef) upper 95%"]
                p_value = cph_365d.summary.loc[param_name, "p"]

            results_list.append({
                'Endpoint': endpoint_name,
                'Model': label,
                'Exposure': group_var,
                'HR': hr,
                'HR Lower 95%CI': lower_ci,
                'HR Upper 95%CI': upper_ci,
                'P-value': p_value,
                'Timeframe': '365-day'
            })

            ######################################################
            #  KM plots (Full Confounders) 
            ######################################################
            if label == "Full Confounders":
                kmf_list = []
                groups_in_data = df_model[group_var].unique()

                for grp in groups_in_data:
                    mask = (df_model[group_var] == grp)
                    kmf = KaplanMeierFitter()
                    legend_label = "LCx" if grp == 1 else "LAD or RCA"
                    kmf.fit(
                        durations=df_model.loc[mask, 'time'],
                        event_observed=df_model.loc[mask, 'event'],
                        label=legend_label
                    )
                    kmf_list.append((kmf, legend_label))

                #=============================================================
                # PLOT 1: Y-axis (0 => 1) + at_risk
                #=============================================================
                fig, ax = plt.subplots(figsize=(10, 8))
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)

                ax.set_xlabel("Time (days)", fontsize=12)
                ax.set_ylabel("Cumulative probability of event", fontsize=12)
                ax.set_ylim(0, 1)
                add_at_risk_counts(*[kmf for kmf, _ in kmf_list], ax=ax)
                plt.tight_layout()
                ax.legend().remove()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/365_{endpoint_name}_365day_full_confounders.png", dpi=300)

                #=============================================================
                # PLOT 2: Zoom (0 => 10% ou max) + annotation HR
                #=============================================================
                fig, ax = plt.subplots(figsize=(7, 5))
                max_prob = 0
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)
                    group_max = kmf.cumulative_density_.iloc[:, 0].max()
                    max_prob = max(max_prob, group_max)

                upper_limit = max(0.1, max_prob) * 1.1
                ax.set_ylim(0, upper_limit)

                # Calcul du test de log-rank
                results = logrank_test(
                    df_model.loc[df_model[group_var] == 1, 'time'],
                    df_model.loc[df_model[group_var] == 0, 'time'],
                    df_model.loc[df_model[group_var] == 1, 'event'],
                    df_model.loc[df_model[group_var] == 0, 'event']
                )
                logrank_p = results.p_value

                hr_text = (
                    f"Adjusted hazard ratio: {hr:.2f}\n"
                    f"95% CI {lower_ci:.2f} to {upper_ci:.2f}; p{'<0.001' if p_value < 0.001 else f'={p_value:.3f}'}\n"
                    f"Log-rank p{'<0.001' if logrank_p < 0.001 else f'={logrank_p:.3f}'}"
                )
                ax.text(
                    0.5, 0.20, hr_text,
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6)
                )

                ax.legend().remove()
                plt.tight_layout()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/365_{endpoint_name}_365day_full_confounders_zoomed.png", dpi=300)

###############################################################################
#           730-Day (2-Year) Truncated Survival Sensitivity Analysis
###############################################################################
for group_var in group_vars:
    for event_col, time_col, endpoint_name in survival_endpoints:
        # On part de df, on dropna sur group_var, event_col et time_col
        df_surv = df.dropna(subset=[group_var, event_col, time_col]).copy()
        
        # Vérifions qu'il y a au moins 2 groupes
        if df_surv[group_var].nunique() < 2:
            print(f"Not enough groups for {group_var} in endpoint {endpoint_name} (730-day).")
            continue
        
        # On crée des colonnes "time" et "event" tronquées à 730 jours
        df_surv['time'] = np.where(df_surv[time_col] > 730, 730, df_surv[time_col])
        df_surv['event'] = np.where((df_surv[time_col] <= 730) & (df_surv[event_col] == 1), 1, 0)

        for label, confs in confounder_sets.items():
            covariates = [group_var] + confs
            
            missing_covars = [c for c in covariates if c not in df_surv.columns]
            if missing_covars:
                print(f"Missing covariates {missing_covars} in {label} model for 730-day {endpoint_name}.")
                continue
            
            relevant_columns = ['time', 'event'] + covariates
            df_model = df_surv[relevant_columns].copy()
            df_model.dropna(inplace=True)

            if df_model[group_var].nunique() < 2:
                print(f"Not enough data for both {group_var} groups after dropna in {label} model for {endpoint_name} (730-day).")
                continue

            cph_730d = CoxPHFitter()
            try:
                cph_730d.fit(df_model, duration_col='time', event_col='event')
            except ConvergenceError:
                results_list.append({
                    'Endpoint': endpoint_name,
                    'Model': label,
                    'Exposure': group_var,
                    'HR': 'convergence halted',
                    'HR Lower 95%CI': 'convergence halted',
                    'HR Upper 95%CI': 'convergence halted',
                    'P-value': 'convergence halted',
                    'Timeframe': '730-day'
                })
                print(f"Convergence halted for {group_var} in {endpoint_name} with {label} model (730-day).")
                continue

            # Test PH
            try:
                ph_test_results_730d = proportional_hazard_test(cph_730d, df_model, time_transform='rank')
                print(f"\n[PH Test Results: {endpoint_name}, Model: {label}, Exposure: {group_var}, 730-day]")
                print(ph_test_results_730d.summary)
            except Exception as ex:
                print(f"PH test error for {endpoint_name}, {label}, {group_var} (730-day):\n{ex}")

            # Extraction des résultats
            if group_var in cph_730d.summary.index:
                hr = cph_730d.summary.loc[group_var, "exp(coef)"]
                lower_ci = cph_730d.summary.loc[group_var, "exp(coef) lower 95%"]
                upper_ci = cph_730d.summary.loc[group_var, "exp(coef) upper 95%"]
                p_value = cph_730d.summary.loc[group_var, "p"]
            else:
                param_names = [idx for idx in cph_730d.summary.index if group_var in idx]
                if len(param_names) == 0:
                    print(f"No parameter found for {group_var} in {label} model for {endpoint_name} (730-day).")
                    continue
                param_name = param_names[0]
                hr = cph_730d.summary.loc[param_name, "exp(coef)"]
                lower_ci = cph_730d.summary.loc[param_name, "exp(coef) lower 95%"]
                upper_ci = cph_730d.summary.loc[param_name, "exp(coef) upper 95%"]
                p_value = cph_730d.summary.loc[param_name, "p"]

            results_list.append({
                'Endpoint': endpoint_name,
                'Model': label,
                'Exposure': group_var,
                'HR': hr,
                'HR Lower 95%CI': lower_ci,
                'HR Upper 95%CI': upper_ci,
                'P-value': p_value,
                'Timeframe': '730-day'
            })

            ######################################################
            #  KM plots (Full Confounders) 
            ######################################################
            if label == "Full Confounders":
                kmf_list = []
                groups_in_data = df_model[group_var].unique()

                for grp in groups_in_data:
                    mask = (df_model[group_var] == grp)
                    kmf = KaplanMeierFitter()
                    legend_label = "LCx" if grp == 1 else "LAD or RCA"
                    kmf.fit(
                        durations=df_model.loc[mask, 'time'],
                        event_observed=df_model.loc[mask, 'event'],
                        label=legend_label
                    )
                    kmf_list.append((kmf, legend_label))

                #=============================================================
                # PLOT 1: Y-axis (0 => 1) + at_risk
                #=============================================================
                fig, ax = plt.subplots(figsize=(10, 8))
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)

                ax.set_xlabel("Time (days)", fontsize=12)
                ax.set_ylabel("Cumulative probability of event", fontsize=12)
                ax.set_ylim(0, 1)
                add_at_risk_counts(*[kmf for kmf, _ in kmf_list], ax=ax)
                plt.tight_layout()
                ax.legend().remove()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/730_{endpoint_name}_730day_full_confounders.png", dpi=300)

                #=============================================================
                # PLOT 2: Zoom (0 => 10% ou max) + annotation HR
                #=============================================================
                fig, ax = plt.subplots(figsize=(7, 5))
                max_prob = 0
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)
                    group_max = kmf.cumulative_density_.iloc[:, 0].max()
                    max_prob = max(max_prob, group_max)

                upper_limit = max(0.1, max_prob) * 1.1
                ax.set_ylim(0, upper_limit)

                # Calcul du test de log-rank
                results = logrank_test(
                    df_model.loc[df_model[group_var] == 1, 'time'],
                    df_model.loc[df_model[group_var] == 0, 'time'],
                    df_model.loc[df_model[group_var] == 1, 'event'],
                    df_model.loc[df_model[group_var] == 0, 'event']
                )
                logrank_p = results.p_value

                hr_text = (
                    f"Adjusted hazard ratio: {hr:.2f}\n"
                    f"95% CI {lower_ci:.2f} to {upper_ci:.2f}; p{'<0.001' if p_value < 0.001 else f'={p_value:.3f}'}\n"
                    f"Log-rank p{'<0.001' if logrank_p < 0.001 else f'={logrank_p:.3f}'}"
                )
                ax.text(
                    0.5, 0.20, hr_text,
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6)
                )

                ax.legend().remove()
                plt.tight_layout()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/730_{endpoint_name}_730day_full_confounders_zoomed.png", dpi=300)

###############################################################################
#           1095-Day (3-Year) Truncated Survival Sensitivity Analysis
###############################################################################
for group_var in group_vars:
    for event_col, time_col, endpoint_name in survival_endpoints:
        # On part de df, on dropna sur group_var, event_col et time_col
        df_surv = df.dropna(subset=[group_var, event_col, time_col]).copy()
        
        # Vérifions qu'il y a au moins 2 groupes
        if df_surv[group_var].nunique() < 2:
            print(f"Not enough groups for {group_var} in endpoint {endpoint_name} (1095-day).")
            continue
        
        # On crée des colonnes "time" et "event" tronquées à 1095 jours
        df_surv['time'] = np.where(df_surv[time_col] > 1095, 1095, df_surv[time_col])
        df_surv['event'] = np.where((df_surv[time_col] <= 1095) & (df_surv[event_col] == 1), 1, 0)

        for label, confs in confounder_sets.items():
            covariates = [group_var] + confs
            
            missing_covars = [c for c in covariates if c not in df_surv.columns]
            if missing_covars:
                print(f"Missing covariates {missing_covars} in {label} model for 1095-day {endpoint_name}.")
                continue
            
            relevant_columns = ['time', 'event'] + covariates
            df_model = df_surv[relevant_columns].copy()
            df_model.dropna(inplace=True)

            if df_model[group_var].nunique() < 2:
                print(f"Not enough data for both {group_var} groups after dropna in {label} model for {endpoint_name} (1095-day).")
                continue

            cph_1095d = CoxPHFitter()
            try:
                cph_1095d.fit(df_model, duration_col='time', event_col='event')
            except ConvergenceError:
                results_list.append({
                    'Endpoint': endpoint_name,
                    'Model': label,
                    'Exposure': group_var,
                    'HR': 'convergence halted',
                    'HR Lower 95%CI': 'convergence halted',
                    'HR Upper 95%CI': 'convergence halted',
                    'P-value': 'convergence halted',
                    'Timeframe': '1095-day'
                })
                print(f"Convergence halted for {group_var} in {endpoint_name} with {label} model (1095-day).")
                continue

            # Test PH
            try:
                ph_test_results_1095d = proportional_hazard_test(cph_1095d, df_model, time_transform='rank')
                print(f"\n[PH Test Results: {endpoint_name}, Model: {label}, Exposure: {group_var}, 1095-day]")
                print(ph_test_results_1095d.summary)
            except Exception as ex:
                print(f"PH test error for {endpoint_name}, {label}, {group_var} (1095-day):\n{ex}")

            # Extraction des résultats
            if group_var in cph_1095d.summary.index:
                hr = cph_1095d.summary.loc[group_var, "exp(coef)"]
                lower_ci = cph_1095d.summary.loc[group_var, "exp(coef) lower 95%"]
                upper_ci = cph_1095d.summary.loc[group_var, "exp(coef) upper 95%"]
                p_value = cph_1095d.summary.loc[group_var, "p"]
            else:
                param_names = [idx for idx in cph_1095d.summary.index if group_var in idx]
                if len(param_names) == 0:
                    print(f"No parameter found for {group_var} in {label} model for {endpoint_name} (1095-day).")
                    continue
                param_name = param_names[0]
                hr = cph_1095d.summary.loc[param_name, "exp(coef)"]
                lower_ci = cph_1095d.summary.loc[param_name, "exp(coef) lower 95%"]
                upper_ci = cph_1095d.summary.loc[param_name, "exp(coef) upper 95%"]
                p_value = cph_1095d.summary.loc[param_name, "p"]

            results_list.append({
                'Endpoint': endpoint_name,
                'Model': label,
                'Exposure': group_var,
                'HR': hr,
                'HR Lower 95%CI': lower_ci,
                'HR Upper 95%CI': upper_ci,
                'P-value': p_value,
                'Timeframe': '1095-day'
            })

            ######################################################
            #  KM plots (Full Confounders) 
            ######################################################
            if label == "Full Confounders":
                kmf_list = []
                groups_in_data = df_model[group_var].unique()

                for grp in groups_in_data:
                    mask = (df_model[group_var] == grp)
                    kmf = KaplanMeierFitter()
                    legend_label = "LCx" if grp == 1 else "LAD or RCA"
                    kmf.fit(
                        durations=df_model.loc[mask, 'time'],
                        event_observed=df_model.loc[mask, 'event'],
                        label=legend_label
                    )
                    kmf_list.append((kmf, legend_label))

                #=============================================================
                # PLOT 1: Y-axis (0 => 1) + at_risk
                #=============================================================
                fig, ax = plt.subplots(figsize=(10, 8))
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)

                ax.set_xlabel("Time (days)", fontsize=12)
                ax.set_ylabel("Cumulative probability of event", fontsize=12)
                ax.set_ylim(0, 1)
                add_at_risk_counts(*[kmf for kmf, _ in kmf_list], ax=ax)
                plt.tight_layout()
                ax.legend().remove()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/1095_{endpoint_name}_1095day_full_confounders.png", dpi=300)

                #=============================================================
                # PLOT 2: Zoom (0 => 10% ou max) + annotation HR
                #=============================================================
                fig, ax = plt.subplots(figsize=(7, 5))
                max_prob = 0
                for kmf, legend_label in kmf_list:
                    kmf.plot_cumulative_density(ax=ax, ci_show=False)
                    group_max = kmf.cumulative_density_.iloc[:, 0].max()
                    max_prob = max(max_prob, group_max)

                upper_limit = max(0.1, max_prob) * 1.1
                ax.set_ylim(0, upper_limit)

                # Calcul du test de log-rank
                results = logrank_test(
                    df_model.loc[df_model[group_var] == 1, 'time'],
                    df_model.loc[df_model[group_var] == 0, 'time'],
                    df_model.loc[df_model[group_var] == 1, 'event'],
                    df_model.loc[df_model[group_var] == 0, 'event']
                )
                logrank_p = results.p_value

                hr_text = (
                    f"Adjusted hazard ratio: {hr:.2f}\n"
                    f"95% CI {lower_ci:.2f} to {upper_ci:.2f}; p{'<0.001' if p_value < 0.001 else f'={p_value:.3f}'}\n"
                    f"Log-rank p{'<0.001' if logrank_p < 0.001 else f'={logrank_p:.3f}'}"
                )
                ax.text(
                    0.5, 0.20, hr_text,
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6)
                )

                ax.legend().remove()
                plt.tight_layout()
                plt.savefig(f"EVALFAST/CULPRITFAST/STATS/KM_plots/1095_{endpoint_name}_1095day_full_confounders_zoomed.png", dpi=300)

####################################
# Sauvegarde des résultats finaux
####################################
results_df = pd.DataFrame(results_list)
results_df.to_csv('EVALFAST/CULPRITFAST/STATS/cox_sensitivity_analysis.csv', index=False)
print("\nSaved cox_sensitivity_analysis.csv")
print("Finished Cox survival sensitivity analysis (5-year & 30-day) + dual KM plots (Full Confounders).")