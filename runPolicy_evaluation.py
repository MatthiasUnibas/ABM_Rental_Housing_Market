#%% RUN POLICY INTERVENTION 
#%%

""" 
This file contains the code to run and plot the ceteris paribus analysis for
a policy intervention. 
"""

#%% [0] Required imports

# import required packages
import numpy.random as rd
import pandas as pd

# imports from other python files
from additional_methods import (runIntervention, plotIntervention, 
                                subplotIntervention, tableIntervention_results)
from parameters import (state_price, 
                        share_state_apartments, 
                        inc_factor_state, 
                        outputs,
                        max_increase)

from setup import path_tables

#%% [1] Define experiment specific settings

months_before_intervention = 50
months_after_intervention = 48
simulations = 100
new_apartments_options = [10, 20, 30, 40, 50, 60]

#%% [1] Run simulations before and after (no) policy intervention

# create lists to store the results
store_int_results = dict()
store_no_int_results = dict()

# loop through defined options for construction values and simulate experiment
for i in range(len(new_apartments_options)):
    new_apartments = new_apartments_options[i]
    print('PARAMETER:', i+1, '/', len(new_apartments_options))
    # set seed (for the purpose of reproducibility) 
    rd.seed(2)
    # run experiments
    results_int, results_no_int, renters, landlords = runIntervention(
        months_before_intervention = months_before_intervention, 
        months_after_intervention = months_after_intervention, 
        simulations = simulations, 
        initialization_period = 0, 
        state_price = state_price, 
        share_state_apartments = share_state_apartments, 
        inc_factor_state = inc_factor_state, 
        outputs = outputs,
        new_apartments = new_apartments_options[i],
        max_increase = max_increase)
    
    # store the results
    store_int_results[str(new_apartments_options[i])] = results_int
    store_no_int_results[str(new_apartments_options[i])] = results_no_int

#%% [2] Plot intervention results

y_labels = ['Mean price', 'Median price', 'Vacancy rate [in %]', 
            'Vacancy rate [in %]', 'Vacancy rate [in %]', 'Utility', 
            'Utility', 'Utility' ]

titles = ['Price', 
          'Price',
          'Private sector', 
          'Public sector',
          'Total',
          'Low-income households', 
          'Middle-income households', 
          'High-income households']

# Separate plots for each outcome with means incl. confidence intervals 
for i in range(len(new_apartments_options)):
    for j in range(len(y_labels)):
        plotIntervention(title = titles[j],
                    ylabel = y_labels[j], 
                    output = outputs[j], 
                    results_int = store_int_results[
                        str(new_apartments_options[i])], 
                    results_no_int = store_no_int_results[
                        str(new_apartments_options[i])], 
                    months_before_intervention = months_before_intervention, 
                    months_after_intervention = months_after_intervention, 
                    pre_month_included = 12,
                    new_apartments = new_apartments_options[i])

# Combine the utility and vacancy plots each in one figure with three subplots
for i in range(len(new_apartments_options)):
    subplotIntervention(results_int_total = store_int_results[
                            str(new_apartments_options[i])],  
                        results_no_int_total = store_no_int_results[
                            str(new_apartments_options[i])], 
                        new_apartments = new_apartments_options[i], 
                        months_before_intervention= months_before_intervention, 
                        months_after_intervention = months_after_intervention, 
                        pre_month_included = 12)

#%% [3] Get tables with mean differences per quarter after intervention

# store tables in a dictionary
all_pval_tables = dict()
for new_apartments in new_apartments_options:       
    for output in outputs:
        all_pval_tables[
            str(new_apartments)+' _ '+output] = tableIntervention_results(
            output = output, 
            results_int = store_int_results[str(new_apartments)], 
            results_no_int = store_no_int_results[str(new_apartments)], 
            months_before_intervention = months_before_intervention, 
            months_after_intervention = months_after_intervention)

# write data to excel file (each table in a separate sheet)
file = path_tables + "\\Ceteris paribus analysis.xlsx"
with pd.ExcelWriter(file) as writer:
    for key in all_pval_tables:
        all_pval_tables[key].to_excel(writer, sheet_name = key, index=True)
writer.save()

# Get Latex code for tables
for key in all_pval_tables:
    print(all_pval_tables[str(key)].to_latex(caption=key, bold_rows=True,
          column_format=('SSSS')))
