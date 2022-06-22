#%% RUN OFAT
#%%

""" 
This file contains the code to run the OFAT analyses.
"""

#%% [0] Required imports

# import required packages
import numpy.random as rd
import numpy as np

# imports from other python files
from setup import path_tables
from additional_methods import (runSimulations,
                                plotOFAT,
                                tabulateResults,
                                saveResults)
from parameters import outputs

#%% [1] OFAT - State Price

# Define Parameters
parameter_values = [900, 1000, 1100, 1200]

# set seed (for the purpose of reproducibility) 
rd.seed(1)

# create dictionary with empty lists to store results
results_ofat_p = {key: [] for key in outputs}
 
#Start simulations   
print("---Start OFAT simulations for state price---")
for p in range(len(parameter_values)):
    print("Parameter", p+1, " /", len(parameter_values))
    # set parameter value for state_price (varies between loops)
    state_price = parameter_values[p]
    # run simulation with current parameter settings
    results_sim, landlords, renters = runSimulations(
        months = 56,
        simulations = 100,
        initialization_period = 50,
        state_price = state_price,
        share_state_apartments = 0.1,
        inc_factor_state = 4,
        outputs = outputs,
        max_increase = 1.1)
    # append results from all simulations of current parameter settings
    results_ofat_p['mean_price'].append(
        np.array(results_sim['mean_price']))
    results_ofat_p['median_price'].append(
        np.array(results_sim['median_price']))
    results_ofat_p['vacancy_rate_p'].append(
        np.array(results_sim['vacancy_rate_p']))
    results_ofat_p['vacancy_rate_s'].append(
        np.array(results_sim['vacancy_rate_s']))
    results_ofat_p['vacancy_rate_t'].append(
        np.array(results_sim['vacancy_rate_t']))
    results_ofat_p['utility_p25'].append(
        np.array(results_sim['utility_p25']))
    results_ofat_p['utility_p50'].append(
        np.array(results_sim['utility_p50']))
    results_ofat_p['utility_p75'].append(
        np.array(results_sim['utility_p75']))

# Generate and store plots to visualize results
plotOFAT("Rent price for public housing", parameter_values, results_ofat_p)

# Test for significance and create tables
results_significance = []
for output in outputs:
    mean_values = results_ofat_p[output]
    results_significance.append(tabulateResults(mean_values,parameter_values)) 

# store resulting tables    
saveResults("\\OFAT_Results_Price public housing.xlsx", path_tables,  
            results_significance)

# Get Latex code for tables
for i in range(len(results_significance)):
    print(results_significance[i].to_latex(caption=outputs[i], bold_rows=True,
          column_format=(len(parameter_values)+1)*'S'))
#%% [2] OFAT - Share of state apartments

# Define Parameters
parameter_values = [0.05, 0.1, 0.15, 0.2]

# set seed (for the purpose of reproducibility) 
rd.seed(2)

# create dictionary with empty lists to store results
results_ofat_s = {key: [] for key in outputs}
 
#Start simulations   
print("---Start OFAT simulations for share of state apartments---")
for p in range(len(parameter_values)):
    print("Parameter", p+1, " /", len(parameter_values))
    # set parameter value for state_price (varies between loops)
    share_state_apartments = parameter_values[p]
    # run simulation with current parameter settings
    results_sim, landlords, renters = runSimulations(
        months= 56,
        simulations=100,
        initialization_period=50,
        state_price = 1000,
        share_state_apartments = share_state_apartments,
        inc_factor_state = 4,
        outputs = outputs,
        max_increase = 1.1)
    # append results from all simulations of current parameter settings
    results_ofat_s['mean_price'].append(
        np.array(results_sim['mean_price']))
    results_ofat_s['median_price'].append(
        np.array(results_sim['median_price']))
    results_ofat_s['vacancy_rate_p'].append(
        np.array(results_sim['vacancy_rate_p']))
    results_ofat_s['vacancy_rate_s'].append(
        np.array(results_sim['vacancy_rate_s']))
    results_ofat_s['vacancy_rate_t'].append(
        np.array(results_sim['vacancy_rate_t']))
    results_ofat_s['utility_p25'].append(
        np.array(results_sim['utility_p25']))
    results_ofat_s['utility_p50'].append(
        np.array(results_sim['utility_p50']))
    results_ofat_s['utility_p75'].append(
        np.array(results_sim['utility_p75']))

# Generate and store plots to visualize results
plotOFAT("Share of public housing", parameter_values, results_ofat_s)

# Test for significance and create tables
results_significance = []
for output in outputs:
    mean_values = results_ofat_s[output]
    results_significance.append(tabulateResults(mean_values,parameter_values)) 

# store resulting tables    
saveResults("\\OFAT_Results_Share of public housing.xlsx", path_tables, 
             results_significance)

# Get Latex code for tables
for i in range(len(results_significance)):
    print(results_significance[i].to_latex(caption=outputs[i], bold_rows=True,
          column_format=(len(parameter_values)+1)*'S'))
    
#%% [3] OFAT - Income-to-rent ratio set by state (η)

# Define Parameters
parameter_values = [3, 4, 5, 6, 7]

# set seed (for the purpose of reproducibility) 
rd.seed(3)

# create dictionary with empty lists to store results
results_ofat_c = {key: [] for key in outputs}
 
#Start simulations   
print("---Start OFAT simulations for income criterion state---")
for p in range(len(parameter_values)):
    print("Parameter",p+1," /",len(parameter_values))
    # set parameter value for state_price (varies between loops)
    inc_factor_state = parameter_values[p]
    # run simulation with current parameter settings
    results_sim, landlords, renters = runSimulations(
        months = 56,
        simulations = 100,
        initialization_period = 50,
        state_price = 1000,
        share_state_apartments = 0.1,
        inc_factor_state = inc_factor_state,
        outputs = outputs,
        max_increase = 1.1)
    # append results from all simulations of current parameter settings
    results_ofat_c['mean_price'].append(
        np.array(results_sim['mean_price']))
    results_ofat_c['median_price'].append(
        np.array(results_sim['median_price']))
    results_ofat_c['vacancy_rate_p'].append(
        np.array(results_sim['vacancy_rate_p']))
    results_ofat_c['vacancy_rate_s'].append(
        np.array(results_sim['vacancy_rate_s']))
    results_ofat_c['vacancy_rate_t'].append(
        np.array(results_sim['vacancy_rate_t']))
    results_ofat_c['utility_p25'].append(
        np.array(results_sim['utility_p25']))
    results_ofat_c['utility_p50'].append(
        np.array(results_sim['utility_p50']))
    results_ofat_c['utility_p75'].append(
        np.array(results_sim['utility_p75']))

# Generate and store plots to visualize results
plotOFAT("Maximum income-to-rent ratio", parameter_values, 
         results_ofat_c)

# Test for significance and create tables
results_significance = []
for output in outputs:
    mean_values = results_ofat_c[output]
    results_significance.append(tabulateResults(mean_values,parameter_values)) 

# store resulting tables    
saveResults("\\OFAT_Results_Maximum income-to-rent ratio.xlsx", path_tables, 
            results_significance)

# Get Latex code for tables
for i in range(len(results_significance)):
    print(results_significance[i].to_latex(caption=outputs[i], bold_rows=True,
          column_format=(len(parameter_values)+1)*'S'))

#%% [4] OFAT - Maximum rent increase factor (τ)

# Define Parameters
parameter_values = [1, 1.05, 1.1, 1.15, 1.2, 1.25]

# set seed (for the purpose of reproducibility) 
rd.seed(4)

# create dictionary with empty lists to store results
results_ofat_rc = {key: [] for key in outputs}
 
#Start simulations   
print("---Start OFAT simulations for state price---")
for p in range(len(parameter_values)):
    print("Parameter", p+1, " /", len(parameter_values))
    # set parameter value for state_price (varies between loops)
    max_increase = parameter_values[p]
    # run simulation with current parameter settings
    results_sim, landlords, renters = runSimulations(
        months = 56,
        simulations = 100,
        initialization_period = 50,
        state_price = 1000,
        share_state_apartments = 0.1,
        inc_factor_state = 4,
        outputs = outputs,
        max_increase = max_increase)
    # append results from all simulations of current parameter settings
    results_ofat_rc['mean_price'].append(
        np.array(results_sim['mean_price']))
    results_ofat_rc['median_price'].append(
        np.array(results_sim['median_price']))
    results_ofat_rc['vacancy_rate_p'].append(
        np.array(results_sim['vacancy_rate_p']))
    results_ofat_rc['vacancy_rate_s'].append(
        np.array(results_sim['vacancy_rate_s']))
    results_ofat_rc['vacancy_rate_t'].append(
        np.array(results_sim['vacancy_rate_t']))
    results_ofat_rc['utility_p25'].append(
        np.array(results_sim['utility_p25']))
    results_ofat_rc['utility_p50'].append(
        np.array(results_sim['utility_p50']))
    results_ofat_rc['utility_p75'].append(
        np.array(results_sim['utility_p75']))

# Generate and store plots to visualize results
plotOFAT("Maximum rent increase factor", parameter_values, results_ofat_rc)

# Test for significance and create tables
results_significance = []
for output in outputs:
    mean_values = results_ofat_rc[output]
    results_significance.append(tabulateResults(mean_values,parameter_values)) 

# store resulting tables    
saveResults("\\OFAT_Results_Maximum rent increase factor.xlsx", path_tables, results_significance)
    
# Get Latex code for tables
for i in range(len(results_significance)):
    print(results_significance[i].to_latex(caption=outputs[i], bold_rows=True,
          column_format=(len(parameter_values)+1)*'S'))