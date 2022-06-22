#%% CALIBRATION
#%%

""" 
This file contains the code to run some basic simulations with the model to
gain some insights regarding its general calibration. It uses the standard
settings for the parameters. It also contains some visualizations of the 
simulations for illustrative purposes.
"""

#%% [0] Required imports

# import required packages
import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# imports from other python files
from additional_methods import runSimulations
from parameters import (outputs,
                        inc_factor_state, 
                        share_state_apartments, 
                        state_price,
                        max_increase)
from setup import path_plots

# adjust fonts
plt.rcParams['font.sans-serif'] = "Helvetica"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "sans-serif"


#%% [1] Run Simulations

rd.seed(0)
results_all, landlords, renters = runSimulations(
    months = 120, 
    simulations = 10, 
    initialization_period = 0, 
    state_price = state_price,
    share_state_apartments = share_state_apartments,
    inc_factor_state = inc_factor_state,
    outputs = outputs,
    max_increase = max_increase)

#%% [2] Visualize results of simulations (based on all simulations)

# define list with titles to be used for plots
titles = ['Mean rent price (private sector)', 
          'Median rent price (private sector)',
          'Vacancy rate in private sector', 
          'Vacancy rate in public sector',
          'Total vacancy rate',
          'Utility of low-income households', 
          'Utility of middle-income households', 
          'Utility of high-income households']

# define list of y labels to be used for plots
y_labels = ['Mean price', 'Median price', 'Vacancy rate', 'Vacancy rate',
            'Vacancy rate','Utility', 'Utility', 'Utility' ]

# Loop through all results and plot results separately
x = np.arange(0, len(results_all['mean_price'][0]), 1)

i = -1
for key in results_all:
    i += 1
    # select pre-defined title and label from list for each plot
    plt.title(titles[i])
    plt.ylabel(y_labels[i])
    # set label for x-axis (same for all plots)
    plt.xlabel('Months')
    # plot all simulations
    for array in results_all[key]:
        plt.plot(x, array, color = 'gray', alpha = 0.3)
    # plot mean of all simulations
    plt.plot(np.arange(0,len(np.array(results_all[key]).mean(axis=0))), 
             np.array(results_all[key]).mean(axis=0), color = 'gray',
             linewidth=2)
    # save plot
    plt.savefig(path_plots +'\\Calibration\\' + str(key) + '-development.png', 
                dpi=300)
    plt.show()
    
# create combined version of all utility plots
plt.title('Utility of households')
plt.ylabel(y_labels[i])
plt.xlabel('Months')
# loop through all simulations for all quantiles and define quantile color
for array in results_all['utility_p25']:
    plt.plot(x, array,color='#62BD69', alpha = 0.3)
plt.plot(np.arange(0,len(np.array(results_all['utility_p25']).mean(axis=0))), 
             np.array(results_all['utility_p25']).mean(axis=0), 
             color = '#62BD69', linewidth=2)
for array in results_all['utility_p50']:
    plt.plot(x, array,color='#358856', alpha = 0.3)
plt.plot(np.arange(0,len(np.array(results_all['utility_p50']).mean(axis=0))), 
             np.array(results_all['utility_p50']).mean(axis=0), 
             color = '#358856', linewidth=2)
for array in results_all['utility_p75']:
    plt.plot(x, array, color='#0C3823', alpha = 0.3)
plt.plot(np.arange(0,len(np.array(results_all['utility_p75']).mean(axis=0))), 
             np.array(results_all['utility_p75']).mean(axis=0), 
             color = '#0C3823', linewidth=2)
# custom label (only show one label for each color)
custom_lines = [Line2D([0], [0], color='#0C3823', lw=4),
                Line2D([0], [0], color='#358856', lw=4),
                Line2D([0], [0], color='#62BD69', lw=4)]
plt.legend(custom_lines, ['high-income', 'middle-income', 
          'low-income'], bbox_to_anchor=(1.05, 1), loc=2, 
           fontsize='medium', frameon=False)
# save plot
plt.savefig(path_plots +'\\Calibration\\utility-development-combined.png', 
            dpi=300, bbox_inches = 'tight')
# display plot
plt.show()
    
#%% [3] Visualize distributions (based on last simulation only)

# combining all plots in one figure    
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (10,5))

# subplot for apartment quality
ax0.hist(landlords.quality, color = 'gray', rwidth=0.9, alpha = 0.6)
ax0.set_xlabel('Quality', size = 'small')
ax0.set_ylabel('Frequency', size = 'small')
ax0.tick_params(axis='both', which='major', labelsize=8)
ax0.set_title('(a) Quality of housing units', size = 'medium')

# subplot for prices (grouped by public/private landlords)
bins = np.linspace(min(landlords.price), max(landlords.price), 10)
ax1.hist(landlords.price[np.where(landlords.private==True)], bins,
         label='private sector', alpha=0.7, rwidth=0.9)
ax1.hist(landlords.price[np.where(landlords.private==False)], bins,
         label='public sector', alpha=0.6, rwidth=0.9, color = 'orange')
ax1.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc=2, frameon=False,
           borderaxespad=0.)
ax1.set_xlabel('Price', size = 'small')
ax1.set_ylabel('Frequency', size = 'small')
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.set_title('(b) Rent price of housing units', size = 'medium')

# subplot for preferences
ax2.hist(renters.preferences, color = 'gray', rwidth=0.9, alpha = 0.6)
ax2.set_xlabel('Preference (α)', size = 'small')
ax2.set_ylabel('Frequency', size = 'small')
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.set_title('(c) Preferences of households', size = 'medium')

# subplot for renters' income 
ax3.hist(renters.income, color = 'gray', rwidth=0.9, alpha = 0.6)
ax3.set_xlabel('Income', size = 'small')
ax3.set_ylabel('Frequency', size = 'small')
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.set_title('(d) Income of households', size = 'medium')

# show and save figure
fig.tight_layout()
plt.savefig(path_plots +'\\Calibration\\histograms.png',dpi=300)
plt.show()

#%% [4] Correlation plots (based on last simulation only)

# create figure including all correlation plots 
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (10,7))

# Subplot 1: Quality-Price (grouped by private/public sector)
ax0.scatter(landlords.quality[np.where(landlords.private == True)],
            landlords.price[np.where(landlords.private == True)], alpha = 0.3)
ax0.scatter(landlords.quality[np.where(landlords.private == False)],
            landlords.price[np.where(landlords.private == False)], alpha = 0.3,
            color = 'orange')
ax0.set_xlabel('Quality', size = 'small')
ax0.set_ylabel('Rent price', size = 'small')
ax0.tick_params(axis='both', which='major', labelsize=8)
ax0.set_title('(a) Quality and price', size = 'medium')

# Subplot 2: Utility and income (grouped by private/public/no apartment)

# First update renters' utility (not happened yet after last month)
renters.updateUtility()

# private sector housing
ax1.scatter(renters.income[np.where(
    (renters.price != 1000) & (renters.price != 0))], 
    renters.utility[np.where((renters.price != 1000) & (renters.price != 0))], 
    alpha = 0.3, label ='private sector')
# public housing
ax1.scatter(renters.income[np.where(renters.price == 1000)],
            renters.utility[np.where(renters.price == 1000)], alpha = 0.3, 
            color = 'orange', label ='public sector')
# no apartment
ax1.scatter(renters.income[np.where(renters.price == 0)],
            renters.utility[np.where(renters.price == 0)], alpha = 0.3, 
            color = 'tomato', label = 'no housing')
ax1.set_xlabel('Renter\'s income', size = 'small')
ax1.set_ylabel('Renter\'s utility', size = 'small')
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.set_title('(b) Income and utility', size = 'medium')
ax1.legend(fontsize='small', loc='upper right', 
           bbox_to_anchor=(-0.5, 1.25, 0.9, 0), borderaxespad=0., ncol=3, 
           mode='expand')

# subplot for renters' income and price (exclude renters with no apartments)
# private sector housing
ax2.scatter(renters.income[np.where(
    (renters.price != 1000) & (renters.price != 0))],
    renters.price[np.where((renters.price != 1000) & (renters.price != 0))], 
    alpha = 0.3)
# public housing
ax2.scatter(renters.income[np.where(renters.price == 1000)],
            renters.price[np.where(renters.price == 1000)], alpha = 0.3,
            color = 'orange')
# no apartment
ax2.scatter(renters.income[np.where(renters.price == 0)],
            renters.price[np.where(renters.price == 0)], alpha = 0.3, 
            color = 'tomato')
ax2.set_xlabel('Renter\'s income', size = 'small')
ax2.set_ylabel('Rent price', size = 'small')
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.set_title('(c) Income and price', size = 'medium')

# subplot for quality and price (separately for private and state)
# plot renters in a private apartment in blue
ax3.scatter(renters.income[np.where(
    (renters.price != 1000) & (renters.price != 0))],
    renters.quality[np.where((renters.price != 1000) & (renters.price != 0))], 
    alpha = 0.3, label='public sector')
# plot renters in a state apartment in orange
ax3.scatter(renters.income[np.where(renters.price == 1000)],
            renters.quality[np.where(renters.price == 1000)], alpha = 0.3,
            label='public sector', color = 'orange')
# plot renters with no apartment in red
ax3.scatter(renters.income[np.where(renters.price == 0)],
            renters.quality[np.where(renters.price == 0)], alpha = 0.3,
            label='no housing', color = 'tomato')
ax3.set_xlabel('Renter\'s income', size = 'small')
ax3.set_ylabel('Quality of housing unit', size = 'small')
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.set_title('(c) Income and quality', size = 'medium')

# show and save figure
fig.tight_layout()
plt.savefig(path_plots +'\\Calibration\\correlation.png',dpi=300)
plt.show()