#%% PARAMETERS
#%%

""" 
This file contains the standard parameter settings to run the simulations.
"""
#%% Parameters

""" General model parameters """
# C: Number of market exchange cycles
cycles = 3

# R: Number of renters
n_renters = 1050

# H: Number of housing units
n_apartments = 1000

"""Renters"""
# I_min: Minimum income
income_min = 3000

# I_max: Maximum income
income_max = 10000

# γ: Maximum affordable rent (in % of income)
max_rent_share = 0.4

# ∆_IC: Probability of an income change
prob_income_change = 1/12

# φ: Size of income shocks
income_change = 0.1

# ∆_R: Probability for random move
prob_random_move = 0.03

# ∆_J: Share of joiners
joiner_min = 0.002
joiner_max = 0.005

# ∆_L: Share of leavers
leaver_min = 0.002
leaver_max = 0.005

# ∆_M: Share of market screeners
screener_share = 0.2 #1/3

# λ: Minimum required utility improvement (in comparison to current utility) 
req_utility_improvement = 1.2

# ε: Minimum for preferred options required for screeners
req_n_preferred_options = 16

# A: Maximum visible apartments for applicants
max_sample_applicants = 50

# θ: Maximum number of applications per month
max_applications = 15 

# α: Preference distribution
preferences_mean = 0.22  #0.3 # 0.5
preferences_std = 0.038 #0.05 #0.075

"""Landlords-specific parameters (private sector)"""
# ∆_I: Rent increase probability 
prob_increase = 0.0 

# τ: Rent increase factor
max_increase = 1.1

# ∆_D: Share of demolished apartments
demolition_min = 0.004
demolition_max = 0.006

# ∆_C: Share of constructed apartments
construction_min = 0.004
construction_max = 0.006

# κ: Rent decrease factor if not successfully rented
rent_decrease_factor = 0.955 #0.92

# ζ: Initial inclusion threshold for comparable quality
q_threshold = 0.01

# ρ: min. required number of comparable apartments to calculate market price
min_n_comparable = 10

"""Landlords-specific parameters (public sector)"""
# ∆_s: Share of state apartments
share_state_apartments = 0.1

# η: Maximum income to rent factor allowed
inc_factor_state = 4

# P_s: Rent price of state apartments 
state_price = 1000

"""Housing unit-specific"""
# Q: Quality range for apartments 
quality_apartment = [70, 30] # corresponds to range [30,100]
quality_max_public = 35 # corresponds to 65

# P_b: Initial base price for apartments (distribution)
p_base_mean = 700
p_base_std = 200

# wq: Quality weight for price initialisation
weight_quality = 40

"""Evaluated outputs""" 
outputs = ['mean_price', 'median_price', 'vacancy_rate_p', 'vacancy_rate_s',
           'vacancy_rate_t', 'utility_p25', 'utility_p50', 'utility_p75']
