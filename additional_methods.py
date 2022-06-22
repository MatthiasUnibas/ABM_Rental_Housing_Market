#%% ADDITIONAL METHODS
#%%

""" 
This file contains additional methods that have been specifically implemented
for the model.
"""

#%% [0] Required imports

# import required packages
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import copy

# imports from other python files
from agents import Renters, Landlords
from setup import path_plots
from parameters import (inc_factor_state, income_min, preferences_mean, 
                        preferences_std, income_max, max_rent_share, 
                        prob_income_change, income_change, prob_random_move,
                        construction_max, joiner_min, joiner_max, leaver_min,
                        leaver_max, prob_increase, quality_max_public, 
                        demolition_min, demolition_max, construction_min, 
                        rent_decrease_factor, screener_share, state_price,
                        req_utility_improvement, req_n_preferred_options, 
                        max_sample_applicants, max_applications, cycles,
                        quality_apartment, q_threshold, min_n_comparable,
                        p_base_std, weight_quality, n_renters, n_apartments)

#%% [1] Methods for initializing and updating the population

def initializeModel(n_renters, n_apartments, share_state_apartments, 
                    state_price):
    """
    Method that creates the initial population of landlords and renters.
    """
    # Calculate number of private and state landlords to be created
    n_private = int(n_apartments * (1 - share_state_apartments))
    n_state = int(n_apartments * share_state_apartments)
    # prepare quality ratings for all apartments so that it can be used for 
    # prices and quality arrays creation afterwards.
    quality = np.append(quality_apartment[1] + (rd.rand(n_private) * 
                        quality_apartment[0]), quality_apartment[1] + (
                        rd.rand(n_state) * quality_max_public)) 
    # create instance of class Landlords containing entire landlord population
    landlords = Landlords(
                private = np.append(np.ones(n_private,dtype=bool), 
                                    np.zeros(n_state,dtype=bool)),
                apartment = np.arange(n_private+n_state)+1,
                quality = quality,
                price = np.append(
                    rd.normal(state_price, p_base_std, n_private) + ( 
                    (quality[0:n_private]-quality_apartment[1]) 
                    * weight_quality), np.ones(n_state)*state_price),                            
                available = np.ones(n_private+n_state,dtype=bool),
                random = rd.rand(n_private+n_state))

    # create instance of class Landlords containing entire renter population
    renters = Renters(uid= np.arange(n_renters)+1, 
                      apartment=np.ones(n_renters,dtype=int)*-1, 
                      price=np.zeros(n_renters), 
                      quality=np.zeros(n_renters),
                      searching=np.ones(n_renters,dtype=bool),
                      income=rd.uniform(income_min,income_max,n_renters),
                      random=rd.rand(n_renters),
                      preferences= rd.normal(preferences_mean, preferences_std,
                                             n_renters),
                      utility=np.zeros(n_renters))
    return(renters,landlords)

def updatePopulation(renters, landlords): 
    """
    Method that controls the joiner and leaver processes for renters. Leavers
    will leave the model entirely and joiners will start to look for an
    apartment immediately after they have joined.
    """
    # randomly draw number of leavers from defined range for population share
    number_of_leavers = round(
        rd.uniform(leaver_min,leaver_max)*len(renters.uid))
    # randomly draw leavers and store their index
    leaver_index = rd.randint(0,len(renters.uid),number_of_leavers)
    #get apartment numbers of leavers 
    leaver_apartments = renters.apartment[leaver_index]
    # set apartment status for landlords of leavers to available
    landlords.available[np.where(
        np.isin(landlords.apartment,leaver_apartments))] = True
    # remove leavers from population
    renters.apartment = np.delete(renters.apartment, leaver_index)
    renters.income = np.delete(renters.income, leaver_index)
    renters.preferences = np.delete(renters.preferences, leaver_index)
    renters.price = np.delete(renters.price, leaver_index)
    renters.quality = np.delete(renters.quality, leaver_index)
    renters.random = np.delete(renters.random, leaver_index)
    renters.searching = np.delete(renters.searching, leaver_index)
    renters.uid = np.delete(renters.uid, leaver_index)
    renters.utility = np.delete(renters.utility, leaver_index)

    # get number of joiners 
    number_of_joiners = round(
        rd.uniform(joiner_min,joiner_max)*len(renters.uid))
    # append attributes of joiners to arrays of existing population
    uid_next = max(renters.uid) + 1
    renters.uid = np.append(renters.uid, 
                            np.arange(uid_next,uid_next + number_of_joiners,1))
    renters.apartment = np.append(renters.apartment, 
                                  np.ones(number_of_joiners, dtype=int)*-1)
    renters.price = np.append(renters.price, np.zeros(number_of_joiners))
    renters.quality = np.append(renters.quality, np.zeros(number_of_joiners))
    renters.searching = np.append(renters.searching, 
                                  np.ones(number_of_joiners, dtype=bool))
    renters.income = np.append(renters.income, rd.uniform(
        income_min, income_max, number_of_joiners))
    renters.random = np.append(renters.random, rd.rand(number_of_joiners))
    renters.preferences = np.append(renters.preferences, 
        rd.normal(preferences_mean, preferences_std, number_of_joiners))
    renters.utility = np.append(renters.utility, np.zeros(number_of_joiners))
    return(renters, landlords)

def updateApartments(renters, landlords):
    """
    Method that controls the construction of new as well as the demolition of
    existing apartments. Renters living in an apartment that will be demolished
    also get updated. 
    """
    # get number of apartments to be demolished  (randomly drawn from 
    # predefined range for share of the current population)
    number_of_demolitions = round(
        rd.uniform(demolition_min, demolition_max)*len(landlords.apartment))
    # get index of private apartments (no state apartments are demolished)
    private_index = np.where(landlords.private==True)[0]
    # randomly select private apartments to be demolished
    demolition_index = rd.choice(private_index,number_of_demolitions)
    # get apartment numbers of demolished apartment 
    demolition_apartment = landlords.apartment[demolition_index]
    # update renters which are living in an apartment that will be domolished
    renters.searching[np.where(
        np.isin(renters.apartment, demolition_apartment))] = True
    renters.price[np.where(
        np.isin(renters.apartment, demolition_apartment))] = 0
    renters.quality[np.where(
        np.isin(renters.apartment, demolition_apartment))] = 0
    renters.apartment[np.where(
        np.isin(renters.apartment, demolition_apartment))] = -1
    #remove landlords from population whose apartment will be demolished
    landlords.apartment = np.delete(landlords.apartment, demolition_index)   
    landlords.private = np.delete(landlords.private, demolition_index)   
    landlords.quality = np.delete(landlords.quality, demolition_index)   
    landlords.price = np.delete(landlords.price, demolition_index)   
    landlords.available = np.delete(landlords.available, demolition_index)   
    landlords.random = np.delete(landlords.random, demolition_index)  
    
    # get number of new apartments to be constructed
    number_of_new_apartments = round(
        rd.uniform(construction_min,construction_max)*len(landlords.apartment))
    # append attributes of new apartments to arrays of existing landlords
    apartment_next = max(landlords.apartment) + 1     
    landlords.apartment = np.append(landlords.apartment,
        np.arange(apartment_next, apartment_next+number_of_new_apartments,1)) 
    #only private appartments added (no state apartments are constructed)
    landlords.private = np.append(landlords.private, 
                                  np.ones(number_of_new_apartments,dtype=bool))   
    landlords.quality = np.append(landlords.quality, 
                                  rd.rand(number_of_new_apartments) *
                                  quality_apartment[0] + quality_apartment[1]) 
    landlords.price = np.append(landlords.price, 
                                np.zeros(number_of_new_apartments))
    landlords.available = np.append(landlords.available,
                                np.ones(number_of_new_apartments,dtype=bool))   
    landlords.random = np.append(landlords.random, 
                                 rd.rand(number_of_new_apartments))  
    return(renters, landlords)  

def constructStateApartments(landlords, new_apartments, state_price):
    """
    Method controls a 'one-time' construction of state apartments. Number of 
    apartments to be constructed can be flexibly chosen when the method is
    called (with the new_apartments parameter).
    """
    # append attributes of new state apartments to existing landlord arrays
    apartment_next = max(landlords.apartment) + 1     
    landlords.apartment = np.append(landlords.apartment,
                    np.arange(apartment_next,apartment_next+new_apartments,1)) 
    landlords.private = np.append(landlords.private, 
                                  np.zeros(new_apartments,dtype=bool))   
    landlords.quality = np.append(landlords.quality, 
        rd.rand(new_apartments)*quality_apartment[0] + quality_apartment[1])   
    landlords.price = np.append(landlords.price, 
                                np.ones(new_apartments)*state_price)   
    landlords.available = np.append(landlords.available, 
                                    np.ones(new_apartments,dtype=bool))   
    landlords.random = np.append(landlords.random, rd.rand(new_apartments))  
    return(landlords)

#%% [2] Methods to run simulations and experiments (Process flows)

# Run model for one month        
def simulateMonth(renters, landlords, m, inc_factor_state, max_increase, 
                  state_price):   
    """ 
    Standard process to simulate one month. Method does not include evaluation 
    of results and cannot be used for first month (slightly different set up
    required due to new initialization of populations).
    """
    # only apply following steps from month 2 onwards
    if m > 0:
        # Update population, apartments, prices, income and utility 
        renters, landlords = updatePopulation(renters, landlords)
        renters, landlords = updateApartments(renters, landlords)
        renters = landlords.updatePrice(renters, prob_increase, max_increase)
        renters.updateIncome(prob_income_change, income_change, income_min, 
                             income_max)
        renters.updateUtility()
        
        # Renters check affordability, move randomly, and screen market   
        landlords = renters.checkAffordability(landlords, max_rent_share)
        landlords = renters.moveRandomly(landlords, prob_random_move)
        landlords = renters.screenMarket(landlords, screener_share, 
                                         req_utility_improvement, 
                                         req_n_preferred_options)
        
        # Landlords adjust pricing
        landlords.setPrice(max_increase, q_threshold, min_n_comparable)
    
    # Market exchange (cycles are mimicking a 3 months notice period)
    for cycle in range(cycles):
        # landlords decrease offered price if apartment remains available
        if cycle > 0:
            landlords.decreasePrice(rent_decrease_factor)
        # Application & Selection
        apartment_info, applicants = renters.application(
            landlords, max_rent_share, inc_factor_state,
            state_price, max_sample_applicants, max_applications)
        renters = landlords.selectTenant(renters, apartment_info,
                                         applicants)
    return (renters,landlords)

# Evaluate one month
def evaluateMonth(renters,landlords):
    """ 
    Method that evaluate outcomes for one month and returns them.
    """
    # create dictionary to store the evaluation results
    results_month = dict()
    # mean price of rented private apartments
    results_month['mean_price'] = np.mean(landlords.price[np.where(
        (landlords.available==False) & (landlords.private==True))])
    # median price of rented private apartments
    results_month['median_price'] = np.median(landlords.price[np.where(
        (landlords.available==False) & (landlords.private==True))])
    # vacancy rate  (for private apartment market) in %
    results_month['vacancy_rate_p'] = (len(np.where(
        (landlords.available==True) & (landlords.private==True))[0]) 
        / len(np.where(landlords.private==True)[0])) * 100
    # vacancy rate  (for state apartment market) in %
    results_month['vacancy_rate_s'] = (len(np.where(
        (landlords.available==True) & (landlords.private==False))[0]) 
        / len(np.where(landlords.private==False)[0])) * 100
    # vacancy rate  (total) in %
    results_month['vacancy_rate_t'] = len(np.where(
        landlords.available==True)[0]) / len(landlords.available) * 100
    # Utility (lowest income quartile) for renters living in an apartment 
    results_month['utility_p25'] = np.mean(renters.utility[np.where(
        renters.income<np.quantile(renters.income,.25))])
    # Utility (middle 50% income class) for renters living in an apartment
    results_month['utility_p50'] = np.mean(renters.utility[np.where(
        (renters.income<=np.quantile(renters.income,.75)) & 
        (renters.income>=np.quantile(renters.income,.25)))])
    # Utility (75 percentile) for renters living in an apartment
    results_month['utility_p75'] = np.mean(renters.utility[np.where(
        renters.income>np.quantile(renters.income,.75))])
    return(results_month)

def runMonths(months, initialization_period, state_price, 
              share_state_apartments, inc_factor_state, outputs, max_increase):    
    """
    Run model for several months and store results (after end of initialization
    period) into arrays within a dictionary.
    """
    #create dictionary to store simulation results
    results_sim = {key: np.empty(0) for key in outputs}
    # initialization of population
    renters, landlords = initializeModel(n_renters, n_apartments, 
                                         share_state_apartments, state_price)
    #simulate months
    for m in range(months):
        # print calculation progress
        if ((m+10) % 10 ==0) & (m+10 < months):
            print('   Months:',m+1,'-',m+10, '(of', months, 'months)')
        elif ((m+10) % 10 ==0) & (m+10 >= months):
            print('   Months:',m+1,'-', months, '(of', months, 'months)')
        # run simulations
        renters, landlords = simulateMonth(renters, landlords, m, 
                                           inc_factor_state, max_increase,
                                           state_price) 
        # start evaluation after initialization period
        if m >= initialization_period:
            results_month = evaluateMonth(renters,landlords)
            # append results from current month to arrays in dictionary
            results_sim['mean_price'] = np.append( 
                results_sim['mean_price'], results_month['mean_price'])
            results_sim['median_price'] = np.append( 
                results_sim['median_price'], results_month['median_price'])
            results_sim['vacancy_rate_p'] = np.append(
                results_sim['vacancy_rate_p'], results_month['vacancy_rate_p'])
            results_sim['vacancy_rate_s'] = np.append(
                results_sim['vacancy_rate_s'], results_month['vacancy_rate_s'])
            results_sim['vacancy_rate_t'] = np.append(
                results_sim['vacancy_rate_t'], results_month['vacancy_rate_t'])
            results_sim['utility_p25'] = np.append(
                results_sim['utility_p25'], results_month['utility_p25'])
            results_sim['utility_p50'] = np.append(
                results_sim['utility_p50'], results_month['utility_p50'])
            results_sim['utility_p75'] = np.append(
                results_sim['utility_p75'], results_month['utility_p75'])
    return(results_sim, renters, landlords)

def runSimulations(months, simulations, initialization_period,state_price, 
                   share_state_apartments, inc_factor_state, outputs, 
                   max_increase):
    """
    Run and evaluate several simulations. 
    
    Parameters
    ----------
    months : integer
        defines for how many months the model should be run per simulation.
    simulations : integer
        defines number of simulations to be run.
    initialization_period : integer
        defines number of months per simulations where the model runs without
        being evaluated yet.
    state_price : float
        sets the price for state_apartments
    share_state_apartments : float [0,1]
        defines the share of state apartments (relative to total apartments)
    inc_factor_state : float
        defines maximum factor the income of households may exceed the price 
        of the state apartment such that they are still allowed to apply for
        the apartment.
    outputs : list
        list containing the labels (/keys) for all required outputs that will
        be evaluated.
        
    Returns
    -------
    results_all : dictionary
        contains all simulations results
        
    """
    # print state parameters
    print("State price:", state_price, 
          "\nShare of state apartment:", share_state_apartments,
          "\nIncome factor state:", inc_factor_state)

    # create dictionary with empty arrays to store results
    results_all = {key: []  for key in outputs}
    # results_all = {key: np.array([], 
    #         dtype=np.int64).reshape(0,evaluation_periods) for key in outputs}
    # run simulations
    for s in range(simulations):
        print("Simulation:", s+1, "/", simulations)
        results_sim, renters, landlords = runMonths(months,
            initialization_period, state_price, share_state_apartments, 
            inc_factor_state, outputs, max_increase)
        # append results from current simulation to arrays in dictionary
        results_all['mean_price'].append(results_sim['mean_price'])
        results_all['median_price'].append(results_sim['median_price'])
        results_all['vacancy_rate_p'].append(results_sim['vacancy_rate_p'])
        results_all['vacancy_rate_s'].append(results_sim['vacancy_rate_s'])
        results_all['vacancy_rate_t'].append(results_sim['vacancy_rate_t'])
        results_all['utility_p25'].append(results_sim['utility_p25'])
        results_all['utility_p50'].append(results_sim['utility_p50'])
        results_all['utility_p75'].append(results_sim['utility_p75'])      
    return(results_all, landlords, renters)

def runPostinvtervention(months, renters, landlords, max_increase): 
    """
    Method to simulate and evaluate the months after a policy intervention - in 
    case of the baseline simulations for the time after the 'non-intervention'.
    
    Parameters
    ----------
    months : integer
        number of months that the model should run after the 
        (non-)intervention
    renters : object
        pooplation of renters at the time right after the (non-)intervention 
    landlords : object
        population of landlords at the time right after the (non-)intervention 
    Returns
    -------
    results_post_int : dictionary
        contains all results of the months after (non-)intervention.
    """
    #outputs 
    outputs = ['mean_price','median_price', 'vacancy_rate_p', 'vacancy_rate_s',
               'vacancy_rate_t', 'utility_p25', 'utility_p50', 'utility_p75']
    # create dictionary with empty arrays to store results
    results_post_int = {key: [] for key in outputs}

    # simulate months (skip month 0 because process deviates for the 
    # first month  due to initialization -> not required here because model
    # was already initiated)
    for m in range(1, months+1):
        # print calculation progress
        if ((m+9) % 10 ==0) & (m+9 < months):
            print('   Months:',m,'-',m+9, '(of', months, 'months)')
        elif ((m+9) % 10 ==0) & (m+9 >= months):
            print('   Months:',m,'-', months, '(of', months, 'months)')
        renters, landlords = simulateMonth(renters, landlords, m, 
                                           inc_factor_state, max_increase, 
                                           state_price) 
        #evaluate each month after intervention
        results_m = evaluateMonth(renters,landlords)  
        # store results in dictionary
        results_post_int['mean_price'].append(results_m['mean_price'])
        results_post_int['median_price'].append(results_m['median_price'])
        results_post_int['vacancy_rate_p'].append(results_m['vacancy_rate_p'])
        results_post_int['vacancy_rate_s'].append(results_m['vacancy_rate_s'])
        results_post_int['vacancy_rate_t'].append(results_m['vacancy_rate_t'])
        results_post_int['utility_p25'].append(results_m['utility_p25'])
        results_post_int['utility_p50'].append(results_m['utility_p50'])
        results_post_int['utility_p75'].append(results_m['utility_p75'])  
    return(results_post_int)



def runIntervention(months_before_intervention, months_after_intervention, 
                      simulations, initialization_period,state_price, 
                      share_state_apartments, inc_factor_state, outputs,
                      new_apartments, max_increase):
    """
    Method that simulates and evaluates policy intervention at a specific point
    in time. It runs the simulations for the time before the intervention, for
    the time after the intervention as well as for the time after the 
    intervention without having implemented the intervention (no-intervention
    baseline case). The policy intervention is the construction of state 
    apartments.

    Parameters
    ----------
    months_before_intervention : integer
        defines for how many months the model should run per simulation before
        the intervention.
    months_after_intervention : integer
        defines for how many months the model should run per simulation after
        the (non-)intervention.
    simulations : integer
        defines number of simulations that should be run
    initialization_period : integer
        defines number of pre-intervention months per simulations where the 
        model runs without being evaluated yet. Set to 0 if entire 
        pre-intervention period should get evaluated.
    landlords_t0 : object
        population of landlords after the pre-intervention period. Identical 
        starting population is used for all simulations.
    renters_t0 : object
        population of renters after the pre-intervention period. Identical 
        starting population is used for all simulations.
    intervention : bool
        additional state apartments are constructed before simulations if the
        parameter is set to true. No additional apartments are constructed if
        the parameter is set to False.

    Returns
    -------
    results_int : dictionary
        dictionary that includes all results from the simulations.

    """
    # create dictionary with empty arrays to store results
    results_all = {key: []  for key in outputs}
    landlords_copies = []
    renters_copies = []
    
    # run simulations (pre intervention)
    for s in range(simulations):
        print("Simulations of pre-intervention period:", s + 1, '/', 
              simulations)
        results_sim, renters, landlords = runMonths(months_before_intervention,
            initialization_period, state_price, share_state_apartments, 
            inc_factor_state, outputs, max_increase)
        # append results from current simulation to arrays in dictionary
        results_all['mean_price'].append(results_sim['mean_price'])
        results_all['median_price'].append(results_sim['median_price'])
        results_all['vacancy_rate_p'].append(results_sim['vacancy_rate_p'])
        results_all['vacancy_rate_s'].append(results_sim['vacancy_rate_s'])
        results_all['vacancy_rate_t'].append(results_sim['vacancy_rate_t'])
        results_all['utility_p25'].append(results_sim['utility_p25'])
        results_all['utility_p50'].append(results_sim['utility_p50'])
        results_all['utility_p75'].append(results_sim['utility_p75']) 
        landlords_copies.append(copy.deepcopy(landlords))
        renters_copies.append(copy.deepcopy(renters))
    
    # create dictionaries with empty arrays to store results
    results_int = {key: []  for key in outputs}
    results_int_total = {key: []  for key in outputs}

    # run simulations after intervention
    for s in range(simulations): 
        print("Simulations with intervention:", s + 1, "/", simulations)
        # implement policy (construction) if intervention = True
        landlords = copy.deepcopy(landlords_copies[s])
        renters = copy.deepcopy(renters_copies[s])
        landlords = constructStateApartments(landlords, new_apartments, 
                                              state_price)
        results_s = runPostinvtervention(months_after_intervention, renters, 
                                          landlords, max_increase)
        # append results from current simulation
        results_int['mean_price'].append(results_s['mean_price'])
        results_int['median_price'].append(results_s['median_price'])
        results_int['vacancy_rate_p'].append(results_s['vacancy_rate_p'])
        results_int['vacancy_rate_s'].append(results_s['vacancy_rate_s'])
        results_int['vacancy_rate_t'].append(results_s['vacancy_rate_t'])
        results_int['utility_p25'].append(results_s['utility_p25'])
        results_int['utility_p50'].append(results_s['utility_p50'])
        results_int['utility_p75'].append(results_s['utility_p75']) 
    # combine results from pre intervention and post intervention
    results_int_total['mean_price'] = np.append(
        results_all['mean_price'], results_int['mean_price'], axis=1)
    results_int_total['median_price'] = np.append(
        results_all['median_price'], results_int['median_price'], axis=1)
    results_int_total['vacancy_rate_p'] = np.append(
        results_all['vacancy_rate_p'], results_int['vacancy_rate_p'], axis=1)
    results_int_total['vacancy_rate_s'] = np.append(
        results_all['vacancy_rate_s'], results_int['vacancy_rate_s'], axis=1)
    results_int_total['vacancy_rate_t'] = np.append(
        results_all['vacancy_rate_t'], results_int['vacancy_rate_t'], axis=1)
    results_int_total['utility_p25'] = np.append(
        results_all['utility_p25'], results_int['utility_p25'], axis=1)
    results_int_total['utility_p50'] = np.append(
        results_all['utility_p50'], results_int['utility_p50'], axis=1)
    results_int_total['utility_p75'] = np.append(
        results_all['utility_p75'], results_int['utility_p75'], axis=1)

    # create dictionaries with empty arrays to store results
    results_no_int = {key: []  for key in outputs}
    results_no_int_total = {key: []  for key in outputs}
    # run simulations after non-intervention
    for s in range(simulations): 
        print("Simulations without intervention:", s + 1, '/', simulations)
        # implement policy (construction) if intervention = True
        landlords = copy.deepcopy(landlords_copies[s])
        renters = copy.deepcopy(renters_copies[s])
        # simulate months after (non-)intervention
        results_s = runPostinvtervention(months_after_intervention, renters, 
                                          landlords, max_increase)
        # append results from current simulation
        results_no_int['mean_price'].append(results_s['mean_price'])
        results_no_int['median_price'].append(results_s['median_price'])
        results_no_int['vacancy_rate_p'].append(results_s['vacancy_rate_p'])
        results_no_int['vacancy_rate_s'].append(results_s['vacancy_rate_s'])
        results_no_int['vacancy_rate_t'].append(results_s['vacancy_rate_t'])
        results_no_int['utility_p25'].append(results_s['utility_p25'])
        results_no_int['utility_p50'].append(results_s['utility_p50'])
        results_no_int['utility_p75'].append(results_s['utility_p75']) 
    # combine results from pre intervention and post intervention
    results_no_int_total['mean_price'] = np.append(
        results_all['mean_price'], results_no_int['mean_price'], axis=1)
    results_no_int_total['median_price'] = np.append(
        results_all['median_price'], results_no_int['median_price'], axis=1)
    results_no_int_total['vacancy_rate_p'] = np.append(
        results_all['vacancy_rate_p'],results_no_int['vacancy_rate_p'], axis=1)
    results_no_int_total['vacancy_rate_s'] = np.append(
        results_all['vacancy_rate_s'],results_no_int['vacancy_rate_s'], axis=1)
    results_no_int_total['vacancy_rate_t'] = np.append(
        results_all['vacancy_rate_t'],results_no_int['vacancy_rate_t'], axis=1)
    results_no_int_total['utility_p25'] = np.append(
        results_all['utility_p25'],results_no_int['utility_p25'], axis=1)
    results_no_int_total['utility_p50'] = np.append(
        results_all['utility_p50'], results_no_int['utility_p50'], axis=1)
    results_no_int_total['utility_p75'] = np.append(
        results_all['utility_p75'], results_no_int['utility_p75'], axis=1)
    return(results_int_total, results_no_int_total, renters, landlords)

#%% [3] Visualization and Processing of OFAT results

def plotOFAT(xlabel, parameter_values, results_ofat):
    """
    Method to plot visualizations of all OFAT outcomes at once and store them.

    Parameters
    ----------
    xlabel : Used for the x-label in the plot (name of varying factor variable)
    parameter_values : different values that were tested for the factor
    results_ofat : the results from the ofat analyses

    """
    
    """Price Mean"""
    # use parameter values for x-axis
    x = np.array((parameter_values))
    #calculate mean over all months for each simulation 
    price_mean_sim = np.array(results_ofat['mean_price']).mean(axis=2)
    # calculate mean and std of simulation means for each parameter setting
    price_mean_parameter = price_mean_sim.mean(axis=1)
    price_std_parameter = price_mean_sim.std(axis=1)
    # plotting
    plt.ylabel('Mean price')
    plt.xlabel(xlabel, labelpad = 10)
    plt.title('Mean rent price (private sector)')
    plt.errorbar(x, price_mean_parameter, price_std_parameter, 
                 linestyle='None', marker='o', markersize=5, capsize=5)
    plt.xticks(parameter_values)
    plt.grid(color = '#DADEDF')
    # save plot
    plt.savefig(path_plots +'\\OFAT\\'+str(xlabel)+'_Mean price.png',dpi=500)
    # display plot
    plt.show()
    
    """Price Median"""
    #calculate mean over all months for each simulation 
    price_median_sim = np.array(results_ofat['median_price']).mean(axis=2)
    # calculate mean and std of simulation means for each parameter setting
    mean = price_median_sim.mean(axis=1)
    std = price_median_sim.std(axis=1)
    # plotting
    plt.ylabel('Median price')
    plt.xlabel(xlabel, labelpad = 10)
    plt.title('Median rent price (private sector)')
    plt.errorbar(x, mean, std, linestyle='None', marker='o', markersize=5, 
                 capsize=5)
    plt.xticks(parameter_values)
    plt.grid(color = '#DADEDF')
    # save plot
    plt.savefig(path_plots +'\\OFAT\\'+str(xlabel)+'_Median price.png',dpi=500)
    # display plot
    plt.show()
    
    """Vacancy"""
    # calculate mean over all months for each simulation 
    vp_mean_sim = np.array(results_ofat['vacancy_rate_p']).mean(axis=2)
    vs_mean_sim = np.array(results_ofat['vacancy_rate_s']).mean(axis=2)
    vt_mean_sim = np.array(results_ofat['vacancy_rate_t']).mean(axis=2)
    # calculate mean and std of simulation means for each parameter setting
    vp_mean_parameter = vp_mean_sim.mean(axis=1) 
    vp_std_parameter = vp_mean_sim.std(axis=1) 
    vs_mean_parameter = vs_mean_sim.mean(axis=1)
    vs_std_parameter = vs_mean_sim.std(axis=1)
    vt_mean_parameter = vt_mean_sim.mean(axis=1) 
    vt_std_parameter = vt_mean_sim.std(axis=1) 
    # plotting
    plt.ylabel('Vacancy rate [in %]')
    plt.xlabel(xlabel, labelpad = 10)
    plt.title('Vacancies')
    # calculate distance to avoid overlapping (jitter)
    delta = (parameter_values[1]-parameter_values[0])/9
    plt.errorbar(x-delta, vp_mean_parameter, vp_std_parameter, 
                 label='private sector', linestyle='None', marker='v', 
                 markersize=5, capsize=4)    
    plt.errorbar(x, vs_mean_parameter, vs_std_parameter, linestyle='None', 
                 marker='o', markersize=5, capsize=4, label = 'public sector')
    plt.errorbar(x+delta, vt_mean_parameter, vt_std_parameter, 
                 linestyle='None', marker='s', markersize=5, capsize=4, 
                 label='total', color = 'black')
    plt.xticks(parameter_values)
    # add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, 
               ncol=3, fontsize = 'small')
    # plt.legend(fontsize='small', loc = 'best', frameon = False)
    plt.grid(color = '#DADEDF')
    # save plot
    plt.savefig(path_plots +'\\OFAT\\'+str(xlabel)+'_Vacancies.png', dpi=500,
                bbox_inches = 'tight')
    #display plot
    plt.show()
  
    """Utility"""
    # calculate mean over all months for each simulation 
    u25_mean_sim = np.array(results_ofat['utility_p25']).mean(axis=2)
    u50_mean_sim = np.array(results_ofat['utility_p50']).mean(axis=2)
    u75_mean_sim = np.array(results_ofat['utility_p75']).mean(axis=2)
    
    # calculate mean of simulation means for each parameter setting
    u25_mean_parameter = u25_mean_sim.mean(axis=1)
    u50_mean_parameter = u50_mean_sim.mean(axis=1)
    u75_mean_parameter = u75_mean_sim.mean(axis=1)
    
    # calculate std of simulation means for each parameter setting
    u25_std_parameter = u25_mean_sim.std(axis=1)
    u50_std_parameter = u50_mean_sim.std(axis=1)
    u75_std_parameter = u75_mean_sim.std(axis=1) 

    # configure plot with three suplots
    fig, ((ax0), (ax1), (ax2)) = plt.subplots(3, 1, figsize = (9,12))
    fig.subplots_adjust(hspace = 0.2)
    # 1st subplot for high-income housholds
    ax0.set_title('(a) High-income households', fontsize = 16)
    ax0.grid(color = '#DADEDF')
    ax0.errorbar(x, u75_mean_parameter, u75_std_parameter, linestyle='None', 
                 marker='o', markersize=5, capsize=5, color='#0C3823')
    ax0.xaxis.set_ticks(parameter_values)
    ax0.xaxis.set_ticklabels([])
    
    # 2nd subplot for middle-income housholds
    ax1.set_ylabel('Utility', fontsize = 14, labelpad = 20)
    ax1.set_title('(b) Middle-income households', fontsize = 16)
    ax1.grid(color = '#DADEDF')
    ax1.errorbar(x, u50_mean_parameter, u50_std_parameter, linestyle='None', 
                 marker='o', markersize=5, capsize=5, color = '#358856')
    ax1.xaxis.set_ticks(parameter_values)
    ax1.xaxis.set_ticklabels([])
    
    # 3rd subplot for low-income housholds
    ax2.set_xlabel(xlabel, labelpad = 20, fontsize = 14)
    ax2.xaxis.set_ticks(parameter_values)
    ax2.xaxis.set_ticklabels(parameter_values)
    ax2.set_title('(c) Low-income households', fontsize = 16)
    ax2.grid(color = '#DADEDF')
    ax2.errorbar(x, u25_mean_parameter, u25_std_parameter, linestyle='None', 
                 marker='o', markersize=5, capsize=5, color='#62BD69')

    # save plot
    plt.savefig(path_plots +'\\OFAT\\'+str(xlabel)+'_Utility.png',dpi=500)
    # display plot
    plt.show()
  
def tabulateResults(mean_values, parameter_values):
    """
    Method to calculates significance tables for OFAT results.

    Parameters
    ----------
    mean_values : list of arrays
        contains results from all simulations, all tested parameter values for
        one outcome.
    parameter_values : list
        list that contains the different parameters for the factor that were
        used for the OFAT simulations.

    Returns
    -------
    diff_pval_tables : dataframe
        dataframe containing all significance tables.
    """
    # Get means of simulations (over months) for each parameter setting
    #mean_values=results_ofat['mean_price']
    sim_mean = np.array(mean_values).mean(axis=2)
    # Get p-values
    p_values = np.zeros((len(parameter_values),len(parameter_values)))
    for i in range(len(sim_mean)):
        for j in range(len(sim_mean)):
            p_values[i,j] = stats.ttest_ind(sim_mean[i], sim_mean[j],axis=0)[1]
    # get mean deviations
    mean_diff = np.zeros((len(parameter_values),len(parameter_values)))
    for i in range(len(sim_mean)):
        for j in range(len(sim_mean)):
            mean_diff[i,j] = round(sim_mean[i].mean()-sim_mean[j].mean(),2)
    # combine p-values and mean deviations in one table
    diff_pval_table = mean_diff.tolist()
    for i in range(len(mean_diff)):
        for j in range(len(mean_diff)):
            if p_values[i,j] < 0.01:
                diff_pval_table[i][j] = str(mean_diff[i,j]) + " ***"
            elif p_values[i,j] < 0.05:
                diff_pval_table[i][j] = str(mean_diff[i,j]) + " **"
            elif p_values[i,j] < 0.1:
                diff_pval_table[i][j] = str(mean_diff[i,j]) + " *"
         
    # format table
    diff_pval_tables = pd.DataFrame(
        diff_pval_table, columns = parameter_values, 
        index = parameter_values)            
    return(diff_pval_tables)

def saveResults(filename, path, results):
    """
    Method to store the resulting table into an excel files. Excel file will
    consist of different worksheets containing the tables.
    Parameters.
    """
    # define file name
    file = path + filename
    # write data to excel files (separately for the different worksheets)
    with pd.ExcelWriter(file) as writer:
        results[0].to_excel(writer, sheet_name="Mean Prices", index=True)
        results[1].to_excel(writer, sheet_name="Median Prices", index=True)
        results[2].to_excel(writer, sheet_name="Vacancies private", index=True)
        results[3].to_excel(writer, sheet_name="Vacancies public", index=True)
        results[4].to_excel(writer, sheet_name="Vacancies total", index=True)
        results[5].to_excel(writer, sheet_name="Utility (low-income)", 
                            index=True)
        results[6].to_excel(writer, sheet_name="Utility (middle-income)", index=True)
        results[7].to_excel(writer, sheet_name="Utility (high-income)", 
                            index=True)

#%% [4] Visualization and Processing of Ceteris Paribus Analysis

def plotIntervention(title, ylabel, output, results_int, results_no_int, 
                      months_before_intervention, months_after_intervention,
                      pre_month_included, new_apartments):
    """
    Method to plot and store ceteris paribus analyses results (separately for 
    one outcome).

    Parameters
    ----------
    title : string
        Is used as the title for the plot.
    ylabel : string
        Is used as the y label of the plot.
    output : string
        Specifies key for the dictionary (for the outcome of interest) to
        retrieve the corresponding data.
    results_int : dict
        results from intervention (treatment group).
    results_no_int : dict
        results from non-intervention (control group).
    months_before_intervention : int
        how many months before the intervention the model has run.
    months_after_intervention : int
        how many months after the intervention have been evaluated.
    pre_month_included : int
        defines how many months before the intervention should be plotted.
        Method will use the specified number of months closest to intervention
        for plotting.
    new_apartments : int
        how many new apartments have been constructed. To name the files
        correctly that store the results.

    """
    # calculate mean and standard deviation
    means_int = np.array(results_int[output]).mean(axis=0)
    std_int = np.array(results_int[output]).std(axis=0)
    means_no_int = np.array(results_no_int[output]).mean(axis=0)
    std_no_int = np.array(results_no_int[output]).std(axis=0)
    # calculate confidence interval based on std
    ci_int = 1.96 * std_int / np.sqrt(len(std_int))
    ci_no_int = 1.96 * std_no_int / np.sqrt(len(std_no_int))
    
    # plot results
    fig, ax = plt.subplots()
    # before intervention
    x = np.arange(-pre_month_included+1, 1)
    y_start = months_before_intervention-pre_month_included
    y_end = months_before_intervention
    ax.plot(x, means_int[y_start:y_end], label= 'before intervention', 
            color = 'gray')  
    ax.fill_between(x, (means_int[y_start:y_end] - ci_int[y_start:y_end]), 
                (means_int[y_start:y_end] + ci_int[y_start:y_end]), 
                color='gray', alpha=.2)
    # vertical line for intervention
    ax.axvline(x=0, label = 'intervention', color = 'black', 
                linestyle = 'dashed')
    # after intervention
    x = np.arange(0, months_after_intervention + 1)
    y_start = months_before_intervention - 1
    ax.plot(x, means_int[y_start:],label='with construction', color = 'purple')
    ax.fill_between(x, (means_int[y_start:] - ci_int[y_start:]), 
                (means_int[y_start:] + ci_int[y_start:]), color='purple', 
                alpha=.2)
    # after non-intervention
    ax.plot(x, means_no_int[y_start:],label='without construction', 
            color = 'g')
    ax.fill_between(x, (means_no_int[y_start:] - ci_no_int[y_start:]), 
                    (means_no_int[y_start:] + ci_no_int[y_start:]), 
                    color='g', alpha=.2)
    plt.ylabel(ylabel)
    plt.xlabel('Months')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),fancybox=True, 
           ncol=2)
    # save plot 
    plt.savefig(path_plots +'\\Policy Intervention\\' + str(new_apartments) +
                '_' + output + '_development.png',dpi=500)
    # display plot
    plt.show()
    
def subplotIntervention(results_int_total, results_no_int_total, 
                        new_apartments, months_before_intervention, 
                        months_after_intervention, pre_month_included):
    """
    Method to plot and save ceteris paribus analyses results combined for 
    utilities (low-, middle-, high income), and for vacancies (public, private, 
    total).

    Parameters
    ----------

    results_int_total : dict
        results from intervention (treatment group).
    results_no_int_total : dict
        results from non-intervention (control group).
    new_apartments : TYPE
        how many new apartments have been constructed. To name the files
        correctly that store the results.
    months_before_intervention : int
        how many months before the intervention the model has run.
    months_after_intervention : int
        how many months after the intervention have been evaluated.
    pre_month_included : int
        defines how many months before the intervention should be plotted.
        Method will use the specified number of months closest to intervention
        for plotting.

    """
    # combine the utility plots in one figure with 3 subplots
    # prepare plot
    fig, ((ax0), (ax1), (ax2)) = plt.subplots(3, 1, figsize = (9,12))
    fig.subplots_adjust(hspace = 0.2)
    pre_month_included = 10
    x_pre = np.arange(-pre_month_included+1, 1)
    y_start_pre = months_before_intervention-pre_month_included
    y_end_pre = months_before_intervention
    x_post = np.arange(0, months_after_intervention + 1)
    y_start_post = months_before_intervention - 1
    
    """subplot for high-income housholds"""
    # calculate mean and standard deviation 
    means_int = np.array(results_int_total['utility_p75']).mean(axis=0)
    std_int = np.array(results_int_total['utility_p75']).std(axis=0)
    means_no_int = np.array(results_no_int_total['utility_p75']).mean(axis=0)
    std_no_int = np.array(results_no_int_total['utility_p75']).std(axis=0)
    
    # calculate confidence interval based on std
    ci_int = 1.96 * std_int / np.sqrt(len(std_int))
    ci_no_int = 1.96 * std_no_int / np.sqrt(len(std_no_int))
    
    # subplot for calculate mean and standard deviation for high-income houshold
    ax0.set_title('(a) High-income households', fontsize = 16)
    ax0.grid(color = '#DADEDF')
    ax0.axvline(x=0, label = 'intervention', color = 'black', 
                linestyle = 'dashed')
    
    # before intervention
    ax0.plot(x_pre, means_int[y_start_pre:y_end_pre], 
             label= 'before intervention', color = 'gray') 
    ax0.fill_between(x_pre, 
            (means_int[y_start_pre:y_end_pre] - ci_int[y_start_pre:y_end_pre]), 
            (means_int[y_start_pre:y_end_pre] + ci_int[y_start_pre:y_end_pre]), 
            color='gray', alpha=.2)
    ax0.xaxis.set_ticklabels([])
    # post intervention
    ax0.plot(x_post, means_int[y_start_post:],label='with construction', 
             color = 'purple')
    ax0.fill_between(x_post, 
                (means_int[y_start_post:] - ci_int[y_start_post:]), 
                (means_int[y_start_post:] + ci_int[y_start_post:]), 
                color='purple', alpha=.2)
    # post non-intervention
    ax0.plot(x_post, means_no_int[y_start_post:],label='without construction', 
            color = 'g')
    ax0.fill_between(x_post, 
                    (means_no_int[y_start_post:] - ci_no_int[y_start_post:]), 
                    (means_no_int[y_start_post:] + ci_no_int[y_start_post:]), 
                    color='g', alpha=.2)
    
    """ Subplot for middle-income housholds"""
    # calculate mean and standard deviation 
    means_int = np.array(results_int_total['utility_p50']).mean(axis=0)
    std_int = np.array(results_int_total['utility_p50']).std(axis=0)
    means_no_int = np.array(results_no_int_total['utility_p50']).mean(axis=0)
    std_no_int = np.array(results_no_int_total['utility_p50']).std(axis=0)
    
    # calculate confidence interval based on std
    ci_int = 1.96 * std_int / np.sqrt(len(std_int))
    ci_no_int = 1.96 * std_no_int / np.sqrt(len(std_no_int))
    
    # subplot 
    ax1.set_title('(b) Middle-income households', fontsize = 16)
    ax1.grid(color = '#DADEDF')
    ax1.set_ylabel('Utility', fontsize = 14, labelpad = 10)
    ax1.axvline(x=0, label = 'intervention', color = 'black', 
                linestyle = 'dashed')
    # before intervention
    ax1.plot(x_pre, means_int[y_start_pre:y_end_pre], 
             label= 'before intervention', color = 'gray') 
    ax1.fill_between(x_pre, 
            (means_int[y_start_pre:y_end_pre] - ci_int[y_start_pre:y_end_pre]), 
            (means_int[y_start_pre:y_end_pre] + ci_int[y_start_pre:y_end_pre]), 
            color='gray', alpha=.2)
    ax1.xaxis.set_ticklabels([])
    # post intervention
    ax1.plot(x_post, means_int[y_start_post:],label='with construction', 
             color = 'purple')
    ax1.fill_between(x_post, (means_int[y_start_post:] - ci_int[y_start_post:]), 
                (means_int[y_start_post:] + ci_int[y_start_post:]), 
                color='purple', alpha=.2)
    # post non-intervention
    ax1.plot(x_post, means_no_int[y_start_post:],label='without construction', 
            color = 'g')
    ax1.fill_between(x_post, 
                    (means_no_int[y_start_post:] - ci_no_int[y_start_post:]), 
                    (means_no_int[y_start_post:] + ci_no_int[y_start_post:]), 
                    color='g', alpha=.2)
    
    """Subplot for low-income housholds"""
    # calculate mean and standard deviation 
    means_int = np.array(results_int_total['utility_p25']).mean(axis=0)
    std_int = np.array(results_int_total['utility_p25']).std(axis=0)
    means_no_int = np.array(results_no_int_total['utility_p25']).mean(axis=0)
    std_no_int = np.array(results_no_int_total['utility_p25']).std(axis=0)
    
    # calculate confidence interval based on std
    ci_int = 1.96 * std_int / np.sqrt(len(std_int))
    ci_no_int = 1.96 * std_no_int / np.sqrt(len(std_no_int))
    
    # subplot 
    ax2.set_title('(c) Low-income households', fontsize = 16)
    ax2.grid(color = '#DADEDF')
    # before intervention
    ax2.plot(x_pre, means_int[y_start_pre:y_end_pre], 
             label= 'before intervention',color = 'gray') 
    ax2.fill_between(x_pre, 
            (means_int[y_start_pre:y_end_pre] - ci_int[y_start_pre:y_end_pre]), 
            (means_int[y_start_pre:y_end_pre] + ci_int[y_start_pre:y_end_pre]), 
            color='gray', alpha=.2)
    ax2.axvline(x=0, label = 'intervention', color = 'black', 
                linestyle = 'dashed')
    # post intervention
    ax2.plot(x_post, means_int[y_start_post:],label='with construction', 
             color = 'purple')
    ax2.fill_between(x_post, 
            (means_int[y_start_post:] - ci_int[y_start_post:]), 
            (means_int[y_start_post:] + ci_int[y_start_post:]), color='purple', 
            alpha=.2)
    # post non-intervention
    ax2.plot(x_post, means_no_int[y_start_post:],label='without construction', 
            color = 'g')
    ax2.fill_between(x_post, 
                    (means_no_int[y_start_post:] - ci_no_int[y_start_post:]), 
                    (means_no_int[y_start_post:] + ci_no_int[y_start_post:]), 
                    color='g', alpha=.2)
    
    # add legend
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),fancybox=True, 
               ncol=2)
    
    # set x-label for all subplots
    plt.xlabel('Months', fontsize = 14, labelpad = 10)
    
    # save plot 
    plt.savefig(path_plots +'\\Policy Intervention\\' + str(new_apartments) +
                '_utility_development.png', bbox_inches='tight', dpi=500)
    # display plot
    plt.show()
    
    # Combine the vacancy rate plots in one figure with three subplots
    
    # prepare plot
    fig, ((ax0), (ax1), (ax2)) = plt.subplots(3, 1, figsize = (9,12))
    fig.subplots_adjust(hspace = 0.2)
    
    """Subplot for private vacancy rate"""
    # calculate mean and standard deviation 
    means_int = np.array(results_int_total['vacancy_rate_p']).mean(axis=0)
    std_int = np.array(results_int_total['vacancy_rate_p']).std(axis=0)
    means_no_int = np.array(
        results_no_int_total['vacancy_rate_p']).mean(axis=0)
    std_no_int = np.array(results_no_int_total['vacancy_rate_p']).std(axis=0)
    
    # calculate confidence interval based on std
    ci_int = 1.96 * std_int / np.sqrt(len(std_int))
    ci_no_int = 1.96 * std_no_int / np.sqrt(len(std_no_int))
    
    # subplot 
    ax0.set_title('(a) Private sector housing units', fontsize = 16)
    ax0.grid(color = '#DADEDF')
    ax0.axvline(x=0, label = 'intervention', color = 'black', 
                linestyle = 'dashed')
    
    # before intervention
    ax0.plot(x_pre, means_int[y_start_pre:y_end_pre], 
             label= 'before intervention', color = 'gray') 
    ax0.fill_between(x_pre, 
            (means_int[y_start_pre:y_end_pre] - ci_int[y_start_pre:y_end_pre]), 
            (means_int[y_start_pre:y_end_pre] + ci_int[y_start_pre:y_end_pre]), 
            color='gray', alpha=.2)
    ax0.xaxis.set_ticklabels([])
    # post intervention
    ax0.plot(x_post, means_int[y_start_post:],label='with construction', 
             color = 'purple')
    ax0.fill_between(x_post, 
                (means_int[y_start_post:] - ci_int[y_start_post:]), 
                (means_int[y_start_post:] + ci_int[y_start_post:]), 
                color='purple', alpha=.2)
    # post non-intervention
    ax0.plot(x_post, means_no_int[y_start_post:],label='without construction', 
            color = 'g')
    ax0.fill_between(x_post, 
                    (means_no_int[y_start_post:] - ci_no_int[y_start_post:]), 
                    (means_no_int[y_start_post:] + ci_no_int[y_start_post:]), 
                    color='g', alpha=.2)
    
    """Subplot for vacancy rate in public sector"""
    # calculate mean and standard deviation 
    means_int = np.array(results_int_total['vacancy_rate_s']).mean(axis=0)
    std_int = np.array(results_int_total['vacancy_rate_s']).std(axis=0)
    means_no_int = np.array(
        results_no_int_total['vacancy_rate_s']).mean(axis=0)
    std_no_int = np.array(results_no_int_total['vacancy_rate_s']).std(axis=0)
    
    # calculate confidence interval based on std
    ci_int = 1.96 * std_int / np.sqrt(len(std_int))
    ci_no_int = 1.96 * std_no_int / np.sqrt(len(std_no_int))
    
    # subplot for calculate mean and standard deviation for high-income houshold
    ax1.set_title('(b) Public sector housing units', fontsize = 16)
    ax1.grid(color = '#DADEDF')
    ax1.set_ylabel('Vacancy rate [in %]', fontsize = 14, labelpad = 10)
    ax1.axvline(x=0, label = 'intervention', color = 'black', 
                linestyle = 'dashed')
    # before intervention
    ax1.plot(x_pre, means_int[y_start_pre:y_end_pre], 
             label= 'before intervention', color = 'gray') 
    ax1.fill_between(x_pre, 
            (means_int[y_start_pre:y_end_pre] - ci_int[y_start_pre:y_end_pre]), 
            (means_int[y_start_pre:y_end_pre] + ci_int[y_start_pre:y_end_pre]), 
            color='gray', alpha=.2)
    ax1.xaxis.set_ticklabels([])
    # post intervention
    ax1.plot(x_post, means_int[y_start_post:],label='with construction', 
             color = 'purple')
    ax1.fill_between(x_post, (means_int[y_start_post:] - ci_int[y_start_post:]), 
            (means_int[y_start_post:] + ci_int[y_start_post:]), color='purple', 
            alpha=.2)
    # post non-intervention
    ax1.plot(x_post, means_no_int[y_start_post:],label='without construction', 
            color = 'g')
    ax1.fill_between(x_post, 
                    (means_no_int[y_start_post:] - ci_no_int[y_start_post:]), 
                    (means_no_int[y_start_post:] + ci_no_int[y_start_post:]), 
                    color='g', alpha=.2)
    
    """"Subplot for total vacancy rate"""
    # calculate mean and standard deviation 
    means_int = np.array(results_int_total['vacancy_rate_t']).mean(axis=0)
    std_int = np.array(results_int_total['vacancy_rate_t']).std(axis=0)
    means_no_int = np.array(
        results_no_int_total['vacancy_rate_t']).mean(axis=0)
    std_no_int = np.array(results_no_int_total['vacancy_rate_t']).std(axis=0)
    
    # calculate confidence interval based on std
    ci_int = 1.96 * std_int / np.sqrt(len(std_int))
    ci_no_int = 1.96 * std_no_int / np.sqrt(len(std_no_int))
    
    # subplot 
    ax2.set_title('(c) All housing units', fontsize = 16)
    ax2.grid(color = '#DADEDF')
    # before intervention
    ax2.plot(x_pre, means_int[y_start_pre:y_end_pre], 
             label= 'before intervention', color = 'gray') 
    ax2.fill_between(x_pre, 
            (means_int[y_start_pre:y_end_pre] - ci_int[y_start_pre:y_end_pre]), 
            (means_int[y_start_pre:y_end_pre] + ci_int[y_start_pre:y_end_pre]), 
            color='gray', alpha=.2)
    ax2.axvline(x=0, label = 'intervention', color = 'black', 
                linestyle = 'dashed')
    # post intervention
    ax2.plot(x_post, means_int[y_start_post:],label='with construction', 
             color = 'purple')
    ax2.fill_between(x_post,(means_int[y_start_post:] - ci_int[y_start_post:]), 
            (means_int[y_start_post:] + ci_int[y_start_post:]), color='purple', 
            alpha=.2)
    # post non-intervention
    ax2.plot(x_post, means_no_int[y_start_post:],label='without construction', 
            color = 'g')
    ax2.fill_between(x_post, 
                    (means_no_int[y_start_post:] - ci_no_int[y_start_post:]), 
                    (means_no_int[y_start_post:] + ci_no_int[y_start_post:]), 
                    color='g', alpha=.2)
    
    # add legend
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),fancybox=True, 
               ncol=2)
    
    # set x-label for all subplots
    plt.xlabel('Months', fontsize = 14, labelpad = 10)
    
    # save plot 
    plt.savefig(path_plots +'\\Policy Intervention\\' + str(new_apartments) +
                '_vacancies_development.png', bbox_inches='tight', dpi=500)
    # display plot
    plt.show()

def tableIntervention_results(output, results_int, results_no_int, 
                              months_before_intervention, 
                              months_after_intervention):
    """
    Method to create table with a comparison of the resulting mean values 
    for control and treatment group and to store these tables in a dataframe.
    Data frame contains mean differences as well as significance levels.

    Parameters
    ----------
    output : string
        outcome of interest to create a table for (e.g. 'mean price')
    results_int : dict
        results from intervention (treatment group)
    results_no_int : dict
        results from non-intervention (control group).
    months_before_intervention : int
        how many months before the intervention the model has run. Required to
        retrieve the right data.
    months_after_intervention : int
        how many months after the intervention have been run per simulation.
        Required to retrieve right data.

    Returns
    -------
    diff_pval_tables: dataframe
        contains mean differences by quarter after simulation and
        corresponding significance level.

    """
    # define index start and end for results
    start = months_before_intervention
    end = months_before_intervention + months_after_intervention
    # Get p-values
    p_values = []
    # steps of 4 to combine quarters (less impacted by random fluctuations )
    for i in range(start, end, 4):
        p_values.append(stats.ttest_ind(
            results_int[output][:,i:i+4].mean(axis=1),
            results_no_int[output][:,i:i+4].mean(axis=1))[1])
        
    # get intervention mean
    mean_int = []
    for i in range(start, end, 4):
        mean_int.append(round(
            results_int[output][:,i:i+4].mean(), 2))   
    
    # get non-intervention mean
    mean_no_int = []
    for i in range(start, end, 4):
        mean_no_int.append(round(
            results_no_int[output][:,i:i+4].mean(), 2)) 
            
    # get mean difference
    mean_diff = []
    for i in range(start, end, 4):
        mean_diff.append(round(
            results_int[output][:,i:i+4].mean() - (
            results_no_int[output][:,i:i+4].mean()),2))
    
    # combine p-values and mean deviations in one table
    for i in range(len(mean_diff)):
        if p_values[i] < 0.01:
            mean_diff[i] = str(mean_diff[i]) + " ***"
        elif p_values[i] < 0.05:
            mean_diff[i] = str(mean_diff[i]) + " **"
        elif p_values[i] < 0.1:
            mean_diff[i] = str(mean_diff[i]) + " *"
    
    # get index labels for quarters
    quarters = []
    for i in range(50,98,4):
        quarters.append(str(i-50) + '-' + str(i-46))
    
    # combine means and mean difference to one array
    data_mean_diff = np.vstack((mean_int, mean_no_int, mean_diff)).T
    # create table
    diff_pval_tables = pd.DataFrame(data_mean_diff, index = quarters, 
                                    columns = ['Mean with construction',
                                               'Mean without construction', 
                                               'Mean difference'])  
    return(diff_pval_tables)       