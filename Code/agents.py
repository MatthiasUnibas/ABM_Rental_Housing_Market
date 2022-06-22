#%% AGENTS
#%%

""" 
This file contains the classes - specifically the class of renters and the 
class of landlords. These classes will be used to create the objects which
contain the entire population of landlords respectively renters.
"""

#%% [0] Import required modules

import numpy as np
import numpy.random as rd

#%% [1] Class for population of landlords

class Landlords(): 
    # Declare instance variables
    def __init__(self, private = None, apartment=None, quality=None, 
                 price=None, available=None, random=None):
       self.private = private
       self.apartment = apartment
       self.quality = quality
       self.price = price 
       self.available = available
       self.random = random
    

    # Define instance methods 
    def setPrice(self, max_increase, q_threshold, min_n_comparable):
        """
        Private landlords set the price for their available apartments. First,
        they check the mean market price for apartments of similar quality. 
        Next,they set the apartment price to the current market price, except 
        if the market price would increase the previous price by more than the 
        maximum increase factor that is legally allowed. In that case, they 
        will simply increase the market price by the maximum allowed increase 
        factor.
        The price setting will only be applied if there is at least one 
        apartment rented on the market (not the case in 1st iteration). Since
        landlords only consider prices of rented apartments for the calculation
        of the current market price.
        """
        # Check if there are any rented apartments
        if len(np.where(self.available==False)[0]) > 0:
            # get index of private landlords with an available apartment
            index_price_setters = np.where(
                (self.available==True) & (self.private==True))
            # loop through landlords who need to set a new price
            for l in index_price_setters[0]:
                # set comparable quality threshold such that 10% is covered
                q_c = (max(self.quality) - min(self.quality)) * q_threshold
                # check if 10 or more rented apartments of comparable quality 
                while len(np.where((self.quality>self.quality[l]-q_c)
                                & (self.quality<self.quality[l]+q_c)
                                & (self.available==False) 
                                & (self.private==True))[0]) < min_n_comparable:
                    # increase comparable quality threshold if needed
                    q_c += q_c
                # calculate current market price
                current_market_price = np.mean(self.price[np.where(
                    (self.quality>self.quality[l]-q_c)
                    & (self.quality<self.quality[l]+q_c) 
                    & (self.available==False) 
                    & (self.private==True))])
                # print(self.price[l], current_market_price, q_c)
                # check if the apartment is new on the market:
                if self.price[l] == 0:
                    # set original price slightly above the market price
                    self.price[l] = current_market_price * max_increase
                # if previously above market price, try to keep high price first
                elif self.price[l] > current_market_price:
                    self.price[l] = self.price[l]
                # ensure that price is not more increased than legally allowed
                elif current_market_price > max_increase*self.price[l]:
                    self.price[l] *= max_increase
                # if the legal restrictions don't apply, use the market price
                else:
                    self.price[l] = current_market_price
            
    def selectTenant(self, renters, apartment_information, applicants):
        """
        Landlords select a new tenant among the applicants. The selection is a 
        random choice among the applicants. Applications to other apartments 
        are revoked after a tenant has been selected by a landlord (avoiding 
        double selection). Income criterion does not need to be checked for
        state apartments, because only households that fulfill the income
        criterion are allowed to apply.
        """
        # Loop through application list (apartment by apartment)
        for i in range(len(apartment_information)):
            # Skip selection if there are no applicants left
            if len(applicants[i]) > 0:
                # retrieve index of randomly chosen applicant
                selected_tenant_index = rd.choice(
                    np.arange(len(applicants[i])))
                # retrieve selected applicants uid
                selected_r = int(applicants[i][selected_tenant_index])
                # store landlords apartment number
                a = apartment_information[i][0]  
                
                # update price, search status and apartment for selected tenant
                renters.apartment[np.where(renters.uid == selected_r)] = a 
                renters.searching[np.where(renters.uid == selected_r)] = False
                renters.price[np.where(renters.apartment == a)] = self.price[
                    np.where(self.apartment == a)]
                renters.quality[np.where(
                    renters.apartment == a)] = self.quality[np.where(
                        self.apartment == a)]
                
                # update availability of landlord after tenant selection
                self.available[np.where(self.apartment == a)] = False
                #remove selected renter from other applications
                for j in range(len(apartment_information)):
                    if selected_r in applicants[j]:
                        index_remove = applicants[j].tolist().index(selected_r)
                        applicants[j] = np.delete(applicants[j], index_remove)
        return(renters)
    
    def decreasePrice(self, rent_decrease_factor):
        """ 
        Method for price reduction of available apartments of landlords by a 
        constant factor (<1). It will be applied in case the apartment was
        offered on the market in the previous cycle, but not successfully
        rented. Method only applies for private landlords because state
        apartments' prices are fixed.
        """
        self.price = np.where((self.available == True) & (self.private==True), 
                              self.price * rent_decrease_factor, self.price)
        
    def updatePrice(self, renters, prob_increase, max_increase): 
        """
        Draw random values ([0,1)) for all landlords, and then update the price
        for apartments of private landlords by factor max_increase, if the 
        randomly drawn value is below the parameter prob_increase. 
        The apartment price will also get updated for renters currently living
        in one of these apartments.
        IMPORTANT: Function has been disabled by setting prob_increase to 0
        due to the theoretical background applied (tenancy rent control). 
        However, function was not removed, such that same results can be
        replicated (because the experiments reported in the thesis have been
        performed with updatePrice and a prob_increase value of 0). Deleting
        the function would make the model slightly more efficient, still lead
        to principally the same results, BUT the results would still slightly 
        deviate due to the random seed changing compared to the presented 
        results in the thesis.
        """       
        self.random = rd.rand(len(self.apartment)) 
        # retrieve 'IDs' of apartments whose price will be increased
        apartment_increase = self.apartment[np.where(
            (self.random < prob_increase) & (self.private==True))] 
        # increase apartment price for landlords
        self.price[np.where(np.isin(self.apartment,
                                    apartment_increase))] *= max_increase
        # increase price for current renters of affected apartments
        renters.price[np.where(np.isin(renters.apartment,
                                        apartment_increase))] *= max_increase
        return(renters)

        

#%%% [2] Class for population of renters

class Renters: 
    # Declare instance variables
    def __init__(self, uid=None, apartment=None, price=None, quality=None, 
                 searching=None,income=None, random=None, preferences=None, 
                 utility=None):
        self.uid = uid
        self.apartment = apartment
        self.price = price 
        self.quality = quality
        self.searching = searching
        self.income = income
        self.random = random
        self.preferences = preferences
        self.utility = utility
    
    # Define instance methods 
    def updateIncome(self, prob_income_change, income_change, income_min,
                     income_max):
        """" 
        Renters' income is updated. The affected renters are chosen by a 
        stochastic process each period. The parameter 'prob_income_change' 
        defines the likelihood of an income change (in one period). The income 
        might either increase or decrease. The income changes are to the power 
        of e to ensure that income distribution remains stable over time. 
        Furthermore, incomes cannot exceed the maximum income, and cannot 
        become lower than the minimum income.
        """
        self.random = rd.rand(len(self.uid))
        self.income = np.where(self.random < prob_income_change, 
                               np.exp(rd.randn()*income_change)*self.income, 
                               self.income)
        self.income = np.where(self.income < income_min,income_min,self.income)
        self.income = np.where(self.income > income_max,income_max,self.income)

    def updateUtility(self):
        """" 
        Renters' current utility level is recalculated based on their current
        income, apartment's quality, apartment's price and their preferences. 
        It corresponds to a Cobb-Douglas utility function.
        """
        self.utility = self.quality**self.preferences * (
            (self.income-self.price).clip(min=0)**(1-self.preferences))

    def checkAffordability(self, landlords, max_rent_share):
       """ 
       Renters check if they can still afford their current apartment. This is
       the case if their income exceeds the rental price at least by factor 2. 
       If that is no longer the case they move out and inform their landlords
       about the change.
       """
       # get apartment ID of unaffordable apartments
       apartments = self.apartment[np.where(
           (self.price != None) & (max_rent_share * self.income < self.price))]
       #print("Sum of no longer affordable apartments:", len(apartments))
       # change renters status, price, quality and ap. if no longer affordable
       self.searching = np.where(np.isin(self.apartment, apartments),
                                 True, self.searching)
       self.price = np.where(np.isin(self.apartment, apartments), 
                             0, self.price)
       self.quality = np.where(np.isin(self.apartment, apartments),
                               0, self.quality)
       self.apartment = np.where(np.isin(self.apartment, apartments),
                                 -1, self.apartment)
       # Update availability of Landlords apartment
       landlords.available = np.where(np.isin(landlords.apartment, apartments),
                                      True, landlords.available)
       return(landlords)


    def moveRandomly(self, landlords, prob_random_move):
        """ 
        Stochastic process that determines which renters decide to move out 
        from their current apartment. The method ensures that the renters who
        move out as well as their landlords are updated accordingly.
        """
        # update random numbers for selection of renters
        self.random = rd.rand(len(self.uid))
        # get apartment IDs of renters moving out (randomly selected)
        apartments = self.apartment[np.where((self.random < prob_random_move)
                                             & (self.searching == False))]
        #print("Number of random movers", len(apartments))
        # update search status, price, quality and apartment for random movers
        self.searching = np.where(np.isin(self.apartment,apartments),
                                  True,self.searching)
        self.price = np.where(np.isin(self.apartment,apartments), 
                              0, self.price)
        self.quality = np.where(np.isin(self.apartment,apartments),
                                0,self.quality)
        self.apartment = np.where(np.isin(self.apartment,apartments),
                                  -1,self.apartment)   
        # Update availability of Landlords apartment
        landlords.available = np.where(np.isin(landlords.apartment,apartments),
                                       True,landlords.available)
        return(landlords)


    def screenMarket(self, landlords, screener_share, req_utility_improvement,
                     req_n_preferred_options):
        """
        This method lets renters currently living in an apartment sporadically
        check the apartment market and evaluate if the available options are
        good enough (quantity and utility wise) so that it is worth for them to
        move out and search for a new home.
        """
        # prepare arrays with information of available apartments
        prices = landlords.price[np.where(landlords.available == True)]
        quality = landlords.quality[np.where(landlords.available == True)]
        # Get index of households living in an apartment (potential screeners)
        idx_pot_screeners = np.where(self.searching==False)
        # Randomly draw sample of potential screeners (actually screening)
        idx_screening = rd.choice(idx_pot_screeners[0], 
            size = round(len(idx_pot_screeners[0])*screener_share),
            replace=False)
        
        # check for all screeners if they have room for improvement:
        for r in range(len(idx_screening)):
            # calculate utility for available apartments (utility alternative)
            utility_alt = quality**self.preferences[idx_screening[r]] * (
                (self.income[idx_screening[r]]-prices).clip(min=0)**(
                1-self.preferences[idx_screening[r]]))
            number_prefered_apartments = len(np.where(utility_alt > 
                    self.utility[idx_screening[r]])[0]*req_utility_improvement)
            # screener leaves if enough preferred options are available
            if number_prefered_apartments >= req_n_preferred_options:               
                # Update availability of sceener's previous apartment
                landlords.available[np.where(landlords.apartment == 
                                    self.apartment[idx_screening[r]])] = True
                # Update screener's status who decided to leave
                self.searching[idx_screening[r]] = True
                self.price[idx_screening[r]] = 0
                self.quality[idx_screening[r]] = 0
                self.apartment[idx_screening[r]] = -1  
        return(landlords)
    
    def application(self, landlords, max_rent_share, inc_factor_state, 
                    state_price, max_sample_applicants, max_applications):
        """
        Method for the application process for renters who are actively 
        searching for an apartment. Arrays are created containing all
        applications by renters for all selected apartments. Specifically, 
        renters do append their own uid to apartments for which they do want to 
        apply. Information will be sent to landlords who will then select new 
        tenants among the applicants.
        Renters can only select among visible apartments - meaning that
        unaffordable apartments and state apartments where they do not fulfill 
        the income criterion are not presented to renters. Furthermore, some
        of the remaining apartments might as well not be visible due to
        imperfect information (random subset is presented). 
        """
        # retrieve required information of available apartments
        apartments = landlords.apartment[np.where(landlords.available == True)]
        prices = landlords.price[np.where(landlords.available == True)]
        quality = landlords.quality[np.where(landlords.available == True)]
        private = landlords.private[np.where(landlords.available == True)]
        apartment_info = np.transpose(np.vstack(
            (apartments,prices,quality,private)))
        # create new array to later collect UIDs of applicants
        applicants_uid = np.zeros_like(apartment_info[:,0])
        # get index from all households currently searching an apartment
        idx_searchers = np.where(self.searching == True)[0]
        # loop through all searchers to select and apply for apartments
        for i in idx_searchers:
            # filter out apartments that are not affordable
            visible_a = apartment_info[np.where(
                apartment_info[:,1]<max_rent_share*self.income[i])]
            # filter out state apartments from visible apartment list 
            # for renters who do not fulfill the income criterion
            if self.income[i] > state_price * inc_factor_state:
                visible_a = visible_a[visible_a[:,3]==True]
            # only continue if at least 1 apartment on the market is affordable
            if visible_a.size > 0:
                # get random sample of available apartment (imperfect inf.)
                # sample size is equal to max_sample_applicants or to all
                # visible apartments in case it is below max_sample_applicants
                sample_size = min(max_sample_applicants,len(visible_a[:,0]))
                idx_sampled_a = rd.choice(np.arange(len(visible_a)),
                                          sample_size,replace=False)
                visible_a = visible_a[idx_sampled_a,:]
                
                # calculate utility for visible apartments    
                utility = visible_a[:,2]**self.preferences[i] * (
                    (self.income[i]-visible_a[:,1]).clip(min=0)**(
                        (1-self.preferences[i])))
                visible_a = np.column_stack((visible_a,utility))
                # select apartments with greatest utility (no more than 
                # max_application can be selected)
                selection_size = min(max_applications,len(visible_a))
                selected_apartments = visible_a[np.argpartition(
                    visible_a[:,4],-selection_size)[-selection_size:],:]
                #create list of same shape to append uid of current searcher
                uid  = np.zeros_like(apartment_info[:,0])
                index_applicants = np.where(np.isin(
                    apartment_info[:,0],selected_apartments[:,0]))
                uid[index_applicants] = self.uid[i]
                # combine applications of current searcher with already 
                # collected applications from other searchers
                applicants_uid = np.column_stack((applicants_uid,uid))

        # create properly formatted list containing all applicants
        applicants_uid_combined = []
        for i in range(len(applicants_uid)):
            # emtpy lists for apartments with no applications received
            if np.sum(applicants_uid[i]) == 0:
                applicants_uid_combined.append([])
            # list with all applicants for other apartments
            else:
                applicants_uid_combined.append(applicants_uid[i,np.nonzero(
                    applicants_uid[i])][0].astype(int))
        # ensure that apartment numbers are formatted as integers
        apartment_info[:,0] = apartment_info[:,0].astype(int)
        #exclude apartments with no applications
        idx_no_applicants = [idx for idx, element in enumerate(
            applicants_uid_combined) if len(element) == 0]
        if len(idx_no_applicants) > 0:
            # sorting required, such that highest indexes are deleted first,
            # and other indexes are not changed after the deletion
            for index in sorted(idx_no_applicants, reverse=True):
                del applicants_uid_combined[index]
            apartment_info = np.delete(apartment_info, idx_no_applicants, 0)
        return(apartment_info,applicants_uid_combined)     