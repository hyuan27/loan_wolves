#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:55:17 2020

@author: chenming
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GNMA:
    
    def __init__(self, rate, T):
        self.rate = rate # CMT rate, directly simulated from Hull-White model
        ### REQUIRED !!
        #This rate include today's rate and all T period rate, total length T+1
        self.maturity = T
        self.principal = 0
    
    def get_rate(self):
        return self.rate
    
    def set_rate(self, rate):
        self.rate = rate

    def set_principal(self, p):
        self.principal = p
    
    ## Function to get annual scheduled payment amount A
    def Scheduled_payment(self):
        discount_vector = np.zeros(self.maturity)
        for i in range(self.maturity):
            discount_vector[i] = 1/(1+self.rate[i])**(i+1)
        
        return self.principal / sum(discount_vector)
    
    def Adjusted_rate(self):
        periodic_cap = 0.005 # +50 basis point
        periodic_floor = -0.005 # -50 basis point
        lifetime_cap = 0.02 # 2 percent
        teaser_spread = 0.0075 # 75 basis point
        normal_spread = 0.0125 # 125 basis point
        bank_fee = 0.004 # 40 basis point to bank when refinance
        
        # refinance is a list of True or False indicating whether refinance happens
        refinance = np.zeros(self.maturity + 1)
        refinance[0] = False # inital time, no refinance case
        # adjusted rate------- the acutal rate paid by homeowners
        adjusted_rate = np.zeros(self.maturity+1)
        ## initial rate is CMT rate + teaser spread, self.rate[0] is today's CMT rate
        adjusted_rate[0] = teaser_spread + self.rate[0]
        # Calculate the adjusted rate
        for i in range(1, self.maturity+1):
            # if next year CMT rate increase, we need to check cap limit
            if self.rate[i] >= self.rate[i-1]:
                newrate = self.rate[i] + normal_spread # CMT rate + normal spread
                periodcap_limit = adjusted_rate[i-1] + periodic_cap
                lifecap_limit = adjusted_rate[0] * (1+lifetime_cap) # 2 percent above intial rate
                
                ## Check whether refinance will happen
                stay_rate = min([newrate, periodcap_limit, lifecap_limit])
                refinance_cost = self.rate[i] + teaser_spread + bank_fee
                
                refinance[i] = refinance_cost < stay_rate
                adjusted_rate[i] = min(stay_rate, refinance_cost)
            
            # else if next year CMT rate drop, we need to consider floor limit
            else:
                newrate = self.rate[i] + normal_spread # CMT rate + normal spread
                periodfloor_limit = adjusted_rate[i-1] + periodic_floor
                lifefloor_limit = adjusted_rate[0] * (1-lifetime_cap) # 2 percent down intial rate
                
                ## Check whether refinance will happen
                stay_rate = max([newrate, periodfloor_limit, lifefloor_limit])
                refinance_cost = self.rate[i] + teaser_spread + bank_fee
                
                refinance[i] = refinance_cost < stay_rate
                adjusted_rate[i] = min(stay_rate, refinance_cost)
            
        return adjusted_rate, refinance
           
        
        
    
    def Payment_Flow(self, A):
        interest_payment = np.zeros(self.maturity)
        principal_payment = np.zeros(self.maturity)
        prepayment = np.zeros(self.maturity)
        
        # indicate whether people will refinance and therefore CPR value
        refinance = self.Adjusted_rate()[1]
        CPR = 0.01 # or 0.1??
        remaining_principal = self.principal
        for i in range(self.maturity):
            if refinance[i+1] == True:
                CPR = 0.04
            interest_payment[i] = remaining_principal * self.rate[i+1]
            principal_payment[i] = max(A - interest_payment[i], 0)
            prepayment[i] = (remaining_principal - principal_payment[i])*CPR
            remaining_principal -= (principal_payment[i] + prepayment[i])
            
                
        return interest_payment, principal_payment, prepayment
    
    
    
    
    def Swap_rate(self, A):
        interest_payment, principal_payment, prepayment = self.Payment_Flow(A)
        service_fee = 0.002 # Ginnie Mae service fee
        # Interest_inflow is the floating rate payment paid to us
        interest_inflow = interest_payment * (1-service_fee)
        # first three year total payment will be
        total_inflow = sum(interest_inflow[:3])
        # first three year cumulative discount rate in our fixed interest payment
        # swap rate * Principal *(cumulative discount rate) == total_inflow        
        cum_discount = 1/(1+self.rate[1]) + 1/(1+self.rate[2])**2 + \
                        1/(1+self.rate[3])**3
        
        swap_rate = total_inflow / (cum_discount * self.principal)
        
        return swap_rate
        


if __name__ == '__main__':
    
    T = 10
    rate = np.array([0.05, 0.052, 0.08, 0.06, 0.04, 
                     0.045, 0.03, 0.035, 0.05, 0.055, 0.06])
    bond = GNMA(rate, T)
    
    bond.set_principal(1000)
    
    print(bond.Scheduled_payment())
    
    adjusted_rate, refinance = bond.Adjusted_rate()
    print(adjusted_rate)
    print(refinance)
    
    A = bond.Scheduled_payment()
    interest_payment, principal_payment, prepayment = bond.Payment_Flow(A)
    print('Interest Payment is:', interest_payment)
    print('Principal Payment is:', principal_payment)
    print('Prepayment is:', prepayment)
        
    Swap_rate = bond.Swap_rate(A)
    print('Swap rate is:', Swap_rate)