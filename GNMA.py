
#Sources 
#https://money.stackexchange.com/questions/61639/what-is-the-formula-for-the-monthly-payment-on-an-adjustable-rate-mortgage

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import interest_rate as interest_rate

class GNMA:

    def __init__(self,
        new_spread = 0.0075,
        seasoned_spread = 0.0125,
        
        lifetime_cap = 0.0200,
        periodic_cap = 0.0050,
        lifetime_floor = None,
        periodic_floor = 0.0050,
        base_CPR = 0.0010,
        add_CPR_pct_inc = 0.3,
        foreclosing_charge = 0.0040,
        servicing_fee = 0.0020):
        
        
        self.new_spread = new_spread #Initial spread over reference rate (teaser rate)
        self.seasoned_spread = seasoned_spread #Spread over reference rate after teaser rate expires

        self.lifetime_cap = lifetime_cap
        self.periodic_cap = periodic_cap
        self.lifetime_floor = lifetime_floor
        self.periodic_floor = periodic_floor
        
        self.base_CPR = base_CPR #Annual constant prepayment rate
        self.add_CPR_pct_inc = add_CPR_pct_inc #CPR jump in prepayments as homeowners refinance and “hop over” to the new teaser rate.
        
        self.foreclosing_charge = foreclosing_charge 
        self.servicing_fee = servicing_fee #Payments of interest and principal are collected by a mortgage servicer, and charge 20 basis point servicing fee to pass over to bond holder


    def sim_pay_schedule(self,ref_rate,init_prin):
        """
        Simulates monthly cash flows - need to change to annual cash flows

        Args:
            maturity ([type]): [description]
            ref_rate ([type]): shape - Time periods , number of paths
            init_prin ([type]): [description]
            cpr ([type]): [description]
        """

        #Simulation consists of two steps. 
        #1. Generate the schedule assuming that the base rate does not change
        #2. As the rate resets, compute any additional interest rate payments that need to be paid 

        
        arr_shape = (ref_rate.shape[0] + 1, ref_rate.shape[1])
        maturity = ref_rate.shape[0]

        self.ref_rate = ref_rate
        self.outstanding_bal = np.zeros(arr_shape)
        self.sch_int_payment = np.zeros(arr_shape)
        self.add_int_payment = np.zeros(arr_shape)
        self.tot_int_payment = np.zeros(arr_shape)
        self.tot_prin_payment = np.zeros(arr_shape)
        self.sch_prin_payment = np.zeros(arr_shape)
        self.sch_tot_payment = np.zeros(arr_shape)
        self.pre_prin_payment = np.zeros(arr_shape)
        self.total_payment = np.zeros(arr_shape)

        #Get the rates for GNMA bonds based on simulated ref rate
        self.gnma_rate = self._get_gnma_rate_sch(ref_rate)
        self.cpr = self._get_cpr(self.gnma_rate,ref_rate)

        #Some computations to generate the initial payment schedule
        #The initial payment schedule is created based on the teaser rate.
        init_rate = ref_rate[0,:] #Scalar value
        _lambda = 1/(1 + init_rate/12.)
        _X = (_lambda/(1 - _lambda))*(1 - _lambda**(12*maturity))
        self.smm = 1 - (1 - self.cpr)**(1/12.)
        
        assert self.cpr.shape == ref_rate.shape
        assert ref_rate.shape == self.gnma_rate.shape
        assert self.smm.shape == self.cpr.shape
        
        #Create scheduled payment
        self.sch_tot_payment[1,:] = init_prin/_X
        for i in range(2,maturity+1):
            self.sch_tot_payment[i,:] = self.sch_tot_payment[i-1,:]*(1 - self.smm[i-1,:])
        
        #Initial outstanding balance at the start of the period
        self.outstanding_bal[0,:] = init_prin
        
        for i in range(1,maturity + 1):
            #Interest payments
            self.sch_int_payment[i,:] = self.outstanding_bal[i-1,:]*init_rate/12.
            self.add_int_payment[i,:] = self.outstanding_bal[i-1,:]*(self.gnma_rate[i-1,:] - init_rate)/12. 
            self.tot_int_payment[i,:] = self.sch_int_payment[i,:] + self.add_int_payment[i,:]
            #Principal payments
            self.sch_prin_payment[i,:] =  self.sch_tot_payment[i,:] - self.sch_int_payment[i,:]
            self.pre_prin_payment[i,:] = (self.outstanding_bal[i-1,:] - self.sch_prin_payment[i,:])*self.smm[i-1,:]
            self.tot_prin_payment[i,:] = self.sch_prin_payment[i,:] + self.pre_prin_payment[i,:]
            #Total payment
            self.total_payment[i,:] = self.tot_int_payment[i,:] + self.tot_prin_payment[i,:]
            #Update outstanding balance
            self.outstanding_bal[i,:] = self.outstanding_bal[i-1,:] - self.tot_prin_payment[i,:]


    def _get_gnma_rate_sch(self,ref_rate):
        #Assuming the ref rate is simulated on an annual basis. So basically the first 12 months will have the same rate.
        gnma_rate = np.zeros(ref_rate.shape)
        #Teaser rate for one year 
        gnma_rate[:12,:] = ref_rate[:12,:] + self.new_spread
        #Reset after one year, this reset is capped and floored at both season and lifetime caps
        gnma_rate[12:,:] = np.minimum(
            np.maximum(np.minimum(ref_rate[12:,:] + self.seasoned_spread, ref_rate[:-12,:] + self.periodic_cap), ref_rate[:-12,:] - self.periodic_floor),
                                                            np.ones(ref_rate[12:,:].shape)*ref_rate[0,:] + self.lifetime_cap)

        return gnma_rate

    def _get_cpr(self,gnma_rate, ref_rate):
        """
        Computes the annual CPR based on changes in interest env
        """
        
        cpr = np.zeros(gnma_rate.shape)


        new_teaser_rate = ref_rate + self.new_spread
        cpr = np.where(gnma_rate - new_teaser_rate > self.foreclosing_charge, self.base_CPR*(1 + self.add_CPR_pct_inc), self.base_CPR)

        return cpr

    def get_swap_value(self,timeperiod):
        """
        Computes the swap value as the average of all the simulation paths
        Args:
            timeperiod (in years): Time period over which the swap needs to be priced
        """
        

        return np.mean(self._get_swap_rate_by_path(timeperiod))

    def _get_swap_rate_by_path(self,timeperiod):
        """
        Computes the swap value for each simulation path
        Args:
            timeperiod (in years): Time period over which the swap needs to be priced
        """
        #Swap value - basically we exchange the floating arm for a fixed arm - need to compute the pv of fixed arm
        #1. Compute the pv of the floating interest rate cash flows. Note that interest rates are stochastic

        #Remember that tot_int_payments have an additional 0 for the starting value
        pv_by_path = np.zeros(self.tot_int_payment[:12*timeperiod + 1,:].shape)
        dt = 1/12.
        pv_by_path = np.sum(self.tot_int_payment[1:12*timeperiod+1,:]*np.exp(-np.cumsum(self.ref_rate[:12*timeperiod],axis = 0)*dt), axis = 0)

        assert pv_by_path.shape == self.ref_rate.shape[1] #Should be equal to number of paths
        
        swap_value_by_path = pv_by_path/(np.sum(np.exp(-np.cumsum(self.ref_rate[:12*timeperiod],axis = 0)*dt),axis = 0)

        assert swap_value_by_path == self.ref_rate.shape[1] #Should be equal to number of paths

        return swap_value_by_path



    def get_pv(self):
        """
        Computes the pv - simulation average of all the cash flows
        """
        return np.mean(self._get_pv_by_path())

    def _get_pv_by_path(self):
        """
        Computes present value of all the cash flows across each path
        """
        pv_by_path = np.zeros(self.total_payment)
        dt = 1/12.
        pv_by_path = np.sum(self.total_payment[1:,:]*np.exp(-np.cumsum(self.ref_rate,axis = 0)*dt), axis = 0)

        assert pv_by_path.shape == self.ref_rate.shape[1] #Should be equal to number of paths

        return pv_by_path


    def print_sample_sim(self,export_to_csv = False):
        #Create dataframe
        '''
        print(self.sch_int_payment.shape)
        print(self.add_int_payment.shape)
        print(self.tot_int_payment.shape)
        print(self.sch_prin_payment.shape)
        print(self.pre_prin_payment.shape)
        print(self.tot_prin_payment.shape)
        print(self.total_payment.shape)
        print(self.outstanding_bal.shape)
        print(self.sch_tot_payment.shape)
        print(self.ref_rate.shape)
        print(self.gnma_rate.shape)
        print(self.cpr.shape)
        '''

        sim_sample_df = (pd.DataFrame({'Scheduled Interest':self.sch_int_payment[1:,0],
                    'Additional Interest':self.add_int_payment[1:,0],
                    'Total Interest':self.tot_int_payment[1:,0],
                    'Scheduled Principal':self.sch_prin_payment[1:,0],
                    'Prepaid Principal':self.pre_prin_payment[1:,0],     
                    'Total Principal':self.tot_prin_payment[1:,0],
                    'Total Payment':self.total_payment[1:,0],
                    'Outstanding Balance':self.outstanding_bal[1:,0],
                    'Scheduled Payment':self.sch_tot_payment[1:,0],
                    'Ref Rate':self.ref_rate[:,0],
                    'GNMA Bond Rate': self.gnma_rate[:,0],
                    'CPR': self.cpr[:,0]}))

        print(sim_sample_df)

        if export_to_csv:
            sim_sample_df.to_csv('Sample Sim File.csv') 

if __name__ == "__main__":
    
    N = 5
    maturity = 30
    init_prin = 125000
    
    num_paths = 10
    timestep = 5
    #Should call the relevant class 
    CMT_model = interest_rate.CMTRateModel()
    time,ref_rate = CMT_model.get_sim_paths(num_paths,timestep)
    
    ref_rate_mod = np.repeat(ref_rate,12,axis = 1)

    g = GNMA()
    g.sim_pay_schedule(ref_rate_mod.T,init_prin)
    g.print_sample_sim(True)




        






