
#Sources 
#https://money.stackexchange.com/questions/61639/what-is-the-formula-for-the-monthly-payment-on-an-adjustable-rate-mortgage

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import hw as hw

class GNMA:

    def __init__(self,
        new_spread = 0.0075,
        seasoned_spread = 0.0125,
        
        lifetime_cap = 0.0200,
        periodic_cap = 0.0050,
        lifetime_floor = 0.0200,
        periodic_floor = 0.0050,
        base_CPR = 0.10,
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


    def _annualize(self,balances):
        ann_ = np.zeros((self.maturity_in_yrs,balances.shape[1]))
        for i in range(self.maturity_in_yrs):
            ann_[i,:] = np.sum(balances[1+12*i:12*(i+1),:],axis = 0)
        
        return ann_

    def get_ann_total_payment (self):
        return self._annualize(self.total_payment)
    
    def get_ann_outstanding_bal(self):
        ann_ = np.zeros((self.maturity_in_yrs,self.outstanding_bal.shape[1]))
        for i in range(self.maturity_in_yrs):
            ann_[i,:] = self.outstanding_bal[12*(i+1),:]
        return ann_

    def get_ann_tot_int_payment(self):
        return self._annualize(self.tot_int_payment)

    def get_ann_tot_prin_payment(self):
        return self._annualize(self.tot_prin_payment)
    
    def get_ann_pre_prin_payment(self):
        return self._annualize(self.pre_prin_payment)
    
    def get_ann_sch_prin_payment(self):
        return self._annualize(self.sch_prin_payment)

    def get_ann_sch_int_payment(self):
        return self._annualize(self.sch_int_payment)
    
    def get_ann_add_int_payment(self):
        return self._annualize(self.add_int_payment)

    def get_ann_outstanding_bal_start(self):
        """Gives the outstanding balance at the start of the period"""
        ann_ = np.zeros((self.maturity_in_yrs,self.outstanding_bal.shape[1]))
        for i in range(self.maturity_in_yrs):
            ann_[i,:] = self.outstanding_bal[12*(i),:]
        return ann_

    def get_sim_results(self):
        
        attr_map = { 
        'outstanding_bal'   : self.outstanding_bal[1:,:]   ,
        'sch_int_payment'   : self.sch_int_payment[1:,:] ,
        'add_int_payment'   : self.add_int_payment[1:,:] ,
        'tot_int_payment'   : self.tot_int_payment[1:,:] ,
        'tot_prin_payment'  : self.tot_prin_payment[1:,:],
        'sch_prin_payment'  : self.sch_prin_payment[1:,:],
        'sch_tot_payment'   : self.sch_tot_payment[1:,:] ,
        'pre_prin_payment'  : self.pre_prin_payment[1:,:],
        'total_payment'     : self.total_payment[1:,:]   ,
        'gnma_rate'         : self.gnma_rate       ,
        'ref_rate'          : self.ref_rate        ,
        'cpr'               : self.cpr },
        
        return attr_map

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
        self.maturity = ref_rate.shape[0]
        self.maturity_in_yrs = int(self.maturity/12)

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
        init_rate = self.gnma_rate[0,:] #Scalar value
        _lambda = 1/(1 + init_rate/12.)
        _X = (_lambda/(1 - _lambda))*(1 - _lambda**(12*self.maturity_in_yrs))
        self.smm = 1 - (1 - self.cpr)**(1/12.)
        
        assert self.cpr.shape == ref_rate.shape
        assert ref_rate.shape == self.gnma_rate.shape
        assert self.smm.shape == self.cpr.shape
        
        #Create scheduled payment
        self.sch_tot_payment[1,:] = init_prin/_X
        for i in range(2,self.maturity+1):
            self.sch_tot_payment[i,:] = self.sch_tot_payment[i-1,:]*(1 - self.smm[i-1,:])
        
        #Initial outstanding balance at the start of the period
        self.outstanding_bal[0,:] = init_prin
        
        for i in range(1,self.maturity + 1):
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
        for i in range(1,self.maturity_in_yrs):
            gnma_rate[12*i:12*(i+1),:] = np.maximum(
            np.minimum(
            np.maximum(
                np.minimum(ref_rate[12*i:12*(i+1),:] + self.seasoned_spread, gnma_rate[12*(i-1):12*i,:] + self.periodic_cap), 
                        gnma_rate[12*(i-1):12*i,:] - self.periodic_floor),
                                                            np.ones(gnma_rate[12*i:12*(i+1),:].shape)*gnma_rate[0,:] + self.lifetime_cap),
                                                            np.ones(gnma_rate[12*i:12*(i+1),:].shape)*gnma_rate[0,:] - self.lifetime_floor)
        
        #gnma_rate[12:,:] = np.minimum(
        #    np.maximum(np.minimum(ref_rate[12:,:] + self.seasoned_spread, ref_rate[:-12,:] + self.periodic_cap), ref_rate[:-12,:] - self.periodic_floor),
        #                                                    np.ones(ref_rate[12:,:].shape)*ref_rate[0,:] + self.lifetime_cap)

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
        
        swap_value_by_path = pv_by_path/(np.sum(np.exp(-np.cumsum(self.ref_rate[:12*timeperiod],axis = 0)*dt),axis = 0))

        assert swap_value_by_path == self.ref_rate.shape[1] #Should be equal to number of paths

        return swap_value_by_path

    def get_pv(self, as_of = 0):
        """
        Computes the pv - simulation average of all the cash flows
        as_of (in years)= Computes pv as of that period. For eg. if 2, then pv of rem total payments as of the end of year 2
        """
        return np.mean(self._get_pv_by_path(as_of))

    def _get_pv_by_path(self,as_of = 0):
        """
        Computes present value of all the cash flows across each path
        """
        
        dt = 1/12.
        pv_by_path = np.sum(self.total_payment[1 + as_of*12:,:]*np.exp(-np.cumsum(self.ref_rate[as_of*12:,:],axis = 0)*dt), axis = 0)

        print(pv_by_path.shape,self.ref_rate.shape[1])
        assert pv_by_path.shape[0] == self.ref_rate.shape[1] #Should be equal to number of paths

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
            sim_sample_df.to_csv('Sample Sim File 2.csv') 

    ### Methods for creating pretty plots for everything

    def plot_ref_rate(self):
        plt.figure()
        for i in range(self.ref_rate.shape[1]):
            plt.plot(self.ref_rate[:,i])    
        plt.show()

    def plot_gnma_rate(self):
        plt.figure()
        for i in range(self.gnma_rate.shape[1]):
            plt.plot(self.gnma_rate[:,i])    
        plt.show()

    def plot_payment(self):
        
        x = np.arange(self.maturity)
        total_payment = np.mean(self.total_payment[1:,:],axis = 1)
        tot_prin_payment = np.mean(self.tot_prin_payment[1:,:],axis = 1)
        tot_int_payment = np.mean(self.tot_int_payment[1:,:],axis = 1)
        
        plt.figure()
        plt.stackplot(x,[tot_prin_payment,tot_int_payment])
        plt.show()
    
    def plot_balance(self):
        
        x = np.arange(self.maturity)
        
        sch_prin_payment = np.mean(self.sch_prin_payment[1:,:],axis = 1)
        pre_prin_payment = np.mean(self.pre_prin_payment[1:,:],axis = 1)
        outstanding_bal = np.mean(self.outstanding_bal[1:,:],axis = 1) - sch_prin_payment - pre_prin_payment
        plt.figure()
        plt.stackplot(x,[sch_prin_payment,pre_prin_payment,outstanding_bal],
                            labels = ['Scheduled Principal', 'Prepaid Principal', 'Outstanding Principal'])
        
        plt.xlabel('Time (in months)')
        plt.title('Balance ammortization')
        plt.legend()
        plt.show()

    def plot_balance_ann(self):
        
        x = np.arange(self.maturity_in_yrs)

        pre_prin_payment = np.mean(self.get_ann_tot_prin_payment(), axis = 1)
        sch_prin_payment = np.mean(self.get_ann_sch_prin_payment(), axis = 1)
        outstanding_bal_init = np.mean(self.get_ann_outstanding_bal_start(),axis = 1) - pre_prin_payment - sch_prin_payment

        print(outstanding_bal_init,sch_prin_payment,pre_prin_payment)
        plt.figure()
        plt.stackplot(x,[sch_prin_payment,pre_prin_payment,outstanding_bal_init],
                            labels = ['Scheduled Principal', 'Prepaid Principal', 'Outstanding Principal'])
        
        plt.xlabel('Time in Years')
        plt.title('Balance ammortization')
        plt.legend()
        plt.show()

    def plot_payments_ann(self):

        x = np.arange(self.maturity_in_yrs)
        #outstanding_bal = np.mean(self.get_ann_outstanding_bal(),axis = 1)
        pre_prin_payment = np.mean(self.get_ann_tot_prin_payment(), axis = 1)
        sch_prin_payment = np.mean(self.get_ann_sch_prin_payment(), axis = 1)
        sch_int_payment = np.mean(self.get_ann_sch_int_payment(),axis = 1)
        add_int_payment = np.mean(self.get_ann_add_int_payment(),axis = 1)
        
        

        plt.figure()
        plt.stackplot(x,[sch_prin_payment,pre_prin_payment,sch_int_payment,add_int_payment],
                            labels = ['Scheduled Principal', 'Prepaid Principal', 'Scheduled Interest Payment','Additional Interest Payment'])
        
        plt.xlabel('Time in Years')
        plt.title('Payment Schedules (Total in any year)')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    
    init_prin = 100
    
    num_paths = 10000
    hullwhite = hw.HullWhiteModel()
    path_cmt, path_ted = hullwhite.simulate_math(num_paths)

    ref_rate_mod = np.repeat(path_cmt,12,axis = 1)

    g = GNMA()
    g.sim_pay_schedule(ref_rate_mod.T,init_prin)

    #g.plot_balance()        
    #g.plot_payment()
    #g.plot_balance_ann()
    #g.plot_payments_ann()
    #g.print_sample_sim(True)
    print(g.get_pv())





