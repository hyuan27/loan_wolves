import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import hw as hw
import GNMA_updated as GNMA


class Deal:

    def __init__(self,bond_instance,path_ted):
        #Run the simulations before passing this in
        self.g = bond_instance
        self.sim_results = bond_instance.get_sim_results()
        self.path_ted = path_ted
        
        self.path_cmt = self.sim_results['ref_rate']
        self.gnma_rate = self.sim_results['gnma_rate']
        self.outstanding_bal = self.sim_results['outstanding_bal']
        self.servicing_charge = self.sim_results['servicing_charge']

        self.sch_int_payment = self.sim_results['sch_int_payment']
        self.add_int_payment = self.sim_results['add_int_payment']
        self.tot_int_payment = self.sim_results['tot_int_payment']
        self.tot_prin_payment = self.sim_results['tot_prin_payment']
        self.sch_prin_payment = self.sim_results['sch_prin_payment']
        self.sch_tot_payment = self.sim_results['sch_tot_payment']
        self.pre_prin_payment = self.sim_results['pre_prin_payment']
        self.total_payment = self.sim_results['total_payment']
        self.init_prin =    self.sim_results['init_prin']
        self.libor_rate = self.path_cmt + self.path_ted
        self.num_paths = self.path_ted.shape[1]
        
        
    
    #################################
    # Methods for client cash flows!! 
    #################################

    def get_client_disc_pnl(self):
        """At the end of 3 years based on sum of all cash flows"""
        pass
        
    def get_client_eop_cash_flow(self):
        #Repo price that the client pays
        eop_cash_flows = np.zeros((3,self.num_paths))
        eop_cash_flows[2,:] = self.repo_price()
        return eop_cash_flows

    def get_client_cash_flow(self):
        """Total cash flows to client by period"""
        return self.get_client_cash_flow_po() + self.get_client_cash_flow_strategy()

    def get_client_cash_flow_po(self):
        """Cash flow from Principal payments"""
        return self.tot_prin_payment[:3,:]

    def ac(self,strat_func):
        strat_cash_flows = np.zeros(self.num_paths,3)
        
        strat1 = self.const_notional_spread_over_libor
        strat1_args = {
            'fixed_spread': 0.02
        }
        
        strat_fun_arg = {'strat_func':(self.const_notional_spread_over_libor, {})

        }
        for func, args in strat_func:
            strat_cash_flows += func(args)
            
        return strat_cash_flows
    

    #################################
    # Methods for firm cash flows!! 
    #################################

    def get_firm_disc_pnl(self):
        """At the end of 3 years based on sum of all cash flows"""
        #return np.sum(self.get_firm_cash_flow(),axis = 0)
        pass

    def get_firm_cash_flow(self):
        """Total cash flows to firm by period"""
        #Interest + Cash from hedge
        return self.get_firm_cash_flow_io() + self.get_firm_cash_flow_strategy() + self.get_firm_cash_flow_hedge()    
    

    def get_firm_cash_flow_io(self):
        """Total cash flows from interest payments"""
        return self.tot_int_payment[:3,:] - self.servicing_charge[:3,:]

    
    def get_firm_cash_flow_strategy(self,**strat_func):
        """ To be determined """
        strat_cash_flows = np.zeros(self.num_paths,3)
        for func, args in strat_func:
            strat_cash_flows += func(args)
            
        return strat_cash_flows
    
    def get_firm_cash_flow_hedge(self,**hedge_func):
        #This is to be determined
        hedge_cash_flows = np.zeros((self.num_paths,3))
        for func, args in hedge_func:
            hedge_cash_flows += func(args)
            
        return hedge_cash_flows
       
    def get_firm_eop_cash_flow(self):
        #Firm pays repo to client and gets out of hedged pos
        eop_cash_flows = np.zeros((3,self.num_paths))
        eop_cash_flows[2,:] = self.repo_price()
        return eop_cash_flows


    ###################################
    # Methods for overall pricing
    ###################################
    def repo_price(self):
        """ Repo price """
        repo_price = g._get_pv_by_path(as_of=3)
        return repo_price
        
    
    def deal_price(self):
        pass
        
    

    ####################################
    # Firm strategies for CLIENT
    ####################################
    # 1. Swaption


    # 2.Constant Notional spread over LIBOR
    def const_notional_spread_over_libor(self,fixed_spread,notional = None):
        #Pays spread over libor
        notional = self._get_avg_outstanding_balance()
        print(notional)
        hedge_cash_flows = np.zeros(self.num_paths,3)
        return notional*(self.libor_rate[:3,:]+ self.fixed_spread)

    def _get_avg_outstanding_balance(self):
        bal = np.zeros((3,self.num_paths))
        bal[0,:] = self.init_prin
        bal[1:,:] = self.outstanding_bal[:2,:]
        return np.mean(bal)
        
    #3. Varying notional spread over LIBOR
    def varying_notional_spread_over_libor(self,fixed_spread, notional ):
        #Based on outstanding balance on the bond
        return notional.T[:,:3]*(self.libor_rate[:,:3]+ self.fixed_spread)

    ####################################
    # Firm strategies for hedging 
    ####################################
    # 1. Delta hedge swaption



    # 2. Interest rate floor on GNMA 
    def hedge_ir_floor(self, gnma_rate, K, notional , periods):
        return notional.T[:,periods]*np.maximum(gnma_rate[:,periods],K)

    # 3. Interest rate cap on GNMA
    def hedge_ir_cap(self,gnma_rate , K, notional , periods):
        return notional.T[:,periods]*np.minimum(gnma_rate[:,periods],K)

    # 4. Margrabe option on LIBOR and CMT
    def hedge_margrabe_option(self, K, notional , periods):
        #Pays (libor - K*gnma_rate)+
        return np.maximum(0,notional*(self.libor_rate[periods,:] - K*self.gnma_rate[periods,:]))

    def hedge_float_rate_note(self,notional):
        return self.gnma_rate[2,:]*notional

    # 5. Put option
    def hedge_put_option(self,K,notional):
        #Notional is the underlying 
        return np.maximum(0,K - notional)

    # 6. Call option
    def hedge_call_option(self,K,notional):
        #Notional is actually the underlying
        return np.maximum(0,notional - K)
    

    #################################
    #Methods to create graphs
    #################################


    def get_client_cash_flows(self,export_to_csv = False):
        
        years = ['Year 1', 'Year 2', 'Year 3']
        df = pd.DataFrame({
            'Year': years,
            'Exp. Principal Payments': self.get_client_cash_flow_po(),
            'Exp. Swap/Swaption Payments': self.get_client_cash_flow_strategy(),
            'Repo': self.get_client_eop_cash_flow(),
            'Total': self.get_firm_cash_flow()
        })
        
        print(df)

        if export_to_csv:
            df.to_csv('Client_cash_flows.csv')


    def get_firm_cash_flows(self,export_to_csv = False):
        
        years = ['Year 1', 'Year 2', 'Year 3']
        df = pd.DataFrame({
            'Year': years,            
            'Exp. Interest Payments': self.get_firm_cash_flow_io(),
            'Exp. Swap/Swaption Payments': self.get_firm_cash_flow_strategy(),
            'Exp. Hedge cash flow': self.get_firm_cash_flow_hedge(),
            'Repo': self.get_firm_eop_cash_flow(),
            'Total': self.get_firm_cash_flow()
        })
        
        print(df)

        if export_to_csv:
            df.to_csv('Firm_cash_flows.csv')
    
    def get_firm_and_client_cash_flows(self,export_to_csv = False):
        
        years = ['Year 1', 'Year 2', 'Year 3']
        df = pd.DataFrame({
            'Year': years,            
            'Firm Exp. Interest Payments': self.get_firm_cash_flow_io(),
            'Firm Exp. Swap/Swaption Payments': self.get_firm_cash_flow_strategy(),
            'Firm Exp. Hedge cash flow': self.get_firm_cash_flow_hedge(),
            'Firm Repo': self.get_firm_eop_cash_flow(),
            'Firm Total': self.get_firm_cash_flow(),
            'Client Exp. Principal Payments': self.get_client_cash_flow_po(),
            'Client Exp. Swap/Swaption Payments': self.get_client_cash_flow_strategy(),
            'Client Repo': self.get_client_eop_cash_flow(),
            'Client Total': self.get_firm_cash_flow()
        })
        
        print(df)

        if export_to_csv:
            df.to_csv('Cash_flows.csv')
    

if __name__ == "__main__":

    init_prin = 100

    num_paths = 10000
    hullwhite = hw.HullWhiteModel()
    path_cmt, path_ted = hullwhite.simulate_math(num_paths)

    g = GNMA.GNMA()
    g.sim_pay_schedule(path_cmt.T,init_prin)
    print('PV at time 0 is ',g.get_pv())
    print('Repo price is ',g.get_pv(3))

    print(path_ted.shape)
    d = Deal(g,path_ted.T)
    
    print(np.mean(d.get_firm_cash_flow_io(),axis = 1))
    print(np.mean(d.get_client_cash_flow_po(),axis = 1))
    print(np.mean(d.get_client_eop_cash_flow(),axis = 1))
    print(np.mean(d.get_firm_eop_cash_flow(),axis = 1))
    print(d._get_avg_outstanding_balance())    
    print(np.mean(d.outstanding_bal[:3,:],axis=1))
    print('Swaptions now')
    print(np.mean(d.libor_rate[:3,:],axis=1))
    print(np.mean(d.gnma_rate[:3,:],axis=1))
    print(np.mean(d.path_cmt[:3,:],axis=1))

    print('Average state contingent cash flows at year 2 ')
    contingent_cash_flow = d.outstanding_bal[1,np.where(d.outstanding_bal[1,:] < 55)]
    print(np.min(d.outstanding_bal[1,:]))
    print(np.max(d.outstanding_bal[1,:]))
    print(contingent_cash_flow.shape)
    print(np.mean(contingent_cash_flow))

    print('Hedged values')
    print('Margrabe values')
    print(np.mean(d.hedge_margrabe_option(1,55,2)))
    print(np.mean(d.hedge_float_rate_note(10)))

    
    