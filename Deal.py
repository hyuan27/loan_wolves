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
        disc_paths = np.exp(-np.cumsum(self.path_cmt[:3,:],axis = 0))
        disc_pnl = np.sum(disc_paths*self.get_client_cash_flow(),axis = 0)
        disc_pnl -= self.init_prin
        return disc_pnl
    
    def get_client_cash_flow(self):
        """Total cash flows to client by period"""
        return self.get_client_cash_flow_po() + self.get_client_cash_flow_strategy() + self.get_client_eop_cash_flow()

    def get_client_eop_cash_flow(self):
        #Repo price that the client pays
        eop_cash_flows = np.zeros((3,self.num_paths))
        eop_cash_flows[2,:] = self.repo_price()
        return eop_cash_flows
    
    def get_client_cash_flow_po(self):
        """Cash flow from Principal payments"""
        return self.tot_prin_payment[:3,:]

    def get_client_cash_flow_strategy(self):
        return self.const_notional_spread_over_libor(fixed_spread = 0,notional = np.array([100,82,55]))
    

    def get_client_sop_cash_flow(self):
        sop_cash_flows = np.zeros((3,self.num_paths))
        sop_cash_flows[0,:] = self.init_prin
        return sop_cash_flows

    #################################
    # Methods for firm cash flows!! 
    #################################

    def get_firm_disc_pnl(self):
        """At the end of 3 years based on sum of all cash flows"""
        #return np.sum(self.get_firm_cash_flow(),axis = 0)
        disc_paths = np.exp(-np.cumsum(self.path_cmt[:3,:],axis = 0))
        disc_pnl = np.sum(disc_paths*self.get_firm_cash_flow(),axis = 0)
        hedge_val = np.sum(disc_paths*self.get_firm_cash_flow_hedge(),axis=0)
        #We would pay for the hedge at time 0, so subtract that value
        disc_pnl -= hedge_val
        #We would sell the bond at the expected future price at year 3
        disc_val_of_repo = self.g._get_pv_by_path(3)*disc_paths[2,:]
        disc_pnl += disc_val_of_repo
        return disc_pnl
    

    def get_firm_cash_flow(self):
        """Total cash flows to firm by period"""
        #Interest + Cash from hedge
        return self.get_firm_cash_flow_io() + self.get_firm_cash_flow_strategy() + self.get_firm_cash_flow_hedge()  \
                                        + self.get_firm_eop_cash_flow()  
    

    def get_firm_cash_flow_io(self):
        """Total cash flows from interest payments"""
        return self.tot_int_payment[:3,:] - self.servicing_charge[:3,:]

    
    def get_firm_cash_flow_strategy(self):
        """ To be determined """
        return -1*self.const_notional_spread_over_libor(fixed_spread = 0,notional = np.array([100,82,55]))
    
    def get_firm_cash_flow_hedge(self):
        #This is to be determined
        hedge_cash_flows = np.zeros((3,self.num_paths))
        hedge_cash_flows[1,:] = self.hedge_margrabe_option(K=1,notional=82,periods=1)
        hedge_cash_flows[2,:] = self.hedge_margrabe_option(K=1,notional=55,periods=2) + self.hedge_float_rate_note(notional=10)

        return hedge_cash_flows
       
    def get_firm_eop_cash_flow(self):
        #Firm pays repo to client and gets out of hedged pos
        eop_cash_flows = np.zeros((3,self.num_paths))
        eop_cash_flows[2,:] = -1*self.repo_price()
        return eop_cash_flows

    
    def get_firm_sop_cash_flow(self):
        #Firm gets money from client and puts on hedges 
        sop_cash_flows = np.zeros((3,self.num_paths))
        hedge_cf = self.get_firm_cash_flow_hedge()

        sop_cash_flows[0,:] = self.init_prin() - np.sum(
            hedge_cf[:,:]*np.exp(-np.cumsum(self.path_cmt[:3,:],axis = 0)), axis = 0)

        return sop_cash_flows



    ###################################
    # Methods for overall pricing
    ###################################
    def repo_price(self):
        """ Repo price """
        #repo_price = g._get_pv_by_path(as_of=3)
        repo_price = 32
        return repo_price
        
    
    def deal_price(self):
        pass
        
    

    ####################################
    # Firm strategies for CLIENT
    ####################################
    # 1. Swaption


    # 2.Constant Notional spread over LIBOR
    def const_notional_spread_over_libor(self,fixed_spread,notional = np.array([100,82,55])):
        #Pays spread over libor
        cf = (self.libor_rate[:3,:]+ fixed_spread).T*notional
        return cf.T

    def _get_avg_outstanding_balance(self):
        bal = np.zeros((3,self.num_paths))
        bal[0,:] = self.init_prin
        bal[1:,:] = self.outstanding_bal[:2,:]
        return np.mean(bal)
        
    #3. Varying notional spread over LIBOR
    def varying_notional_spread_over_libor(self,fixed_spread, notional ):
        #Based on outstanding balance on the bond
        return notional.T[:,:3]*(self.libor_rate[:,:3]+ fixed_spread)

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
            'Firm Exp. Interest Payments':          np.mean(self.get_firm_cash_flow_io(),axis = 1),
            'Firm Exp. Swap Payments':              np.mean(self.get_firm_cash_flow_strategy(),axis = 1),
            'Firm Exp. Hedge cash flow':            np.mean(self.get_firm_cash_flow_hedge(),axis = 1),
            'Firm Repo':                            np.mean(self.get_firm_eop_cash_flow(),axis = 1),
            'Firm Total':                           np.mean(self.get_firm_cash_flow(),axis = 1),
            'Client Exp. Principal Payments':       np.mean(self.get_client_cash_flow_po(),axis = 1),
            'Client Exp. Swap/Swaption Payments':   np.mean(self.get_client_cash_flow_strategy(),axis = 1),
            'Client Repo':                          np.mean(self.get_client_eop_cash_flow(),axis = 1),
            'Client Total':                         np.mean(self.get_client_cash_flow(),axis = 1)
        })
        
        print(df)

        if export_to_csv:
            df.to_csv('Cash_flows.csv')
    
    
    def get_firm_and_client_cash_flows_disc(self,export_to_csv = False):
        
        years = ['Year 1', 'Year 2', 'Year 3']

        disc_paths = np.exp(-np.cumsum(self.path_cmt[:3,:],axis = 0))

        get_firm_cash_flow_io = self.get_firm_cash_flow_io()*disc_paths
        get_firm_cash_flow_strategy = self.get_firm_cash_flow_strategy()*disc_paths
        get_firm_cash_flow_hedge = self.get_firm_cash_flow_hedge()*disc_paths
        get_firm_eop_cash_flow = self.get_firm_eop_cash_flow()*disc_paths
        get_firm_cash_flow = self.get_firm_cash_flow()*disc_paths
        get_client_cash_flow_po = self.get_client_cash_flow_po()*disc_paths
        get_client_cash_flow_strategy = self.get_client_cash_flow_strategy()*disc_paths
        get_client_eop_cash_flow = self.get_client_eop_cash_flow()*disc_paths
        get_client_cash_flow = self.get_client_cash_flow()*disc_paths


        df_disc = pd.DataFrame({
            'Year': years,            
            'Firm Exp. Interest Payments':          np.mean(get_firm_cash_flow_io,axis = 1),
            'Firm Exp. Swap Payments':              np.mean(get_firm_cash_flow_strategy,axis = 1),
            'Firm Exp. Hedge cash flow':            np.mean(get_firm_cash_flow_hedge,axis = 1),
            'Firm Repo':                            np.mean(get_firm_eop_cash_flow,axis = 1),
            'Firm Total':                           np.mean(get_firm_cash_flow,axis = 1),
            'Client Exp. Principal Payments':       np.mean(get_client_cash_flow_po,axis = 1),
            'Client Exp. Swap/Swaption Payments':   np.mean(get_client_cash_flow_strategy,axis = 1),
            'Client Repo':                          np.mean(get_client_eop_cash_flow,axis = 1),
            'Client Total':                         np.mean(get_client_cash_flow,axis = 1)
        })
        
        print(df_disc)

        if export_to_csv:
            df_disc.to_csv('Cash_flows_disc_2.csv')
    

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

    disc = np.exp(-np.cumsum(d.path_cmt[:3,:],axis = 0))
    print('Discounted Firm valued Repo price is ',g.get_pv(3))
    print('Discounted Firm valued Repo price is ',np.mean(g._get_pv_by_path(3)*disc[2,:]))

    
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
    print(np.mean(d.hedge_margrabe_option(1,82,1)))
    print(np.mean(d.hedge_float_rate_note(10)))

    #d.get_firm_and_client_cash_flows(False)
    #d.get_firm_and_client_cash_flows_disc(True)
    
    print('Firm discounted PNL')
    print(np.mean(d.get_firm_disc_pnl()))

    print('Client discounted PNL')
    print(np.mean(d.get_client_disc_pnl()))

    #Henry - look at this!
    plt.hist(d.get_firm_disc_pnl(),density=True)
    plt.xlabel('Profit')
    plt.title('PNL for FIRM')
    plt.show()

    plt.hist(d.get_client_disc_pnl(),density=True)
    plt.xlabel('Profit')
    plt.title('PNL for CLIENT')
    plt.show()