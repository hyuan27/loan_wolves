import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np

class HullWhiteModel:
    """
    Dummy Model for testing and example
    """
    def __init__(self, sigma, a, term_structure):
        self.sigma = sigma
        self.a = a
        self.term_structure = term_structure
        self.length = 10
        # set maturity dates for term structure, cannot use durations, so used an arbitrary date instead
        spotDates = [ql.Date(15, 1, 2015), ql.Date(15, 1, 2016),ql.Date(15, 1, 2017),
                     ql.Date(15, 1, 2018),ql.Date(15, 1, 2019),ql.Date(15, 1, 2020),
                     ql.Date(15, 1, 2021), ql.Date(15, 1, 2022), ql.Date(15, 1, 2023), 
                     ql.Date(15, 1, 2024), ql.Date(15, 1, 2025)]
        todaysDate = ql.Date(15, 1, 2015)
        # annual compound convention, linear interpolation for term structure
        ql.Settings.instance().evaluationDate = todaysDate
        spotCurve = ql.ZeroCurve(spotDates, term_structure, ql.Thirty360(), ql.UnitedStates(), ql.Linear(),
                                 ql.Compounded, ql.Annual)
        spotCurveHandle = ql.YieldTermStructureHandle(spotCurve)
        self.hw_process = ql.HullWhiteProcess(spotCurveHandle, a, sigma)
        
        
    def sim_paths(self, num_paths, timestep):
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
        seq = ql.GaussianPathGenerator(self.hw_process, self.length, timestep, rng, False)
        paths = np.zeros((num_paths, timestep+1))
        for i in range(num_paths):
            sample_path = seq.next()
            path = sample_path.value()
            time = [path.time(j) for j in range(len(path))]
            value = [path[j] for j in range(len(path))]
            paths[i, :] = np.array(value)
        return time, paths




# Classes below are just a wrapper to whatever functions you are writing.
# We will call these in the GNMA class to price and value mortgages.

#No need to make any changes here

class CMTRateModel():

    def __init__(self):
        self.term_structure = [0, 0.0312, 0.0320, 0.0325, 0.0328, 0.0333, 0.0337, 0.0340, 0.0343, 0.0345, 0.0347]
        self.a = 0.1
        self.sigma = 0.15
        self.model = HullWhiteModel(self.sigma, self.a, self.term_structure)
    def get_sim_paths(self,num_paths,timestep):
        #Just calls the corresponding sim_model
        return self.model.sim_paths(num_paths, timestep)  
    
class TED():
    def __init__(self):
        super().__init__()
        self.term_structure = [0, 0.0045, 0.0094, 0.0084, 0.0095, 0.0092, 0.0089, 0.0087, 0.0085, 0.0084, 0.0083]
        self.a = 1.8
        self.sigma = 0.05
        self.model = HullWhiteModel(self.sigma, self.a, self.term_structure)
    def get_sim_paths(self,num_paths,timestep):
        #Just calls the corresponding sim_model
        return self.model.sim_paths(num_paths, timestep)  
#LIBOR rates
'''
class LIBORModel():
    """
    We might need to do some processing here based on what is actually the output of simulation.
    """
    def __init__(self):
        pass

    def get_sim_paths(self,num_paths,timestep):
        #Just calls the corresponding sim_model
        return sim_model.get_sim_paths(num_paths,*args)           


class FXModel(InterestRateModel):

    def __init__(self):
        super().__init__()

    def get_sim_paths(self,num_paths,sim_model,*args):
        #Just calls the corresponding sim_model
        return sim_model.get_sim_paths(num_paths,*args)             

'''

if __name__ == '__main__':
    #Example of how the classes will be called

    num_paths = 10
    timestep = 10
    CMT_model = CMTRateModel()

    time,paths = CMT_model.get_sim_paths(num_paths,timestep)
    for i in range(num_paths):
        plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
    plt.title("Hull-White Short Rate Simulation")
    plt.show()