import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np

class HullWhiteModelDummy:
    """
    Dummy Model for testing. Please define the actuall class below.
    """

    #Source:http://gouthamanbalaraman.com/blog/hull-white-simulation-quantlib-python.html
    @staticmethod
    def get_sim_paths(num_paths,timestep = None):
        sigma = 0.1
        a = 0.1
        timestep = 360
        length = 30 # in years
        forward_rate = 0.05
        day_count = ql.Thirty360()
        todays_date = ql.Date(15, 1, 2015)

        ql.Settings.instance().evaluationDate = todays_date

        spot_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
        spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)

        hw_process = ql.HullWhiteProcess(spot_curve_handle, a, sigma)
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
        seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)
        
        #Get simulation paths

        arr = np.zeros((num_paths, timestep+1))
        for i in range(num_paths):
            sample_path = seq.next()
            path = sample_path.value()
            time = [path.time(j) for j in range(len(path))]
            value = [path[j] for j in range(len(path))]
            arr[i, :] = np.array(value)
        
        return (np.array(time), arr)



class HullWhiteModel:
    
    @staticmethod
    def get_sim_paths(num_paths,timestep):
        raise NotImplementedError 

class VasicekModel:
    @staticmethod
    def get_sim_paths(num_paths,timestep):
        raise NotImplementedError
    




# Classes below are just a wrapper to whatever functions you are writing.
# We will call these in the GNMA class to price and value mortgages.

#No need to make any changes here

#Base class
class InterestRateModel:
    """
    Base class for all interest/FX rate models
    """
    def __init__(self):
        pass

    def simulate_paths(self,num_paths,sim_model,*args):
        #Must be implemented by all sub classes
        raise NotImplementedError

#LIBOR rates
class LIBORModel(InterestRateModel):
    """
    We might need to do some processing here based on what is actually the output of simulation.
    """
    def __init__(self):
        super().__init__()

    def get_sim_paths(self,num_paths,sim_model,*args):
        #Just calls the corresponding sim_model
            return sim_model.get_sim_paths(num_paths,*args)            


class CMTRateModel(InterestRateModel):

    def __init__(self):
        super().__init__()

    def get_sim_paths(self,num_paths,sim_model,*args):
        #Just calls the corresponding sim_model
        return sim_model.get_sim_paths(num_paths,*args)            

class FXModel(InterestRateModel):

    def __init__(self):
        super().__init__()

    def get_sim_paths(self,num_paths,sim_model,*args):
        #Just calls the corresponding sim_model
        return sim_model.get_sim_paths(num_paths,*args)             



if __name__ == '__main__':
    #Example of how the classes will be called

    num_paths = 10
    CMT_model = CMTRateModel()

    model_args = dict({'timestep' : 360})
    
    time,paths = CMT_model.get_sim_paths(num_paths,HullWhiteModelDummy,*model_args)
    for i in range(num_paths):
        plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
    plt.title("Hull-White Short Rate Simulation")
    plt.show()