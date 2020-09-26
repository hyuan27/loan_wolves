import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

class HullWhiteModel:
    def __init__(self):
        self.corr = 0.1
        self.a_cmt = 0.1
        self.a_ted = 1.8
        self.time = np.arange(1,11)
        self.term_cmt = interpolate.interp1d(np.arange(1,11), self.spot_curve([0.0312, 0.0320, 0.0325, 0.0328, 0.0333, 0.0337, 0.0340, 0.0343, 0.0345, 0.0347]), fill_value = 'extrapolate')
        self.term_ted = interpolate.interp1d(np.arange(1,11), self.spot_curve([0.0045, 0.0094, 0.0084, 0.0095, 0.0092, 0.0089, 0.0087, 0.0085, 0.0084, 0.0083]), fill_value = 'extrapolate')
        self.sigma_cmt = 0.15*0.0312
        self.sigma_ted = 0.05*0.0045
    def simulate(self, num_paths):
        z1 = np.random.normal(0, 1, (num_paths, 9))
        z = np.random.normal(0, 1, (num_paths, 9))
        z2 = self.corr*z1+np.sqrt(1-self.corr**2)*z
        theta_cmt = self.spot_curve([0.0312, 0.0320, 0.0325, 0.0328, 0.0333, 0.0337, 0.0340, 0.0343, 0.0345, 0.0347])
        theta_ted = self.spot_curve([0.0045, 0.0094, 0.0084, 0.0095, 0.0092, 0.0089, 0.0087, 0.0085, 0.0084, 0.0083])
        
        path_cmt = np.ones((num_paths, 10))*0.0312
        path_ted = np.ones((num_paths, 10))*0.0045
        for i in range(1, 10):
            path_cmt[:, i] = path_cmt[:, i-1]+self.a_cmt*(theta_cmt[i-1]-path_cmt[:, i-1]) + self.sigma_cmt*z1[:, i-1]
            path_ted[:, i] = path_ted[:, i-1]+self.a_ted*(theta_ted[i-1]-path_ted[:, i-1]) + self.sigma_ted*z2[:, i-1]
        return path_cmt, path_ted
    
    def spot_curve(self, y):
        t = np.arange(1,11.)
        s = [] # output array for spot rates
        for i in range(0, len(t)): #calculate i-th spot rate
            sum = 0
            for j in range(0, i): #by iterating through 0..i
                sum += y[i] / (1 + s[j])**t[j]
            value = ((1+y[i]) / (1-sum))**(1/t[i]) - 1
            s.append(value)
        return np.array(s)
    
    def simulate_math(self, num_paths):
        z1 = np.random.normal(0, 1, (num_paths, 9))
        z = np.random.normal(0, 1, (num_paths, 9))
        z2 = self.corr*z1+np.sqrt(1-self.corr**2)*z
        
        delta = 1
        f_cmt = []
        f_ted = []
        ft_cmt = []
        ft_ted = []
        for i in np.arange(1, 10):
            print(self.term_cmt(i+delta))
            f_cmt.append(self.ifr(i, self.term_cmt, delta))
            f_ted.append(self.ifr(i, self.term_ted, delta))
            ft_cmt.append((self.ifr(i+delta, self.term_cmt, delta)-self.ifr(i, self.term_cmt, delta))/delta)
            ft_ted.append((self.ifr(i+delta, self.term_ted, delta)-self.ifr(i, self.term_ted, delta))/delta)
        
        theta_cmt = np.array(ft_cmt)+self.a_cmt*np.array(f_cmt)+self.sigma_cmt**2/self.a_cmt*(1-np.exp(-self.a_cmt*np.arange(1,10)))
        theta_ted = np.array(ft_ted)+self.a_ted*np.array(f_ted)+self.sigma_ted**2/self.a_ted*(1-np.exp(-self.a_ted*np.arange(1,10)))
        path_cmt = np.ones((num_paths, 10))*0.0312
        path_ted = np.ones((num_paths, 10))*0.0045
        
        for i in range(1, 10):
            path_cmt[:, i] = path_cmt[:, i-1] + theta_cmt[i-1] - self.a_cmt*path_cmt[:, i-1] + self.sigma_cmt*z1[:, i-1]
            path_ted[:, i] = path_ted[:, i-1] + theta_ted[i-1] - self.a_ted*path_ted[:, i-1] + self.sigma_ted*z2[:, i-1]
            
        return path_cmt, path_ted
    
        
    def bond_price(self, y, time):
        return np.exp(-y*time)
    # calculate instantaneous forward rate
    def ifr(self, t, func, delta):
        return -(np.log(self.bond_price(func(t+delta), t+delta))-np.log(self.bond_price(func(t), t)))/delta
    
    
if __name__ == '__main__':
    #Example of how the classes will be called
    hullwhite = HullWhiteModel()
    
    # or use .simulate(1000) for the simple assumption model that theta equals the spot rates
    path_cmt, path_ted = hullwhite.simulate_math(1000)
    
    plt.figure()
    
    for i in range(1000):
        plt.plot([i for i in range(1, 11)], path_cmt[i])
    plt.figure()
    for i in range(1000):
        plt.plot([i for i in range(1, 11)], path_ted[i])
    
    print(np.average(path_cmt, axis = 0))
    print(np.average(path_ted, axis = 0))