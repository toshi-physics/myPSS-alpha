import numpy as np

class defect:
    def __init__(self, grid_size, dr):
        self.grid_size=grid_size
        self.positions={}
        self.charge=np.zeros(grid_size)
        self.k_list = [np.fft.fftfreq(grid_size[i], d=dr[i])*2*np.pi for i in range(len(grid_size))]
        self.k_grids = np.meshgrid(*k_list, indexing='ij')
        self.kx = 1j*self.kgrids[0]
        self.ky = 1j*self.kgrids[1]
        
        self.dealias_factor = 2/3
        self.dealias = np.ones_like(self.k_grids[0], dtype=bool)
        for i, ki in enumerate(self.k_grids):
            kmax_dealias_i = np.max(np.abs(self.k_list[i])) * self.dealias_factor
            self.dealias &= (np.abs(ki) < kmax_dealias_i)

    def find_defect(Qxx, Qxy):
        Qxxhat = np.fft.fftn(Qxx)
        Qxyhat = np.fft.fftn(Qxy)
        charge_hat = self.kx*Qxxhat*self.ky*Qxyhat-self.ky*Qxxhat*self.qx*Qxyhat
        self.charge = np.fft.ifftn(charge_hat*self.dealias).real()
        