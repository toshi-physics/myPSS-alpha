import numpy as np
import json, argparse, os
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox


def main():

    initParser = argparse.ArgumentParser(description='model_Q_v_rho_create_videos')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir

    if not os.path.exists(savedir+'/videos/'):
        os.makedirs(savedir+'/videos/')
    
    if os.path.isfile(savedir+"/parameters.json"):
        with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
    
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)
    K         = parameters["K"]        # elastic constant, square of nematic correlation length
    
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])
    Lx        = mx*dx
    Ly        = my*dy
    
    #setup a meshgrid
    tol = 0.001
    
    x   = np.linspace(0+tol, Lx-tol, mx)
    y   = np.linspace(0+tol, Ly-tol, my)
    xv, yv  = np.meshgrid(x,y, indexing='ij')
    
    times = np.arange(0, n_dump, 1)*dt_dump

    
    figrho, axrho= plt.subplots(figsize=(12, 8), ncols=1)
    figv, axv= plt.subplots(figsize=(12, 8), ncols=1)
    figQ, axQ= plt.subplots(figsize=(12, 8), ncols=1)
    figvort, axvort= plt.subplots(figsize=(12, 8), ncols=1)

    n=1
    p_factor = 2
    
    rho = np.loadtxt(savedir+'/data/'+'rho.csv.{:d}'.format(n), delimiter=',')
    Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(n), delimiter=',')
    Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(n), delimiter=',')
    curldivQ = np.loadtxt(savedir+'/data/'+'curldivQ.csv.{:d}'.format(n), delimiter=',')
    vx = np.loadtxt(savedir+'/data/'+'vx.csv.{:d}'.format(n), delimiter=',')
    vy = np.loadtxt(savedir+'/data/'+'vy.csv.{:d}'.format(n), delimiter=',')
    #pixelate fields because the point density is too high
    #vx = pixelate(vx, p_factor)
    #vy = pixelate(vy, p_factor)
    v  = np.sqrt(vx**2 + vy**2)
    S = np.sqrt(Qxx**2+Qxy**2)
    pxv = xv[p_factor:-1:p_factor, p_factor:-1:p_factor]
    pyv = yv[p_factor:-1:p_factor, p_factor:-1:p_factor]
    pvx = vx[p_factor:-1:p_factor, p_factor:-1:p_factor]
    pvy = vy[p_factor:-1:p_factor, p_factor:-1:p_factor]
    #Sp = pixelate(S, p_factor)
    theta = np.arctan2(Qxy, Qxx)/2
    nx    = np.cos(theta) [p_factor:-1:p_factor, p_factor:-1:p_factor]
    ny    = np.sin(theta) [p_factor:-1:p_factor, p_factor:-1:p_factor]
    Snx   = S [p_factor:-1:p_factor, p_factor:-1:p_factor] * nx
    Sny   = S [p_factor:-1:p_factor, p_factor:-1:p_factor] * ny
    vscale = 0.05
    nscale = 0.3
    
    crho = [axrho.pcolormesh(xv, yv, rho, cmap='viridis', vmin=0, vmax=2.3), axrho.quiver(pxv, pyv, Snx, Sny, color='b', pivot='middle', headlength=0, headaxislength=0, scale=nscale, scale_units='xy')]
    cv   = [axv.pcolormesh(xv, yv, v, cmap='viridis', vmin=0, vmax=0.6), axv.quiver(pxv, pyv, pvx, pvy, color='w', pivot='middle', scale=vscale, scale_units='xy')]
    cQ   = [axQ.pcolormesh(xv, yv, S, cmap='viridis', vmin=0, vmax=1), axQ.quiver(pxv, pyv,nx, ny, color='k', pivot='middle', headlength=0, headaxislength=0)]
    cvort= [axvort.pcolormesh(xv, yv, curldivQ, vmin=-0.1, vmax=0.1), axvort.quiver(pxv, pyv, pvx, pvy, color='w', pivot='middle', scale=vscale, scale_units='xy')]

    figrho.colorbar(crho[0])
    axrho.set_title(r"$\rho$")
    figv.colorbar(cv[0])
    axv.set_title(r"$v$")
    figQ.colorbar(cQ[0])
    axQ.set_title('S')
    figvort.colorbar(cvort[0])
    axvort.set_title('Vorticity')
    
    tbaxrho = figrho.add_axes([0.2, 0.93, 0.04, 0.04])
    tbrho = TextBox(tbaxrho, 'time')
    tbaxv = figv.add_axes([0.2, 0.93, 0.04, 0.04])
    tbv = TextBox(tbaxv, 'time')
    tbaxQ = figQ.add_axes([0.2, 0.93, 0.04, 0.04])
    tbQ = TextBox(tbaxQ, 'time')    
    tbaxvort = figvort.add_axes([0.2, 0.93, 0.04, 0.04])
    tbvort = TextBox(tbaxvort, 'time')
    
    def plt_snapshot_rho(val):        
        rho = np.loadtxt(savedir+'/data/'+'rho.csv.{:d}'.format(val), delimiter=',')
        Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(val), delimiter=',')
        Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(val), delimiter=',')
        S = np.sqrt(Qxx**2+Qxy**2) 
        theta = np.arctan2(Qxy, Qxx)/2
        Snx    = (S*np.cos(theta)) [p_factor:-1:p_factor, p_factor:-1:p_factor]
        Sny    = (S*np.sin(theta)) [p_factor:-1:p_factor, p_factor:-1:p_factor]
        
        crho[0].set_array(rho)
        crho[1].set_UVC(Snx, Sny)
        tbrho.set_val(round(times[val],2))
        
        figrho.canvas.draw_idle()

    def plt_snapshot_v(val):        
        vx = np.loadtxt(savedir+'/data/'+'vx.csv.{:d}'.format(val), delimiter=',')
        vy = np.loadtxt(savedir+'/data/'+'vy.csv.{:d}'.format(val), delimiter=',')
        pvx = vx[p_factor:-1:p_factor, p_factor:-1:p_factor]
        pvy = vy[p_factor:-1:p_factor, p_factor:-1:p_factor]
        v  = np.sqrt(vx**2 + vy**2)
        #vx = np.where(v==0, vx, vx/v)
        #vy = np.where(v==0, vy, vy/v)
        
        cv[0].set_array(v)
        cv[1].set_UVC(pvx, pvy)
        tbv.set_val(round(times[val],2))
        
        figv.canvas.draw_idle()

    def plt_snapshot_Q(val):        
        Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(val), delimiter=',')
        Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(val), delimiter=',')
        S = np.sqrt(Qxx**2+Qxy**2)
        theta = np.arctan2(Qxy, Qxx)/2
        nx    = np.cos(theta)
        ny    = np.sin(theta)
        
        cQ[0].set_array(S)
        cQ[1].set_UVC(nx[p_factor:-1:p_factor, p_factor:-1:p_factor], ny[p_factor:-1:p_factor, p_factor:-1:p_factor])
        tbQ.set_val(round(times[val],2))

        figQ.canvas.draw_idle()
    
    def plt_snapshot_vort(val):        
        curldivQ = np.loadtxt(savedir+'/data/'+'curldivQ.csv.{:d}'.format(val), delimiter=',')
        vx = np.loadtxt(savedir+'/data/'+'vx.csv.{:d}'.format(val), delimiter=',')
        vy = np.loadtxt(savedir+'/data/'+'vy.csv.{:d}'.format(val), delimiter=',')
        pvx = vx[p_factor:-1:p_factor, p_factor:-1:p_factor]
        pvy = vy[p_factor:-1:p_factor, p_factor:-1:p_factor]
        
        cvort[0].set_array(curldivQ)
        cvort[1].set_UVC(pvx, pvy)
        tbvort.set_val(round(times[val],2))

        figvort.canvas.draw_idle()

    
    from matplotlib.animation import FuncAnimation
    animrho = FuncAnimation(figrho, plt_snapshot_rho, frames = n_dump, interval=100, repeat=True)
    animrho.save(savedir+'/videos/'+'rho.mp4')

    animv = FuncAnimation(figv, plt_snapshot_v, frames = n_dump, interval=100, repeat=True)
    animv.save(savedir+'/videos/'+'v.mp4')

    animQ = FuncAnimation(figQ, plt_snapshot_Q, frames = n_dump, interval=100, repeat=True)
    animQ.save(savedir+'/videos/'+'Q.mp4')

    animvort = FuncAnimation(figvort, plt_snapshot_vort, frames = n_dump, interval=100, repeat=True)
    animvort.save(savedir+'/videos/'+'vorticity.mp4')

def pixelate(x, gridpoints):
    nx, ny = np.shape(x)
    xpad = np.pad(x, (gridpoints, gridpoints), 'wrap')
    ret = np.zeros(np.shape(x))
    for cx in np.arange(nx):
        for cy in np.arange(ny):
            ret[cx, cy] += np.average(xpad[cx:cx+2*gridpoints, cy:cy+2*gridpoints])
    ret = ret[gridpoints:-1:gridpoints, gridpoints:-1:gridpoints]
    return ret

if __name__=="__main__":
    main()
