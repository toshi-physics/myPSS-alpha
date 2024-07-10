import numpy as np
import json, argparse, os

initParser = argparse.ArgumentParser(description='model_Q_v_rho_create_avgs')
initParser.add_argument('-s','--save_dir', help='directory to save data')
initargs = initParser.parse_args()
savedir = initargs.save_dir

if not os.path.exists(savedir+'/processed_data/'):
        os.makedirs(savedir+'/processed_data/')

if os.path.isfile(savedir+"/parameters.json"):
    with open(savedir+"/parameters.json") as jsonFile:
          parameters = json.load(jsonFile)

T         = parameters["T"]        # final time
dt_dump   = parameters["dt_dump"]
n_steps   = int(parameters["n_steps"])  # number of time steps

dt        = T / n_steps     # time step size
n_dump    = round(T / dt_dump)

maxrho = 0
minrho = 0
maxvx  = 0
minvx  = 0
maxvy  = 0
minvy  = 0
maxvel = 0
minvel = 0
maxS   = 0
minS   = 0

i=0
for n in np.arange(n_dump):
    maxrho = np.max(minrho, np.max(np.loadtxt(savedir+'/data/rho.csv.{:d}'.format(n), delimiter=',')))
    maxvx = np.max(maxvx, np.max(np.loadtxt(savedir+'/data/vx.csv.{:d}'.format(n), delimiter=',')))
    maxvy = np.max(maxvy, np.max(np.loadtxt(savedir+'/data/vy.csv.{:d}'.format(n), delimiter=',')))
    minvx = np.min(minvx, np.min(np.loadtxt(savedir+'/data/vx.csv.{:d}'.format(n), delimiter=',')))
    minvy = np.min(minvy, np.min(np.loadtxt(savedir+'/data/vy.csv.{:d}'.format(n), delimiter=',')))
    S   = np.sqrt(Qxx**2 + Qxy**2)
    meanS[i] = np.average(S)
    theta = np.arctan2(Qxy, Qxx)
    meantheta[i] += np.average(theta)
    stdtheta[i] += np.std(theta)
    i+=1

np.savetxt(savedir+'/processed_data/'+'maxmin.txt', [maxrho, minrho, maxvx, minvx, maxvy, minvy, maxvel, minvel, maxS, minS])
