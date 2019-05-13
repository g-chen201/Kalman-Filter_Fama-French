# -*- coding: utf-8 -*-
# Time: 3/7/2019 9:16 AM
# Author: Guanlin Chen

import pandas as pd
import numpy as np
import math as m
from scipy.optimize import minimize

pd.set_option('expand_frame_repr', False)  # do not wrap
pd.set_option('display.max_rows', 1000)


class Datum:
    x = None
    y = None


class Prediction:
    beta = None
    P = None
    eta = None
    f = None

    def __init__(self, theobs):
        self.obs = theobs

    def do(self):
        mu = self.obs.fltr.pars.mu
        F = self.obs.fltr.pars.F
        Q = self.obs.fltr.pars.Q
        R = self.obs.fltr.pars.R
        beta00 = self.obs.fltr.pars.beta00
        P00 = self.obs.fltr.pars.P00
        t = self.obs.t
        if t == 0:
            btt = beta00
            Ptt = P00
        else:
            btt = self.obs.fltr.obs[t - 1].upd.beta
            Ptt = self.obs.fltr.obs[t - 1].upd.P
        self.beta = mu + F * btt
        self.P = F * Ptt * F.transpose() + Q
        y = self.obs.datum.y
        x = self.obs.datum.x
        eta = y - x * self.beta
        f = x * self.P * x.transpose() + R
        self.eta = eta
        self.f = f
        # self.loglik = 2*m.pi*f + eta.transpose()*np.linalg.inv(f)*eta
        self.loglik = eta.transpose() * np.linalg.inv(f) * eta


class Updating:
    beta = None
    P = None
    K = None

    def __init__(self, theobs): self.obs = theobs

    def do(self):
        betapred = self.obs.pred.beta
        Ppred = self.obs.pred.P
        eta = self.obs.pred.eta
        f = self.obs.pred.f
        x = self.obs.datum.x
        self.K = Ppred * x.transpose() * np.linalg.inv(f)
        self.beta = betapred + self.K * eta
        self.P = Ppred - self.K * x * Ppred


class Obs:
    t = None
    datum = Datum()

    def __init__(self, fltr):
        self.fltr = fltr
        self.pred = Prediction(self)
        self.upd = Updating(self)


class Parameters:
    def __init__(self, nx):
        self.mu = np.matrix(np.random.rand(nx, 1))
        self.F = np.matrix(np.random.rand(nx, nx))
        self.R = np.matrix(np.random.rand(1, 1))
        self.q = np.matrix(np.tril(np.random.rand(nx, nx)))
        self.Q = np.matrix(self.q.transpose() * self.q)
        self.beta00 = np.matrix(np.random.rand(nx, 1))
        self.p = np.matrix(np.tril(np.random.rand(nx, nx)))
        self.P00 = np.matrix(self.p.transpose() * self.p)
        self.names = sorted(self.__dict__.keys())
        self.derived = ['Q', 'P00']
        self.estimated = list(set(self.names) - set(self.derived))

    def pack(self):
        package = np.matrix([]).T
        for aname in self.estimated:
            mx = getattr(self, aname)
            mxr = np.reshape(mx, (-1, 1), order='F')
            if aname == 'R': mxr = np.matrix(m.sqrt(mxr))
            if aname == 'q' or aname == 'p':
                mxr = np.reshape(mxr[mxr != 0], (-1, 1), order='F')
            package = np.concatenate((package, mxr), axis=0)
        return (np.array(package))

    def unpack(self, package):
        begin = 0  # Control position in the package.
        for aname in self.estimated:
            mx = getattr(self, aname)
            nr, nc = np.shape(mx)
            if aname != 'q' and aname != 'p':
                finish = begin + nr * nc
                newmx = np.matrix(package[begin:finish])
                newmx = np.reshape(newmx, (nr, nc), order='F')
                begin = finish
            else:
                newmx = np.matrix(np.zeros((nr, nc)))
                for i in range(nc):
                    for j in range(i, nr):
                        newmx[j, i] = package[begin]
                        begin = begin + 1
            setattr(self, aname, newmx)
        self.Q = self.q.T * self.q
        self.P00 = self.p.T * self.p
        self.R = self.R ** 2


#        self.F = np.eye(np.shape(self.F)[0])-self.F

class Filter:
    obs = []

    def __init__(self, thefile, intercept=None):
        thedata = pd.read_csv(thefile)
        nobs, ncol = thedata.shape
        self.nobs = nobs
        self.T = nobs - 1
        if intercept is None:
            response = input('Would you like to have an intercept? (Y/n):')
            if response[0] == 'Y':
                intercept = True
            else:
                intercept = False
        self.nx = ncol
        if not intercept: self.nx = ncol - 1
        self.pars = Parameters(self.nx)
        for t in range(nobs):
            newobs = Obs(self)
            newobs.t = t
            newobs.datum.y = np.matrix(thedata.iloc[t, 0])
            newobs.datum.x = np.matrix(thedata.iloc[t, 1:])
            if intercept is True:
                newobs.datum.x = np.concatenate((np.matrix(1), newobs.datum.x), axis=1)
            self.obs.append(newobs)

    def run(self):
        ll = 0
        for anobs in self.obs:
            anobs.pred.do()
            anobs.upd.do()
            ll = ll + anobs.pred.loglik
            self.loglikelihood = -0.5 * float(ll)

    def objective(self, pckg):
        self.pars.unpack(pckg)
        self.run()
        print(self.loglikelihood)
        return -self.loglikelihood

    #    def Rconstr(self,pckg):
    #        self.pars.unpack(pckg)
    #        R = float(self.pars.R)
    #        return 1000000000*(R-0.0000001)
    def Fconstr1(self, pckg):
        self.pars.unpack(pckg)
        F = self.pars.F
        test1 = np.eye(np.shape(F)[0]) - F
        ev = np.linalg.eig(test1)[0]
        if len(ev[ev == 0]) > 0: return -1

    def Fconstr2(self, pckg):
        self.pars.unpack(pckg)
        F = self.pars.F
        #        vQ = np.reshape(self.pars.Q,(-1,1),order='F')
        FkF = np.kron(F, F)
        test2 = np.eye(np.shape(FkF)[0]) - FkF
        #        test2 = np.linalg.inv(np.eye(np.shape(FkF)[0]) - FkF)*vQ
        #        test2 = np.reshape(test2,np.shape(F),order='F')
        ev = np.linalg.eig(test2)[0]
        if len(ev[ev <= 0]) > 0: return -1

        vQ = np.reshape(self.pars.Q, (-1, 1), order='F')
        FkF = np.kron(F, F)
        test2 = np.linalg.inv(np.eye(np.shape(FkF)[0]) - FkF) * vQ
        test2 = np.reshape(test2, np.shape(F), order='F')
        ev = np.linalg.eig(test2)[0]
        if len(ev[ev <= 0]) > 0: return -1

    def estimate(self):
        p0 = self.pars.pack()
        #        constrR = {"type":"ineq", "fun": self.Rconstr}
        constrF1 = {"type": "ineq", "fun": self.Fconstr1}
        constrF2 = {"type": "ineq", "fun": self.Fconstr2}
        #        cons = [constrF1, constrF2]
        cons = [constrF1]
        minimize(fun=self.objective,
                 x0=p0,
                 #                method='SLSQP',
                 method='COBYLA',
                 constraints=cons,
                 tol=0.01,
                 options={'maxiter': 10000}
                 )



ibm_model = Filter('IBM.csv', True)
ibm_model.estimate()

# test accuracy rate of prediction
df = pd.read_csv('IBM.csv')
beta = []
for obs in ibm_model.obs:
    beta.append(obs.upd.beta.T.tolist()[0])
df_beta = pd.DataFrame(beta, columns=['intercet', 'beta1', 'beta2', 'beta3'])
df = pd.concat([df, df_beta], axis=1)
df['pred'] = df['intercet'] + df['Mkt-rf'] * df['beta1'] + df['HML'] * df['beta2'] + df['SMB'] * df['beta3']

df['pred_original'] = df['Mkt-rf'] * 0.8325 + df['HML'] * (-0.114) + df['SMB'] * (-0.293) # original regression coefficient
df = df[['Mkt-rf', 'pred', 'pred_original']][-100:]
e1 = (df['pred'] - df['Mkt-rf']).pow(2).sum()
e2 = (df['pred_original'] - df['Mkt-rf']).pow(2).sum()
