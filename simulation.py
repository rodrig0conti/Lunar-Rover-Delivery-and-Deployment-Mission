import time as _time
import numpy as np
from mathutils import rk4_step
class History:
    def __init__(self,moon):
        self.moon=moon; self.t=[]; self.states=[]; self.energy=[]
    def append(self,t,s,e):
        self.t.append(t); self.states.append(np.array(s,copy=True)); self.energy.append(e)
    def finalise(self):
        self.t=np.asarray(self.t); self.states=np.asarray(self.states); self.energy=np.asarray(self.energy)
        self.r=self.states[:,0:3]; self.v=self.states[:,3:6]; self.mass=self.states[:,6]
        if self.states.shape[1]==14:
            self.q=self.states[:,7:11]; self.omega=self.states[:,11:14]
        else:
            self.q=None; self.omega=None
        self.radius=np.linalg.norm(self.r,axis=1); self.speed=np.linalg.norm(self.v,axis=1)
        self.altitude=self.radius-self.moon.radius
        n=len(self.t); self.lat=np.empty(n); self.lon=np.empty(n)
        for i in range(n):
            self.lat[i],self.lon[i],_=self.moon.geodetic(self.r[i],self.t[i])
        self.energy_error=np.abs((self.energy-self.energy[0])/self.energy[0])
        self.propellant_used=self.mass[0]-self.mass[-1]
class Simulation:
    def __init__(self,moon,dynamics,dt=1.0,log_every=1):
        self.moon=moon; self.dynamics=dynamics; self.dt=dt; self.log_every=log_every
        self.stop_conditions=[]; self.stop_reason=None
        self.add_stop_condition("surface impact",
            lambda t,s: np.linalg.norm(s[0:3])<=self.moon.radius)
        self.guidance=None
    def add_stop_condition(self,name,c): self.stop_conditions.append((name,c))
    def _check_stop(self,t,s):
        for n,c in self.stop_conditions:
            if c(t,s): return n
        return None
    def run(self,state0,t_final,t0=0.0,verbose=True):
        h=History(self.moon); t=t0; state=np.array(state0,float); step=0
        self.stop_reason=None
        h.append(t,state,self.dynamics.specific_energy(state))
        w0=_time.perf_counter()
        while t<t_final-1e-12:
            if self.guidance is not None: self.guidance(t,state)
            dt=min(self.dt,t_final-t)
            state=rk4_step(self.dynamics.derivative,t,state,dt)
            if len(state)==14:
                from attitude import quat_normalise
                state[7:11]=quat_normalise(state[7:11])
            t+=dt; step+=1
            reason=self._check_stop(t,state)
            if reason is not None:
                self.stop_reason=reason
                h.append(t,state,self.dynamics.specific_energy(state)); break
            if step%self.log_every==0:
                h.append(t,state,self.dynamics.specific_energy(state))
        if h.t[-1]<t: h.append(t,state,self.dynamics.specific_energy(state))
        h.finalise()
        if verbose:
            print(f"Propagated {t-t0:.1f} s in {_time.perf_counter()-w0:.2f} s wall ({step} steps).")
            if self.stop_reason: print(f"Stopped early: {self.stop_reason}.")
        return h