from fbm import FBM
import numpy as np
import matplotlib.pyplot as plt

def g(x,t):
    return 1+x**2 - t**3

def G(x):
    return x**2

ST_STATES = []
ST_TIMES = []

for i in range(100):
    f = FBM(n=200, hurst=0.75, length=1, method='daviesharte')

    # Generate a fBm realization
    fbm_sample = f.fbm()
    
    # Get the times associated with the fBm
    t_values = f.times()
    gs = g(fbm_sample, t_values)
    g_cum = gs.cumsum()
    Gs = G(fbm_sample)
    stopping = np.argmax(Gs>=g_cum)
    ST_STATES.append(fbm_sample[stopping])
    ST_TIMES.append(t_values[stopping])
    plt.scatter(t_values, fbm_sample, c='grey',s=1)
    
plt.scatter(ST_TIMES, ST_STATES, c='r',s=5)
plt.show()