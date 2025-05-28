import numpy as np 


def lb_from_lgW(ms, lgW, lga):
    try:
        return min(ms[lgW < lga])-ms[1]+ms[0]
    except:
        return 1
    

def star(xs, alpha = 0.05, ms = None):
    if(not ms):
        ms = np.linspace(0,1,10000)

    lga = np.log(1/alpha)
    lgWs = np.zeros(ms.shape)
    sg2 = 0


    for t, x in enumerate(xs):
        t += 0.0001
        S = np.minimum(sg2/t + (ms + 0.0001)*len(xs)/t/t,  0.001+ ms*(1-ms))

        lmbd = np.sqrt(2*(np.maximum(lga-lgWs, 0))/((len(xs) - t) * S))
        lmbd = np.clip(lmbd, 1/(ms -1-0.001), 1/(ms + 0.001))

        lgWs += np.log(1+lmbd*(x-ms))
        sg2 += (x-ms)**2
    r = np.random.rand()
    return lb_from_lgW(ms, lgWs, lga + np.log(r))