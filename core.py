import numpy as np
from matplotlib import pyplot as plt 
import scipy.stats as stats 
import math
import cmath
from statsmodels.stats.proportion import proportion_confint as pc 
import numpy as np 
from scipy.stats import binom 
try:
    from bound_manual_solver import b_alpha_linear, b_alpha_l2norm
except:
    print('PMBSS import failed')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from joblib import Parallel, delayed


def add_name(name, defaults=None):
    def _add_name(f):
        class tmp:
            def __repr__(self):
                return name 
            def __call__(self, *args, **kwargs):
                if defaults:
                    actual_kwargs = dict(defaults)
                    actual_kwargs.update(kwargs)
                    return f(*args, **actual_kwargs)
                return f(*args, **kwargs)
        return tmp()
    return _add_name


def lb_from_lgW(ms, lgW, lga):
    try:
        return min(ms[lgW < lga])-ms[1]+ms[0]
    except:
        return 1
    

@add_name('PMBSS')
def pmbss(xs, lga):
    alpha = 1/np.exp(lga)
    return 1 - b_alpha_l2norm(1-xs,  alpha)


@add_name('T-test')
def ttest(xs, lga):
    conf = 1-1/np.exp(lga)
    n = len(xs)
    mn = np.mean(xs)
    std = np.std(xs, ddof=1)

    margin = stats.t.ppf(conf, df=n-1) * std/n**0.5

    return mn - margin


@add_name('Hoeffding')
def hoeffding(xs, lga):
    return xs.mean() - np.sqrt(lga/len(xs)/2)

@add_name('CP')
def cp(xs, lga):
    alpha = 1/np.exp(lga)

    return pc(xs.sum(),len(xs),alpha=2*alpha, method='beta')[0]

@add_name('random-CP')
def randcp(xs, lga):
    alpha = 1/np.exp(lga)
    x = xs.sum()
    n = len(xs)
    v = np.random.rand()
    lo = 0
    hi = 1

    for _ in range(25):
        mid = (hi + lo)/2
        a = 1-binom.cdf(x,n,mid) + v*binom.pmf(x,n,mid)
        if(a > alpha):
            hi = mid 
        else:
            lo = mid 
    return lo



@add_name('Emp. Bernstein')
def ebb(xs, lga):
    v = np.var(xs)
    return xs.mean() - np.sqrt(2*v*lga/len(xs)) -3*lga/len(xs)


@add_name('Hedged-CI')
def hedged(xs, lga, ms= None):
    if(not ms):
        ms = np.linspace(0,1,10000)

    t = np.arange(1, len(xs)+1)
    mu_hat = (0.5 + np.cumsum(xs))/(t+1)
    sigma2 = (0.25 + np.cumsum((xs - mu_hat)**2))/(t+1)
    sigma2_ = np.append(0.25, sigma2[:-1])

    lgWs = np.zeros(ms.shape)

    for idx, x in enumerate(xs):
        lmbd = np.sqrt(2*lga/(len(xs) * sigma2_[idx])) * np.ones(ms.shape)
        lmbd = np.clip(lmbd, 1/(ms -1-0.001), 1/(ms +0.001))
        lgWs += np.log(1 + lmbd*(x-ms))

    return lb_from_lgW(ms, lgWs, lga)


@add_name('STaR-Bets')
def star(xs, lga, ms = None, c = 1):
    if(not ms):
        ms = np.linspace(0,1,10000)

    lgWs = np.zeros(ms.shape)
    sg2 = 0


    for t, x in enumerate(xs):
        t += 0.0001
        S = np.minimum(sg2/t + c * (ms + 0.0001)*len(xs)/t/t,  0.001+ ms*(1-ms))

        lmbd = np.sqrt(2*(np.maximum(lga-lgWs, 0))/((len(xs) - t) * S))
        lmbd = np.clip(lmbd, 1/(ms -1-0.001), 1/(ms + 0.001))

        lgWs += np.log(1+lmbd*(x-ms))
        sg2 += (x-ms)**2
    r = np.random.rand()
    return lb_from_lgW(ms, lgWs, lga + np.log(r))


@add_name('STaR-Bets w/o last bet')
def star2(xs, lga, ms = None, c = 1):
    if(not ms):
        ms = np.linspace(0,1,10000)

    lgWs = np.zeros(ms.shape)
    sg2 = 0


    for t, x in enumerate(xs):
        t += 0.0001
        S = np.minimum(sg2/t + c * (ms + 0.0001)*len(xs)/t/t,  0.001+ ms*(1-ms))

        lmbd = np.sqrt(2*(np.maximum(lga-lgWs, 0))/((len(xs) - t) * S))
        lmbd = np.clip(lmbd, 1/(ms -1-0.001), 1/(ms + 0.001))

        lgWs += np.log(1+lmbd*(x-ms))
        sg2 += (x-ms)**2
    r = np.random.rand()
    return lb_from_lgW(ms, lgWs, lga)


@add_name('Bets')
def bets(xs, lga, ms = None, c = 1):
    if(not ms):
        ms = np.linspace(0,1,10000)

    lgWs = np.zeros(ms.shape)
    sg2 = 0


    for t, x in enumerate(xs):
        t += 0.0001
        S = np.minimum(sg2/t + c * (ms + 0.0001)*len(xs)/t/t,  0.001+ ms*(1-ms))

        lmbd = np.sqrt(2*lga/((len(xs)) * S))
        lmbd = np.clip(lmbd, 1/(ms -1-0.001), 1/(ms + 0.001))

        lgWs += np.log(1+lmbd*(x-ms))
        sg2 += (x-ms)**2
    r = np.random.rand()
    return lb_from_lgW(ms, lgWs, lga+np.log(r))

    
@add_name('Bets w/o last bet')
def bets2(xs, lga, ms = None, c = 1):
    if(not ms):
        ms = np.linspace(0,1,10000)

    lgWs = np.zeros(ms.shape)
    sg2 = 0


    for t, x in enumerate(xs):
        t += 0.0001
        S = np.minimum(sg2/t + c * (ms + 0.0001)*len(xs)/t/t,  0.001+ ms*(1-ms))

        lmbd = np.sqrt(2*lga/((len(xs)) * S))
        lmbd = np.clip(lmbd, 1/(ms -1-0.001), 1/(ms + 0.001))

        lgWs += np.log(1+lmbd*(x-ms))
        sg2 += (x-ms)**2
    r = np.random.rand()
    return lb_from_lgW(ms, lgWs, lga)


@add_name('STaR-Hoeffding')
def star_hoeff(xs, lga, ms = None, c = 1):
    if(not ms):
        ms = np.linspace(0,1,10000)
    lgWs = np.zeros(ms.shape)

    for t, x in enumerate(xs):
        lmbd = np.sqrt(8 * np.maximum(0,(lga-lgWs))/(len(xs)-t))
        lgW += lmbd*(x-ms) - lmbd**2/8
    r = np.random.rand()
    return lb_from_lgW(ms, lgWs, lga + np.log(r))

@add_name('Hoeffding')
def star_hoeff(xs, lga, ms = None, c = 1):
    if(not ms):
        ms = np.linspace(0,1,10000)
    lgWs = np.zeros(ms.shape)

    for t, x in enumerate(xs):
        lmbd = np.sqrt(8 * lga)/(len(xs))
        lgW += lmbd*(x-ms) - lmbd**2/8
    r = np.random.rand()
    return lb_from_lgW(ms, lgWs, lga)


def ecdf(data):
    sorted_data = np.sort(data)
    n = len(data)
    y = np.arange(n) / (n-1)  
    return sorted_data, y


def gen_shuffle(gen):
    def fun():
        future_seed = np.random.randint(0, 2**32)
        np.random.seed(5)
        out = gen()
        np.random.seed(future_seed)
        np.random.shuffle(out)
        return out 
    return fun



D = {'Bernoulli': lambda n, mu: np.random.binomial(n=1, p=mu, size=n),
      'Beta':      lambda n, a,b:np.random.beta(a,b,n),
}






def experiment(funs, gen, ax = None, alpha = 0.05, reps=1000):
    lga = np.log(1/alpha)
    Lbs = [[] for _ in funs]

    def inner_fn():
        xs = gen()
        return [fun(xs, lga) for fun in funs]

    results = Parallel(n_jobs=-1)(delayed(inner_fn)() for _  in range(reps))
    Lbs = np.array(results).T

    if(ax):
        for lbs, f in zip(Lbs, funs):
            ax.plot(*ecdf(lbs), label=str(f))

    return Lbs

def make_experiment_beta(funs, n=100, As = [0.1, 0.5, 2], Bs = [0.1, 0.5, 2], alpha = 0.05, reps = 100, wrap=lambda x:x):
    fig, axes = plt.subplots(len(As), len(Bs), figsize=(14,8))# , sharey=True, sharex='col')
    axes = np.array(axes)
    fig.subplots_adjust(top=0.85, left=0.15)

    for j, b in enumerate(Bs):
        x_pos = axes[0, j].get_position().x0 + axes[0, j].get_position().width / 2
        fig.text(x_pos, 0.87, f"b = {b}", ha="center", fontweight='bold')

    for i, a in enumerate(As):
        y_pos = axes[i, 0].get_position().y0 + axes[i, 0].get_position().height / 2
        fig.text(0.025, y_pos, f"a = {a}", ha="center", va="center", rotation=90, fontweight='bold')


    fig.text(0.5, 0.05, 'lower confidence bound', ha='center', va='center', fontweight="bold")
    fig.text(0.045, 0.5, 'quantile', ha='center', va='center', rotation='vertical', fontweight="bold")
    idx = 1
    for i, a in enumerate(As):
        for j, b in enumerate(Bs):
            print(idx)
            idx += 1
            gen = wrap(lambda : D['Beta'](n,a,b))
            experiment(funs, gen, axes[i, j], alpha, reps)  # Call the experiment function
            l, h = axes[i, j].get_xlim()
            l = max(l,0)

            vert = a/(a+b)

            axes[i, j].vlines(vert,0,1, colors='magenta')
            axes[i, j].set_ylim(0,1)

            
            axes[i, j].hlines(1-alpha, 0,1, colors='magenta')
            axes[i, j].set_xlim(l,vert  + 0.15*(vert-l))

            axins = inset_axes(axes[i,j], width=0.4, height=0.4, loc='lower right')

            for line in axes[i,j].get_lines():
                axins.plot(line.get_xdata(), line.get_ydata())
            ax = axes[i,j]

            axins.hlines(binom.ppf(0.95, reps, 1-alpha)/reps, vert - (vert-l)/10, vert + (vert-l)/10, colors='k', linewidth=0.7)
            axins.hlines(1-binom.ppf(0.95, reps, alpha)/reps, vert - (vert-l)/10, vert + (vert-l)/10, colors='k', linewidth=0.7)

            axins.hlines(1-alpha, vert - (vert-l)/10, vert + (vert-l)/10, colors='magenta', linewidth=0.3)
            axins.vlines(vert, 1-2*alpha ,1,colors='magenta', linewidth=0.3)
            axins.set_xlim(vert - (vert-l)/10, vert + (vert-l)/10)  
            axins.set_ylim(1-2*alpha, 1) 
            axins.set_xticks([]) 
            axins.set_yticks([])  
            axins.grid(True)

            x_min, x_max = vert - (vert-l)/10, vert + (vert-l)/10
            y_min, y_max = 1-2*alpha, 1
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                            fill=False, edgecolor='k', linewidth=0.5, linestyle='--')

            ax.add_patch(rect)

            rect_corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            inset_corners = [(0, 0), (1, 0), (1, 1), (0, 1)] 

            for rect_corner, inset_corner in zip(rect_corners, inset_corners):
                con = ConnectionPatch(
                    xyA=rect_corner, coordsA=ax.transData,  
                    xyB=inset_corner, coordsB=axins.transAxes,  
                    color='k', linestyle='--', linewidth=0.2, alpha=0.3
                )
                fig.add_artist(con)



    fig.suptitle(f"Beta,  n = {n},  delta = {alpha}, averaged over {reps} runs",  fontweight="bold")
    handles_by_label = {} 
    
    for ax in fig.axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in handles_by_label:
                handles_by_label[label] = handle

    legend = fig.legend(handles_by_label.values(), handles_by_label.keys(),loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles_by_label))

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])
    ax = plt.gca()
    for line in legend.get_lines():
        line.set_linewidth(3)



def make_experiment_bernoulli(funs, ns = [10, 50,100,200], ps = [0.1, 0.5, 0.9], alpha = 0.05, reps = 100, wrap= lambda x:x):

    fig, axes = plt.subplots(len(ns), len(ps), figsize=(14,8), sharey=True)
    axes = np.array(axes)

    fig.subplots_adjust(top=0.85, left=0.15)

    for j, p in enumerate(ps):
        x_pos = axes[0, j].get_position().x0 + axes[0, j].get_position().width / 2
        fig.text(x_pos, 0.87, f"p = {p}", ha="center")

    for i, n in enumerate(ns):
        y_pos = axes[i, 0].get_position().y0 + axes[i, 0].get_position().height / 2
        fig.text(0.025, y_pos, f"n = {n}", ha="center", va="center", rotation=90)

    
    fig.text(0.5, 0.05, 'lower confidence bound', ha='center', va='center', fontweight="bold")
    fig.text(0.045, 0.5, 'quantile', ha='center', va='center', rotation='vertical', fontweight="bold")


    idx = 1
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            print(idx)
            idx += 1
            gen = wrap(lambda : D['Bernoulli'](n, p))
            experiment(funs, gen, axes[i,j], alpha, reps)

            axes[i,j].vlines(p,0,1, colors='magenta')
            axes[i,j].set_ylim(0,1)
            l, h = axes[i, j].get_xlim()
            l = max(l,0)
            
            axes[i, j].hlines(1-alpha, 0, 1, colors='magenta')
            axes[i, j].set_xlim(l,p + 0.15*(p-l))

            vert= p



            axins = inset_axes(axes[i,j], width=0.4, height=0.4, loc='lower right')

            for line in axes[i,j].get_lines():
                axins.plot(line.get_xdata(), line.get_ydata())
            ax = axes[i,j]

            axins.hlines(binom.ppf(0.95, reps, 1-alpha)/reps, vert - (vert-l)/10, vert + (vert-l)/10, colors='k', linewidth=0.7)
            axins.hlines(1-binom.ppf(0.95, reps, alpha)/reps, vert - (vert-l)/10, vert + (vert-l)/10, colors='k', linewidth=0.7)
            axins.hlines(1-alpha, vert - (vert-l)/10, vert + (vert-l)/10, colors='magenta', linewidth=0.3)
            axins.vlines(vert, 1-2*alpha ,1,colors='magenta', linewidth=0.3)
            axins.set_xlim(vert - (vert-l)/10, vert + (vert-l)/10)  
            axins.set_ylim(1-2*alpha, 1) 
            axins.set_xticks([]) 
            axins.set_yticks([])  
            axins.grid(True)

            x_min, x_max = vert - (vert-l)/10, vert + (vert-l)/10
            y_min, y_max = 1-2*alpha, 1
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                            fill=False, edgecolor='k', linewidth=0.5, linestyle='--')

            ax.add_patch(rect)

            rect_corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            inset_corners = [(0, 0), (1, 0), (1, 1), (0, 1)] 

            for rect_corner, inset_corner in zip(rect_corners, inset_corners):
                con = ConnectionPatch(
                    xyA=rect_corner, coordsA=ax.transData,  
                    xyB=inset_corner, coordsB=axins.transAxes,  
                    color='k', linestyle='--', linewidth=0.2, alpha=0.3
                )
                fig.add_artist(con)


    fig.suptitle(f"Bernoulli, delta = {alpha}, averaged over {reps} runs", fontweight="bold")

    handles_by_label = {} 

    for ax in fig.axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in handles_by_label:
                handles_by_label[label] = handle

    legend = fig.legend(handles_by_label.values(), handles_by_label.keys(),loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles_by_label))
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])
    for line in legend.get_lines():
        line.set_linewidth(3)


if(__name__ == '__main__'):
    Fs = [star, bets, hedged, bets2, star2]
    make_experiment_bernoulli(Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[10, 20, 50, 100])
    plt.savefig('t1.pdf')

    make_experiment_beta(Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 100)
    plt.savefig('t2.pdf')

    make_experiment_beta(Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 1000)
    plt.savefig('t3.pdf')
    
