from core import *


plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12, 
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 8, 
    'mathtext.fontset': 'cm',
    'lines.linewidth': 0.8,
    'lines.markersize': 5,
    'figure.dpi': 3000,
})

plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'both'
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# Enable minor ticks
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

Bernoulli_Fs = [star, hedged, cp, randcp]
Beta_Fs = [star, hedged, ttest]


def sweep_constant_factors():
    def highlight():
        fig = plt.gcf() 
        for ax in fig.axes:
            for line in ax.lines:
                if line.get_color() == '#d62728':  
                    line.set_linewidth(1.5)
                    line.set_zorder(100)

    exps = np.arange(-3,7)
    Fs = []
    for exponent in exps:
        Fs.append(add_name(rf"c = $5^{{{exponent }}}$", {'c':5.0**exponent})(star))

    make_experiment_bernoulli(Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[10, 50, 250, 1000])
    highlight()
    plt.savefig('bernoulli_sweep_c.pdf')

    make_experiment_beta(Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 10)
    highlight()
    plt.savefig('beta_sweep_c1.pdf')
    make_experiment_beta(Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 50)
    highlight()
    plt.savefig('beta_sweep_c2.pdf')
    make_experiment_beta(Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 500)
    highlight()
    plt.savefig('beta_sweep_c3.pdf')

def standard_plots():
    make_experiment_bernoulli(Bernoulli_Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[10, 20, 50, 100])
    plt.savefig('bernoulli_standard_1.pdf')
    make_experiment_bernoulli(Bernoulli_Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[200, 400, 800, 1000])
    plt.savefig('bernoulli_standard_2.pdf')

    make_experiment_beta(Beta_Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 10)
    plt.savefig('beta_standard_1.pdf')
    make_experiment_beta(Beta_Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 50)
    plt.savefig('beta_standard_2.pdf')
    make_experiment_beta(Beta_Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 500)
    plt.savefig('beta_standard_3.pdf')

def permutation_plot():
    make_experiment_bernoulli(Bernoulli_Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[10, 20, 50, 100], wrap=gen_shuffle)
    plt.savefig('bernoulli_shuffle.pdf')

    make_experiment_beta(Beta_Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 50, wrap = gen_shuffle)
    plt.savefig('beta_shuffle.pdf')
    

def compare_bets_lastround_plot():
    Fs = [star, bets, hedged, bets2, star2]
    make_experiment_bernoulli(Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[10, 20, 50, 100])
    plt.savefig('bernoulli_last_round.pdf')
    make_experiment_beta(Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 100)
    plt.savefig('beta_last_round1.pdf')
    make_experiment_beta(Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 1000)
    plt.savefig('beta_last_round2.pdf')


def sweep_deltas():
    make_experiment_bernoulli(Bernoulli_Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[10, 20, 50, 100], alpha = 0.001)
    plt.savefig('bernoulli_delta1.pdf')
    make_experiment_bernoulli(Bernoulli_Fs, reps=1000, ps=[0.1, 0.5, 0.9], ns=[10, 20, 50, 100], alpha = 0.01)
    plt.savefig('bernoulli_delta2.pdf')
    make_experiment_beta(Beta_Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 100, alpha=0.001)
    plt.savefig('beta_delta1.pdf')
    make_experiment_beta(Beta_Fs, reps=1000, As=[0.1, 0.5, 2, 5], Bs=[0.1, 0.5, 2, 5], n= 100, alpha = 0.01)
    plt.savefig('betad_delta2.pdf')



if(__name__ == '__main__'):
    sweep_constant_factors()
    standard_plots()
    permutation_plot()
    compare_bets_lastround_plot()
    sweep_deltas()
