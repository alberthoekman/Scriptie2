import arch.univariate.mean

from Simulation import Simulation
import os
import pickle
import pandas as pd
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
import scipy as sc
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

def get_power(y):
    x = np.arange(1, len(y) + 1)
    logx = np.log(x)
    logy = np.log(y+1)
    coeffs = list(np.polyfit(logx, logy, 1))
    exponent = -coeffs[0]

    return exponent

def init_df(n):
    return pd.DataFrame({
        "mean": pd.Series(dtype="float"),
        "var": pd.Series(dtype="float"),
        "kurtosis": pd.Series(dtype="float"),
        "srange": pd.Series(dtype="float"),
        "r_plaw1": pd.Series(dtype="float"),
        "r_plaw2": pd.Series(dtype="float"),
        "abs_ac_plaw": pd.Series(dtype="float"),
        "GARCH": pd.Series(dtype="float"),
        "ARCH": pd.Series(dtype="float"),
        "LM": pd.Series(dtype="float"),
        "FIGARCH": pd.Series(dtype="float"),
    }, index=range(n))

def init_locs():
    return {
        "mean": 0,
        "var": 1,
        "kurtosis": 2,
        "srange": 3,
        "r_plaw1": 4,
        "r_plaw2": 5,
        "abs_ac_plaw": 6,
        "sq_ac_plaw": 7,
        "GARCH": 8,
        "ARCH": 9,
        "LM": 10,
        "FIGARCH": 11,
    }

def get_autocorr(returns):
    mean = np.mean(returns)
    corr_returns = returns - mean
    std = np.std(returns)
    sq_returns = returns ** 2
    corr_sq_returns = sq_returns - np.mean(sq_returns)
    abs_returns = abs(returns)
    corr_abs_returns = abs_returns - np.mean(abs_returns)

    autocorr1 = np.correlate(corr_returns, corr_returns, mode='full')
    autocorr1 = autocorr1 / (len(corr_returns) * (std ** 2))
    autocorr1 = autocorr1[len(autocorr1) // 2 + 1:]

    autocorr2 = np.correlate(corr_sq_returns, corr_sq_returns, mode='full')
    autocorr2 = autocorr2 / (len(corr_sq_returns) * (np.std(sq_returns) ** 2))
    autocorr2 = autocorr2[len(autocorr2) // 2 + 1:]

    autocorr3 = np.correlate(corr_abs_returns, corr_abs_returns, mode='full')
    autocorr3 = autocorr3 / (len(abs_returns) * (np.std(abs_returns) ** 2))
    autocorr3 = autocorr3[len(autocorr3) // 2 + 1:]

    abs_power = get_power(autocorr3)
    sq_power = get_power(autocorr2)

    return autocorr1, autocorr2, autocorr3, abs_power, sq_power

def get_figarch(returns, n, values, locs):
    model = arch.univariate.ARX(returns, lags=1, rescale=False)
    model.volatility = arch.univariate.volatility.FIGARCH(p=1, q=1)
    res = model.fit()
    values.iloc[n, locs['GARCH']] = res.params['omega']
    values.iloc[n, locs['ARCH']] = res.params['phi']
    values.iloc[n, locs['LM']] = res.params['d']
    values.iloc[n, locs['FIGARCH']] = res.params['beta']

    return values
    # returns = abs(returns)
    # arma = ARIMA(returns, order=(1, 0, 0)).fit()
    # resid = arma.resid * 100
    # returns = abs(returns) * 100
    # fit = arch_model(returns, vol="FIGARCH").fit()
    # res = acorr_ljungbox(fit.std_resid ** 2, 100)
    # res2 = fit.arch_lm_test(100, standardized=True)
    pass

def single_post_process(df, n, values, locs):
    returns = np.asarray(df['return'])
    returns = returns[999:]

    values.iloc[n, locs['mean']] = np.mean(returns)
    values.iloc[n, locs['var']] = np.var(returns)
    values.iloc[n, locs['kurtosis']] = np.kurtosis(returns)
    values.iloc[n, locs['srange']] = sc.stats.studentized_range(returns)[0]

    autocorr1, autocorr2, autocorr3, abs_power, sq_power = get_autocorr(returns)
    values.iloc[n, locs['abs_ac_plaw']] = abs_power
    values.iloc[n, locs['sq_ac_plaw']] = sq_power

    plaw1, plaw2 = get_plaws(returns)
    values.iloc[n, locs['r_plaw1']] = plaw1
    values.iloc[n, locs['r_plaw2']] = plaw2

    values_df = get_figarch(returns, n, values, locs)

    return autocorr1, autocorr2, autocorr3, values_df

    # fit = arch_model(norm_returns, vol="FIGARCH").fit()
    # res = acorr_ljungbox(fit.std_resid**2, 100)
    # res2 = fit.arch_lm_test(100, standardized=True)
    # res3 = sc.jarque_bera(fit.std_resid)
    # res4 = sc.shapiro(fit.std_resid)


def powerlaw(x, power, inter):
    return inter*(x**-power)

def get_plaws(data):
    abso = abs(data)
    x = np.sort(abso)
    y = np.arange(1, len(x) + 1) / len(x)
    y = 1 - y

    cond1 = np.mean(abso) + np.std(abso)
    cond2 = np.mean(abso) + (np.std(abso)*2)

    indexes1 = np.where((x >= cond1) & (x < cond2))
    indexes2 = np.where(x >= cond2)
    x1 = x[indexes1]
    y1 = y[indexes1]
    x2 = x[indexes2]
    y2 = y[indexes2]

    plaw1 = sc.optimize.curve_fit(
        powerlaw,
        xdata=x1,
        ydata=y1,
        p0=[-0.3, 2],
        maxfev=10000
    )

    plaw2 = sc.optimize.curve_fit(
        powerlaw,
        xdata=x2,
        ydata=y2,
        p0=[-0.3, 2],
        maxfev=10000
    )

    # xxs1 = [powerlaw(xx, plaw1[0][0], plaw1[0][1]) for xx in x1]
    # xxs2 = [powerlaw(xx, plaw2[0][0], plaw2[0][1]) for xx in x2]
    #
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.xlim([10**-3, 1])
    # # plt.ylim([10**-4, 1])
    # plt.plot(x, y, marker='o', linestyle='-', color='b')
    # plt.plot(x1, xxs1,  linestyle='-', color='r')
    # plt.plot(x2, xxs2,  linestyle='-', color='y')
    # plt.show()
    #
    # print(str(plaw1))
    # print(str(plaw2))

    return -plaw1[0][0], -plaw2[0][0]

def plot():
    autocorr1 = pickle.load(open("autocorr1.p", "rb"))
    autocorr1 = autocorr1[:200]
    autocorr2 = pickle.load(open("autocorr2.p", "rb"))
    autocorr2 = autocorr2[:200]
    autocorr3 = pickle.load(open("autocorr3.p", "rb"))
    autocorr3 = autocorr3[:200]
    values = pickle.load(open("values.p", 'rb'))
    print(str(values[0]) + '\n')
    print(str(values[1]) + '\n')

    lag = np.arange(1, len(autocorr1) + 1)
    plt.stem(lag, autocorr1, basefmt='gray')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Returns')
    plt.show()
    plt.stem(lag, autocorr2, basefmt='gray')
    plt.title('Squared Returns')
    plt.show()
    plt.stem(lag, autocorr3, basefmt='gray')
    plt.title('Absolute Returns')
    plt.show()

if __name__ == '__main__':
    sim = Simulation()
    n = 1000
    locs = init_locs()
    returns = np.zeros((n, 5000))
    abs_returns = np.zeros((n, 5000))
    sq_returns = np.zeros((n, 5000))
    values_df = init_df(n)

    for i in range(0, n):
        sim.reset()
        sim.run()
        autocorr1, autocorr2, autocorr3, values = single_post_process(sim.df, i, values_df, locs)
        returns[i, :] = autocorr1
        abs_returns[i, :] = autocorr2
        sq_returns[i, :] = autocorr3
        values_df = values

    values_df.loc['mean'] = values_df.mean()

    cwd = os.path.dirname(os.path.realpath(__file__))
    # # price_path = os.path.join(cwd, "../Scriptie/data", "data.p")
    # price_path = os.path.join(cwd, "data.p")
    # price_path = os.path.abspath(price_path)

    autocorr_path1 = os.path.join(cwd, "data", "autocorr1.p")
    autocorr_path1 = os.path.abspath(autocorr_path1)
    pickle.dump(np.mean(returns, axis=0), open(autocorr_path1, "wb"))

    autocorr_path2 = os.path.join(cwd, "data", "autocorr2.p")
    autocorr_path2 = os.path.abspath(autocorr_path2)
    pickle.dump(np.mean(abs_returns, axis=0), open(autocorr_path2, "wb"))

    autocorr_path3 = os.path.join(cwd, "data", "autocorr3.p")
    autocorr_path3 = os.path.abspath(autocorr_path3)
    pickle.dump(np.mean(sq_returns, axis=0), open(autocorr_path3, "wb"))

    values_path1 = os.path.join(cwd, "data", "values.p")
    values_path1 = os.path.abspath(values_path1)
    pickle.dump(values_df, open(values_path1, "wb"))
