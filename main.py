import arch.univariate.mean

from Simulation import Simulation
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

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
        "sq_ac_plaw": pd.Series(dtype="float"),
        "garch_a": pd.Series(dtype="float"),
        "garch_b": pd.Series(dtype="float"),
        "garch_alpha_0": pd.Series(dtype="float"),
        "garch_alpha_1": pd.Series(dtype="float"),
        "garch_beta": pd.Series(dtype="float"),
        "garch_a_std": pd.Series(dtype="float"),
        "garch_b_std": pd.Series(dtype="float"),
        "garch_alpha_0_std": pd.Series(dtype="float"),
        "garch_alpha_1_std": pd.Series(dtype="float"),
        "garch_beta_std": pd.Series(dtype="float"),
        "garch_a_sig": pd.Series(),
        "garch_b_sig": pd.Series(),
        "garch_alpha_0_sig": pd.Series(),
        "garch_alpha_1_sig": pd.Series(),
        "garch_beta_sig": pd.Series(),
        "fi_a": pd.Series(dtype="float"),
        "fi_b": pd.Series(dtype="float"),
        "fi_alpha_0": pd.Series(dtype="float"),
        "fi_phi": pd.Series(dtype="float"),
        "fi_d": pd.Series(dtype="float"),
        "fi_beta": pd.Series(dtype="float"),
        "fi_a_std": pd.Series(dtype="float"),
        "fi_b_std": pd.Series(dtype="float"),
        "fi_alpha_0_std": pd.Series(dtype="float"),
        "fi_phi_std": pd.Series(dtype="float"),
        "fi_d_std": pd.Series(dtype="float"),
        "fi_beta_std": pd.Series(dtype="float"),
        "fi_a_sig": pd.Series(),
        "fi_b_sig": pd.Series(),
        "fi_alpha_0_sig": pd.Series(),
        "fi_phi_sig": pd.Series(),
        "fi_d_sig": pd.Series(),
        "fi_beta_sig": pd.Series(),
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
        "garch_a": 8,
        "garch_b": 9,
        "garch_alpha_0": 10,
        "garch_alpha_1": 11,
        "garch_beta": 12,
        "garch_a_std": 13,
        "garch_b_std": 14,
        "garch_alpha_0_std": 15,
        "garch_alpha_1_std": 16,
        "garch_beta_std": 17,
        "garch_a_sig": 18,
        "garch_b_sig": 19,
        "garch_alpha_0_sig": 20,
        "garch_alpha_1_sig": 21,
        "garch_beta_sig": 22,
        "fi_a": 23,
        "fi_b": 24,
        "fi_alpha_0": 25,
        "fi_phi": 26,
        "fi_d": 27,
        "fi_beta": 28,
        "fi_a_std": 29,
        "fi_b_std": 30,
        "fi_alpha_0_std": 31,
        "fi_phi_std": 32,
        "fi_d_std": 33,
        "fi_beta_std": 34,
        "fi_a_sig": 35,
        "fi_b_sig": 36,
        "fi_alpha_0_sig": 37,
        "fi_phi_sig": 38,
        "fi_d_sig": 39,
        "fi_beta_sig": 40,
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
    model.volatility = arch.univariate.volatility.GARCH(p=1, q=1)
    res = model.fit(disp=False)

    values.iloc[n, locs['garch_a']] = res.params['omega']
    values.iloc[n, locs['garch_b']] = res.params['y[1]']
    values.iloc[n, locs['garch_alpha_0']] = res.params['Const']
    values.iloc[n, locs['garch_alpha_1']] = res.params['alpha[1]']
    values.iloc[n, locs['garch_beta']] = res.params['beta[1]']

    values.iloc[n, locs['garch_a_std']] = res.std_err['omega']
    values.iloc[n, locs['garch_b_std']] = res.std_err['y[1]']
    values.iloc[n, locs['garch_alpha_0_std']] = res.std_err['Const']
    values.iloc[n, locs['garch_alpha_1_std']] = res.std_err['alpha[1]']
    values.iloc[n, locs['garch_beta_std']] = res.std_err['beta[1]']

    values.iloc[n, locs['garch_a_sig']] = res.params['omega'] >= 0.05
    values.iloc[n, locs['garch_b_sig']] = res.params['y[1]'] >= 0.05
    values.iloc[n, locs['garch_alpha_0_sig']] = res.params['Const'] >= 0.05
    values.iloc[n, locs['garch_alpha_1_sig']] = res.params['alpha[1]'] >= 0.05
    values.iloc[n, locs['garch_beta_sig']] = res.params['beta[1]'] >= 0.05

    model.volatility = arch.univariate.volatility.FIGARCH(p=1, q=1)
    res = model.fit(disp=False)

    values.iloc[n, locs['fi_a']] = res.params['omega']
    values.iloc[n, locs['fi_b']] = res.params['y[1]']
    values.iloc[n, locs['fi_alpha_0']] = res.params['Const']
    values.iloc[n, locs['fi_phi']] = res.params['phi']
    values.iloc[n, locs['fi_d']] = res.params['d']
    values.iloc[n, locs['fi_beta']] = res.params['beta']

    values.iloc[n, locs['fi_a_std']] = res.std_err['omega']
    values.iloc[n, locs['fi_b_std']] = res.std_err['y[1]']
    values.iloc[n, locs['fi_alpha_0_std']] = res.std_err['Const']
    values.iloc[n, locs['fi_phi_std']] = res.std_err['phi']
    values.iloc[n, locs['fi_d_std']] = res.std_err['d']
    values.iloc[n, locs['fi_beta_std']] = res.std_err['beta']

    values.iloc[n, locs['fi_a_sig']] = res.pvalues['omega'] >= 0.05
    values.iloc[n, locs['fi_b_sig']] = res.pvalues['y[1]'] >= 0.05
    values.iloc[n, locs['fi_alpha_0_sig']] = res.pvalues['Const'] >= 0.05
    values.iloc[n, locs['fi_phi_sig']] = res.pvalues['phi'] >= 0.05
    values.iloc[n, locs['fi_d_sig']] = res.pvalues['d'] >= 0.05
    values.iloc[n, locs['fi_beta_sig']] = res.pvalues['beta'] >= 0.05

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
    values.iloc[n, locs['kurtosis']] = sc.stats.kurtosis(returns)
    values.iloc[n, locs['srange']] = (max(returns) - min(returns)) / np.std(returns)

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

    if len(y1) > 0:
        plaw1 = sc.optimize.curve_fit(
            powerlaw,
            xdata=x1,
            ydata=y1,
            p0=[-0.3, 2],
            maxfev=10000
        )[0][0]
    else:
        plaw1 = 0

    if len(y2) > 0:
        plaw2 = sc.optimize.curve_fit(
            powerlaw,
            xdata=x2,
            ydata=y2,
            p0=[-0.3, 2],
            maxfev=10000
        )[0][0]
    else:
        plaw2 = 0

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

    return -plaw1, -plaw2

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
    # cwd = os.path.dirname(os.path.realpath(__file__))
    # padda = os.path.join(cwd, "../Scriptie2/data/", "test.p")
    # padda = os.path.abspath(padda)
    #
    # ret = pickle.load(open(padda, 'rb'))
    # fit = get_figarch(ret, 0, [], [])
    # pass
    # sim = Simulation()
    # sim.run()
    # returns = np.asarray(sim.df['return'])
    # returns = returns[999:]
    # cwd = os.path.dirname(os.path.realpath(__file__))
    # autocorr_path1 = os.path.join(cwd, "data", "test.p")
    # autocorr_path1 = os.path.abspath(autocorr_path1)
    # pickle.dump(returns, open(autocorr_path1, "wb"))

    # exit()
    sim = Simulation()
    n = 1000
    locs = init_locs()
    returns = np.zeros((n, 5000))
    abs_returns = np.zeros((n, 5000))
    sq_returns = np.zeros((n, 5000))
    values_df = init_df(n)

    for i in range(0, n):
        sim.reset()
        try:
            sim.run()
        except (OverflowError, FloatingPointError):
            print(str(i) + '\n')
            print(sim.df['price'].to_string(index=False))
        autocorr1, autocorr2, autocorr3, values = single_post_process(sim.df, i, values_df, locs)
        returns[i, :] = autocorr1
        abs_returns[i, :] = autocorr2
        sq_returns[i, :] = autocorr3
        values_df = values

    values_df.loc[n] = values_df.mean(numeric_only=True)

    a = values_df.loc[0:n, 'garch_a_sig'].value_counts(normalize=True)
    b = values_df.loc[0:n, 'garch_b_sig'].value_counts(normalize=True)
    c = values_df.loc[0:n, 'garch_alpha_0_sig'].value_counts(normalize=True)
    d = values_df.loc[0:n, 'garch_alpha_1_sig'].value_counts(normalize=True)
    e = values_df.loc[0:n, 'garch_beta_sig'].value_counts(normalize=True)

    a.index = a.index.astype('string')
    b.index = b.index.astype('string')
    c.index = c.index.astype('string')
    d.index = d.index.astype('string')
    e.index = e.index.astype('string')

    values_df['garch_a_sig'].at[n] = a['True'] if 'True' in a else 0.0
    values_df['garch_b_sig'].at[n] = b['True'] if 'True' in b else 0.0
    values_df['garch_alpha_0_sig'].at[n] = c['True'] if 'True' in c else 0.0
    values_df['garch_alpha_1_sig'].at[n] = d['True'] if 'True' in d else 0.0
    values_df['garch_beta_sig'].at[n] = e['True'] if 'True' in e else 0.0

    f = values_df.loc[0:n, 'fi_a_sig'].value_counts(normalize=True)
    g = values_df.loc[0:n, 'fi_b_sig'].value_counts(normalize=True)
    h = values_df.loc[0:n, 'fi_alpha_0_sig'].value_counts(normalize=True)
    i = values_df.loc[0:n, 'fi_phi_sig'].value_counts(normalize=True)
    j = values_df.loc[0:n, 'fi_d_sig'].value_counts(normalize=True)
    k = values_df.loc[0:n, 'fi_beta_sig'].value_counts(normalize=True)

    f.index = f.index.astype('string')
    g.index = g.index.astype('string')
    h.index = h.index.astype('string')
    i.index = i.index.astype('string')
    j.index = j.index.astype('string')
    k.index = k.index.astype('string')

    values_df['fi_a_sig'].at[n] = f['True'] if 'True' in f else 0.0
    values_df['fi_b_sig'].at[n] = g['True'] if 'True' in g else 0.0
    values_df['fi_alpha_0_sig'].at[n] = h['True'] if 'True' in h else 0.0
    values_df['fi_phi_sig'].at[n] = i['True'] if 'True' in i else 0.0
    values_df['fi_d_sig'].at[n] = j['True'] if 'True' in j else 0.0
    values_df['fi_beta_sig'].at[n] = k['True'] if 'True' in k else 0.0

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
