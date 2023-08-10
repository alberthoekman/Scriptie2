import numpy as np
import pandas as pd
import arch
import scipy as sc
import os
import pickle


def get_ac_plaw(y):
    x = np.arange(1, len(y) + 1)
    logx = np.log(x)
    logy = np.log(y+1)
    coeffs = list(np.polyfit(logx, logy, 1))
    exponent = -coeffs[0]

    return exponent


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

    abs_power = get_ac_plaw(autocorr3)
    sq_power = get_ac_plaw(autocorr2)

    return autocorr1, autocorr2, autocorr3, abs_power, sq_power


def get_figarch(returns, n, values, locs):
    model = arch.univariate.ARX(returns, lags=1, rescale=False)
    model.volatility = arch.univariate.volatility.GARCH(p=1, q=1)
    model.distribution = arch.univariate.distribution.StudentsT()
    res = model.fit(disp=False)

    values.iloc[n, locs['garch_a']] = res.params['omega']
    values.iloc[n, locs['garch_b']] = res.params['y[1]']
    values.iloc[n, locs['garch_alpha_0']] = res.params['Const'] / 1000
    values.iloc[n, locs['garch_alpha_1']] = res.params['alpha[1]']
    values.iloc[n, locs['garch_beta']] = res.params['beta[1]']

    values.iloc[n, locs['garch_a_std']] = res.std_err['omega']
    values.iloc[n, locs['garch_b_std']] = res.std_err['y[1]']
    values.iloc[n, locs['garch_alpha_0_std']] = res.std_err['Const'] / 1000
    values.iloc[n, locs['garch_alpha_1_std']] = res.std_err['alpha[1]']
    values.iloc[n, locs['garch_beta_std']] = res.std_err['beta[1]']

    values.iloc[n, locs['garch_a_sig']] = res.pvalues['omega'] <= 0.05
    values.iloc[n, locs['garch_b_sig']] = res.pvalues['y[1]'] <= 0.05
    values.iloc[n, locs['garch_alpha_0_sig']] = res.pvalues['Const'] <= 0.05
    values.iloc[n, locs['garch_alpha_1_sig']] = res.pvalues['alpha[1]'] <= 0.05
    values.iloc[n, locs['garch_beta_sig']] = res.pvalues['beta[1]'] <= 0.05

    model.volatility = arch.univariate.volatility.FIGARCH(p=1, q=1)
    res = model.fit(disp=False)

    values.iloc[n, locs['fi_a']] = res.params['omega']
    values.iloc[n, locs['fi_b']] = res.params['y[1]']
    values.iloc[n, locs['fi_alpha_0']] = res.params['Const'] / 1000
    values.iloc[n, locs['fi_phi']] = res.params['phi']
    values.iloc[n, locs['fi_d']] = res.params['d']
    values.iloc[n, locs['fi_beta']] = res.params['beta']

    values.iloc[n, locs['fi_a_std']] = res.std_err['omega']
    values.iloc[n, locs['fi_b_std']] = res.std_err['y[1]']
    values.iloc[n, locs['fi_alpha_0_std']] = res.std_err['Const'] / 1000
    values.iloc[n, locs['fi_phi_std']] = res.std_err['phi']
    values.iloc[n, locs['fi_d_std']] = res.std_err['d']
    values.iloc[n, locs['fi_beta_std']] = res.std_err['beta']

    values.iloc[n, locs['fi_a_sig']] = res.pvalues['omega'] <= 0.05
    values.iloc[n, locs['fi_b_sig']] = res.pvalues['y[1]'] <= 0.05
    values.iloc[n, locs['fi_alpha_0_sig']] = res.pvalues['Const'] <= 0.05
    values.iloc[n, locs['fi_phi_sig']] = res.pvalues['phi'] <= 0.05
    values.iloc[n, locs['fi_d_sig']] = res.pvalues['d'] <= 0.05
    values.iloc[n, locs['fi_beta_sig']] = res.pvalues['beta'] <= 0.05

    return values


def powerlaw(x, power, inter):
    return inter*(x**-power)


def get_ret_plaws(data):
    abso = abs(data)
    x = np.sort(abso)
    y = np.arange(1, len(x) + 1) / len(x)
    y = 1 - y

    cond1 = np.mean(abso) + np.std(abso)
    # cond2 = np.mean(abso) + (np.std(abso)*2)

    # indexes1 = np.where((x >= cond1) & (x < cond2))
    # indexes2 = np.where(x >= cond2)
    indexes3 = np.where(x >= cond1)
    # x1 = x[indexes1]
    # y1 = y[indexes1]
    # x2 = x[indexes2]
    # y2 = y[indexes2]
    x3 = x[indexes3]
    y3 = y[indexes3]

    # if len(y1) > 0:
    #     plaw1 = sc.optimize.curve_fit(
    #         powerlaw,
    #         xdata=x1,
    #         ydata=y1,
    #         p0=[-0.3, 2],
    #         maxfev=10000
    #     )[0][0]
    # else:
    #     plaw1 = 0
    #
    # if len(y2) > 0:
    #     plaw2 = sc.optimize.curve_fit(
    #         powerlaw,
    #         xdata=x2,
    #         ydata=y2,
    #         p0=[-0.3, 2],
    #         maxfev=10000
    #     )[0][0]
    # else:
    #     plaw2 = 0

    if len(y3) > 0:
        plaw3 = sc.optimize.curve_fit(
            powerlaw,
            xdata=x3,
            ydata=y3,
            p0=[-0.3, 2],
            maxfev=10000
        )[0][0]
    else:
        plaw3 = 0

    return plaw3

def process_sig(values_df, n):
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

    return values_df

def dump_data(data, name):
    cwd = os.path.dirname(os.path.realpath(__file__))
    padda = os.path.join(cwd, "data5", name)
    padda = os.path.abspath(padda)
    pickle.dump(data, open(padda, "wb"))

def single_post_process(df, n, values, locs):
    returns = np.asarray(df['return'])
    returns = returns[999:]

    values.iloc[n, locs['mean']] = np.mean(returns)
    values.iloc[n, locs['var']] = np.var(returns)
    values.iloc[n, locs['kurtosis']] = sc.stats.kurtosis(returns)
    values.iloc[n, locs['skewness']] = sc.stats.skew(returns)
    values.iloc[n, locs['min']] = min(returns)
    values.iloc[n, locs['max']] = max(returns)
    values.iloc[n, locs['srange']] = (max(returns) - min(returns)) / np.std(returns)

    returns = returns * 1000

    autocorr1, autocorr2, autocorr3, abs_power, sq_power = get_autocorr(returns)
    values.iloc[n, locs['abs_ac_plaw']] = abs_power
    values.iloc[n, locs['sq_ac_plaw']] = sq_power

    plaw3 = get_ret_plaws(returns)
    # values.iloc[n, locs['r_plaw1']] = plaw1
    # values.iloc[n, locs['r_plaw2']] = plaw2
    values.iloc[n, locs['r_plaw3']] = plaw3

    values_df = get_figarch(returns, n, values, locs)

    return autocorr1, autocorr2, autocorr3, values_df
