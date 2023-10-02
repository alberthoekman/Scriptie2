from Simulation import Simulation
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
from arch.utility.exceptions import ConvergenceWarning
import arch
import pickle
import os

class WarningException(Exception):
    pass

warnings.filterwarnings("always", "", ConvergenceWarning)

def dump_data(data, name):
    cwd = os.path.dirname(os.path.realpath(__file__))
    padda = os.path.join(cwd, name)
    padda = os.path.abspath(padda)
    pickle.dump(data, open(padda, "wb"))

def get_aic(returns):
    model = arch.univariate.ARX(returns, lags=1, rescale=False)
    model.volatility = arch.univariate.volatility.GARCH(p=1, q=1)
    res = model.fit(disp=False)

    resulta = [res.aic, res.bic]

    model.volatility = arch.univariate.volatility.FIGARCH(p=1, q=1)
    res = model.fit(disp=False)
    resulta.append(res.aic)
    resulta.append(res.bic)

    return resulta

def process_sig(values_df, n):
    a = values_df.loc[0:n, 'AIC_FIGARCH_lower'].value_counts(normalize=True)
    b = values_df.loc[0:n, 'BIC_FIGARCH_lower'].value_counts(normalize=True)

    a.index = a.index.astype('string')
    b.index = b.index.astype('string')

    res1 = a['True'] if 'True' in a else 0.0
    res2 = b['True'] if 'True' in b else 0.0

    return [res1, res2]

def init_df(ne):
    return pd.DataFrame({
        "AIC_GARCH": pd.Series(dtype='float'),
        "AIC_FIGARCH": pd.Series(dtype='float'),
        "BIC_GARCH": pd.Series(dtype='float'),
        "BIC_FIGARCH": pd.Series(dtype='float'),
        "AIC_FIGARCH_lower": pd.Series(),
        "BIC_FIGARCH_lower": pd.Series()
    }, index=range(ne))

combos = [
    (5, 5),
    (4, 6),
    (3, 7),
    (2, 8),
    (1, 9),
    (6, 4),
    (7, 3),
    (8, 2),
    (9, 1),
    (10, 0),
    (0, 10)
]

opper = pd.DataFrame({
    "AIC_GARCH": pd.Series(dtype='float'),
    "AIC_FIGARCH": pd.Series(dtype='float'),
    "BIC_GARCH": pd.Series(dtype='float'),
    "BIC_FIGARCH": pd.Series(dtype='float'),
    "AIC_FIGARCH_lower": pd.Series(),
    "BIC_FIGARCH_lower": pd.Series()
}, index=['5/5', '4/6', '3/7', '2/8', '1/9', '6/4', '7/3', '8/2', '9/1', '10/0', '0/10'])

for combo in combos:
    sim = Simulation(combo[0], combo[1])
    n = 1000
    values_df = init_df(n)

    for i in range(0, n):
        sim.reset()
        try:
            sim.run()
        except (OverflowError, FloatingPointError):
            print(str(i) + '\n')
            print(sim.df['price'].to_string(index=False))
            continue

        try:
            with warnings.catch_warnings(record=True) as wars:
                rets = np.asarray(sim.df['return'])
                rets = rets[999:]
                rets = rets * 1000
                results = get_aic(rets)
                values_df.loc[i, 'AIC_GARCH'] = results[0]
                values_df.loc[i, 'BIC_GARCH'] = results[1]
                values_df.loc[i, 'AIC_FIGARCH'] = results[2]
                values_df.loc[i, 'BIC_FIGARCH'] = results[3]
                values_df.loc[i, 'AIC_FIGARCH_lower'] = True if results[2] < results[0] else False
                values_df.loc[i, 'BIC_FIGARCH_lower'] = True if results[3] < results[1] else False

            for war in wars:
                if issubclass(war.category, ConvergenceWarning):
                    raise WarningException("moi")
        except WarningException as e:
            continue

    comboname = str(combo[0]) + '/' + str(combo[1])
    sigs = process_sig(values_df, n)

    opper.loc[comboname, 'AIC_GARCH'] = np.nanmean(values_df['AIC_GARCH'])
    opper.loc[comboname, 'AIC_FIGARCH'] = np.nanmean(values_df['AIC_FIGARCH'])
    opper.loc[comboname, 'BIC_GARCH'] = np.nanmean(values_df['BIC_GARCH'])
    opper.loc[comboname, 'BIC_FIGARCH'] = np.nanmean(values_df['BIC_FIGARCH'])
    opper.loc[comboname, 'AIC_FIGARCH_lower'] = sigs[0]
    opper.loc[comboname, 'BIC_FIGARCH_lower'] = sigs[1]

dump_data(opper, 'aic.p')
