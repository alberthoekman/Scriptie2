from Simulation import Simulation
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
from arch.utility.exceptions import ConvergenceWarning
import arch
import pickle
import os
import gc
import sys

class WarningException(Exception):
    pass

warnings.filterwarnings("always", "", ConvergenceWarning)

def dump_data(data, name):
    actual = str(name[0]) + ':' + str(name[1]) + '.p'
    cwd = os.path.dirname(os.path.realpath(__file__))
    padda = os.path.join(cwd, 'aics', actual)
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

    values_df.loc[n, 'AIC_FIGARCH_lower'] = a['True'] if 'True' in a else 0.0
    values_df.loc[n, 'BIC_FIGARCH_lower'] = b['True'] if 'True' in b else 0.0

    return values_df

def init_df(ne):
    return pd.DataFrame({
        "AIC_GARCH": pd.Series(dtype='float'),
        "AIC_FIGARCH": pd.Series(dtype='float'),
        "BIC_GARCH": pd.Series(dtype='float'),
        "BIC_FIGARCH": pd.Series(dtype='float'),
        "AIC_FIGARCH_lower": pd.Series(),
        "BIC_FIGARCH_lower": pd.Series()
    }, index=range(ne))

def init_locs():
    return {
        "AIC_GARCH": 0,
        "AIC_FIGARCH": 1,
        "BIC_GARCH": 2,
        "BIC_FIGARCH": 3,
        "AIC_FIGARCH_lower": 4,
        "BIC_FIGARCH_lower": 5
    }

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
    (0, 10),
    (95, 5)
]

locs = init_locs()

combo = combos[int(sys.argv[1])]
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
            values_df.iloc[i, locs['AIC_GARCH']] = results[0]
            values_df.iloc[i, locs['BIC_GARCH']] = results[1]
            values_df.iloc[i, locs['AIC_FIGARCH']] = results[2]
            values_df.iloc[i, locs['BIC_FIGARCH']] = results[3]
            values_df.iloc[i, locs['AIC_FIGARCH_lower']] = True if results[2] < results[0] else False
            values_df.iloc[i, locs['BIC_FIGARCH_lower']] = True if results[3] < results[1] else False

        for war in wars:
            if issubclass(war.category, ConvergenceWarning):
                raise WarningException("moi")
    except WarningException as e:
        continue

    del rets
    del results
    gc.collect()

values_df.loc[n] = values_df.mean(numeric_only=True)
values_df = process_sig(values_df, n)

# opper.loc[comboname, 'AIC_GARCH'] = np.nanmean(values_df['AIC_GARCH'])
# opper.loc[comboname, 'AIC_FIGARCH'] = np.nanmean(values_df['AIC_FIGARCH'])
# opper.loc[comboname, 'BIC_GARCH'] = np.nanmean(values_df['BIC_GARCH'])
# opper.loc[comboname, 'BIC_FIGARCH'] = np.nanmean(values_df['BIC_FIGARCH'])
# opper.loc[comboname, 'AIC_FIGARCH_lower'] = sigs[0]
# opper.loc[comboname, 'BIC_FIGARCH_lower'] = sigs[1]
#
# del sim
# del values_df
gc.collect()

dump_data(values_df, combo)
