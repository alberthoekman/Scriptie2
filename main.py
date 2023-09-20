from Simulation import Simulation
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
import Analyze as an
from arch.utility.exceptions import ConvergenceWarning

class WarningException(Exception):
    pass

warnings.filterwarnings("always", "", ConvergenceWarning)

def init_df(ne):
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
        "fi_a_sig": pd.Series(dtype="object"),
        "fi_b_sig": pd.Series(dtype="object"),
        "fi_alpha_0_sig": pd.Series(dtype="object"),
        "fi_phi_sig": pd.Series(dtype="object"),
        "fi_d_sig": pd.Series(dtype="object"),
        "fi_beta_sig": pd.Series(dtype="object"),
        "r_plaw3": pd.Series(dtype="float"),
        "min": pd.Series(dtype="float"),
        "max": pd.Series(dtype="float"),
        "skewness": pd.Series(dtype="float"),
    }, index=range(ne))


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
        "r_plaw3": 41,
        "min": 42,
        "max": 43,
        "skewness": 44,
    }


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
    returns = np.full((n, 5000), np.nan)
    abs_returns = np.full((n, 5000), np.nan)
    sq_returns = np.full((n, 5000), np.nan)
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
                autocorr1, autocorr2, autocorr3, values = an.single_post_process(sim.df, i, values_df, locs)

            for war in wars:
                if issubclass(war.category, ConvergenceWarning):
                    raise WarningException("moi")
        except WarningException as e:
            continue
        returns[i, :] = autocorr1
        abs_returns[i, :] = autocorr2
        sq_returns[i, :] = autocorr3
        values_df = values

    values_df.loc[n] = values_df.mean(numeric_only=True)
    values_df = an.process_sig(values_df, n)
    an.dump_data(np.nanmean(returns, axis=0), 'autocorr1.p')
    an.dump_data(np.nanmean(abs_returns, axis=0), 'autocorr2.p')
    an.dump_data(np.nanmean(sq_returns, axis=0), 'autocorr3.p')
    an.dump_data(values_df, 'values.p')
