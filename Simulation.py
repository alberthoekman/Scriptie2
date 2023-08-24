from Fundamentalist import Fundamentalist
from Chartist import Chartist
from Market import Market
from numpy.random import default_rng
import numpy as np
import pandas as pd


class Simulation:
    def __init__(self):
        self.market = Market(self)
        self.rng = default_rng()
        self.t_max = 6000
        self.start_price = 100
        self.annual = 250
        self.n_fundamentalists = 1
        self.n_chartists = 9

        self.risk_free_rate = 0.05
        self.risk_free_return = 1 + (self.risk_free_rate / self.annual)
        self.volatility = 0.2
        self.fundamental_mean = 100
        self.fundamental_variance = ((self.volatility * self.fundamental_mean) ** 2) / self.annual
        self.error_variance = 0.01265
        self.decay_rate = 0.85
        # self.noise_demand_variance = 1 * ((self.n_fundamentalists + self.n_chartists) / 2)
        self.noise_demand_variance = 1

        self.market_rate = 2 / (self.n_fundamentalists + self.n_chartists)
        self.fundamentalist_rate = 0.1
        self.fundamentalist_aversion = 0.8
        self.chartist_rate = 0.3
        self.chartist_aversion = 0.8
        self.chartist_reaction = 1

        self.fundamentalists = []
        self.chartists = []
        self.rng = default_rng()
        self.df = self.init_df()
        self.locs = self.init_locs()
        self.populate_agents()

        # self.dividend_mean = self.fundamental_mean * (self.risk_free_return - 1)
        # self.dividend_variance = (self.risk_free_rate ** 2) * self.fundamental_variance

    def reset(self):
        self.df = False
        self.df = self.init_df()

    def run(self):
        # np.seterr(all='raise')
        current = self.df.iloc[0, self.locs["price"]]
        fundamental = self.df.iloc[0, self.locs["fundamental"]]
        sample_mean = self.df.iloc[0, self.locs["mean"]]
        sample_var = self.df.iloc[0, self.locs["var"]]

        for t in range(0, self.t_max):
            f_demand = 0
            c_demand = 0

            if self.n_chartists > 0:
                for chartist in self.chartists:
                    c_demand += chartist.calculate_demand(sample_mean, sample_var, current)

            if self.n_fundamentalists > 0:
                for fundamentalist in self.fundamentalists:
                    # est_fundamental = self.market.calculate_next_fundamental(fundamental)
                    f_demand += fundamentalist.calculate_demand(current, fundamental)

            self.df.iloc[t, self.locs["f_demand"]] = f_demand
            self.df.iloc[t, self.locs["c_demand"]] = c_demand

            if t != self.t_max-1:
                new = self.market.calculate_new_price(current, f_demand, c_demand)
                returns = np.log(new) - np.log(current)
                current = new
                fundamental = self.market.calculate_next_fundamental(fundamental)
                sample_var = self.market.calculate_next_sample_variance(sample_var, sample_mean, current)
                sample_mean = self.market.calculate_next_sample_mean(sample_mean, current)

                self.df.iloc[t+1, self.locs["price"]] = current
                self.df.iloc[t+1, self.locs["return"]] = returns
                self.df.iloc[t+1, self.locs["fundamental"]] = fundamental
                self.df.iloc[t+1, self.locs["mean"]] = sample_mean
                self.df.iloc[t+1, self.locs["var"]] = sample_var

    def populate_agents(self):
        if self.n_fundamentalists > 0:
            for i in range(0, self.n_fundamentalists):
                fund = Fundamentalist(
                    self,
                    self.fundamentalist_rate,
                    self.fundamentalist_aversion
                )

                self.fundamentalists.append(fund)

        if self.n_chartists > 0:
            for i in range(0, self.n_chartists):
                chart = Chartist(
                    self,
                    self.chartist_rate,
                    self.chartist_aversion,
                    # self.rng.uniform(0.01, 1)
                    self.chartist_reaction
                )

                self.chartists.append(chart)

    def init_df(self):
        df = pd.DataFrame({
            "t": pd.Series(dtype="int"),
            "price": pd.Series(dtype="float"),
            "return": pd.Series(dtype="float"),
            "f_demand": pd.Series(dtype="float"),
            "c_demand": pd.Series(dtype="float"),
            "mean": pd.Series(dtype="float"),
            "var": pd.Series(dtype="float"),
            "fundamental": pd.Series(dtype="float"),
        }, index=range(self.t_max))

        df["t"] = range(0, self.t_max)
        df.iloc[0, 1] = self.start_price
        df.iloc[0, 2] = 0
        df.iloc[0, 3] = 0
        df.iloc[0, 4] = 0
        df.iloc[0, 5] = self.start_price
        df.iloc[0, 6] = self.fundamental_variance
        df.iloc[0, 7] = self.fundamental_mean

        return df

    def init_locs(self):
        return {
            "t": 0,
            "price": 1,
            "return": 2,
            "f_demand": 3,
            "c_demand": 4,
            "mean": 5,
            "var": 6,
            "fundamental": 7,
        }


