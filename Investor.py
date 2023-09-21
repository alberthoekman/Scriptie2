from abc import ABCMeta, abstractmethod

class Investor(metaclass=ABCMeta):
    def __init__(self, simulation, rate, aversion):
        self.rate = rate
        self.sim = simulation
        self.aversion = aversion

    def calculate_gain(self, expected, current):
        r = self.sim.risk_free_return

        return expected + self.sim.dividend_mean - (r * current)

    # def calculate_wealth(self, gain):
    #     r = self.sim.risk_free_return
    #
    #     if self.shares == 0:
    #         self.wealth = r * self.wealth
    #         return
    #
    #     wealth = (r * self.wealth) + (gain * self.shares)
    #     self.wealth = wealth
