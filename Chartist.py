from Investor import Investor

class Chartist(Investor):
    def __init__(self, sim, rate, aversion, reaction):
        super().__init__(sim, rate, aversion)
        self.reaction = reaction

    def calculate_demand(self, sample_mean, sample_var, current):
        price = self.estimate_price(current, sample_mean)
        vol = self.estimate_vol(sample_var)

        return price / vol

    def estimate_price(self, current, sample_mean):
        one = self.rate * (current - sample_mean)

        return current + one

    def estimate_vol(self, sample_var):
        rfr = self.sim.risk_free_rate ** 2
        one = self.reaction * sample_var
        two = 1 + rfr + one
        var = self.sim.fundamental_variance

        return self.aversion * var * two


    # def calculate_demand(self, sample_mean, sample_var, current):
    #     price = current + (self.rate * (current - sample_mean))
    #     div = self.sim.market.calculate_next_dividend()
    #     reaction = self.sim.chartist_reaction
    #     vol = self.sim.fundamental_variance + (reaction * sample_var)
    #
    #     gain = self.sim.market.calculate_gain(current, price, div)
    #     averse = self.sim.chartist_aversion
    #
    #     return gain / (averse * vol)

