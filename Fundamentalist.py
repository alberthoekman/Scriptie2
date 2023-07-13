from Investor import Investor

class Fundamentalist(Investor):
    def calculate_demand(self, current, fundamental):
        price = self.estimate_price(current, fundamental)
        vol = self.estimate_vol()

        return price / vol

    def estimate_price(self, current, fundamental):
        one = self.rate * (fundamental - current)

        return current + one

    def estimate_vol(self):
        rfr = self.sim.risk_free_rate**2
        vol = (1 + rfr) * self.sim.fundamental_variance

        return self.aversion * vol

    # def calculate_demand(self, current, fundamental):
    #     price = self.estimate_price(current, fundamental)
    #     div = self.sim.market.calculate_next_dividend()
    #     vol = self.sim.fundamental_variance
    #     gain = self.sim.market.calculate_gain(current, price, div)
    #     averse = self.sim.fundamentalist_aversion
    #
    #     return gain / (averse * vol)
    #
    # def estimate_price(self, current, fundamental):
    #     f = self.sim.market.calculate_next_fundamental(fundamental)
    #     return current + (self.rate * (f - current))
