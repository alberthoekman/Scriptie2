from Investor import Investor

class Fundamentalist(Investor):
    def calculate_demand(self, current, fundamental):
        price = self.estimate_price(current, fundamental)
        gain = self.calculate_gain(price, current)
        vol = self.estimate_vol()

        return gain / vol

    def estimate_price(self, current, fundamental):
        one = self.rate * (fundamental - current)

        return current + one

    def estimate_vol(self):
        one = (1 + self.sim.risk_free_rate ** 2) * self.sim.fundamental_variance
        return self.aversion * one

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
