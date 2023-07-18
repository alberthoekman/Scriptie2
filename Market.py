class Market:
    def __init__(self, simulation):
        self.sim = simulation

    def calculate_next_fundamental(self, current):
        noise = self.sim.rng.normal()
        var = self.sim.error_variance

        return current * (1 + (var * noise))

    def calculate_gain(self, prev, current):
        r = self.sim.risk_free_return

        return current - (r * prev)

    def calculate_next_sample_mean(self, prev_mean, current):
        delta = self.sim.decay_rate

        return (delta * prev_mean) + ((1-delta) * current)

    def calculate_next_sample_variance(self, prev_var, prev_mean, current):
        delta = self.sim.decay_rate

        return (delta * prev_var) + (delta * (1-delta) * ((current-prev_mean)**2))

    def calculate_new_price(self, current, f_demand, c_demand):
        net_demand = f_demand + c_demand
        rate = self.sim.market_rate
        noise = self.sim.rng.normal(0, self.sim.noise_demand_variance**2)
        new = current + (rate * net_demand) + noise

        if new <= 0:
            new = 0.01

        return new

