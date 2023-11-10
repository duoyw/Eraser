import random


class Histogram:
    def __init__(self, data):
        """
        :param data: [(bin_floor,size),(bin_floor,size),...(max,0)]
        """
        self.data = data
        self.random = random.Random()

    def pick_values_from_each_bin(self, count_for_bin):
        # [(minBound,maxB,choose_vals),]
        vals_for_bins = []
        for i in range(len(self.data) - 1):
            min_val = self.data[i][0]
            max_val = self.data[i + 1][0]
            vals = []
            for j in range(count_for_bin):
                vals.append(self.random.uniform(float(min_val), float(max_val)))
            vals_for_bins.append((min_val, max_val, vals))
        return vals_for_bins
