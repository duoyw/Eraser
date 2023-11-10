class Batch:
    def __init__(self, values, batch_size):
        self.values = values
        self.cur_idx = 0
        self.batch_size = batch_size

    def next(self):
        start = self.cur_idx
        end = min(len(self.values), self.cur_idx + self.batch_size)
        self.cur_idx = end
        return self.values[start:end]

    def is_end(self):
        return self.cur_idx == len(self.values)
