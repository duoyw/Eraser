class EncordCollector:
    def __init__(self):
        self.encord_vals = []

    def add(self, encord):
        self.encord_vals.append(encord)

    def get_all(self):
        return self.encord_vals

    def clear(self):
        self.encord_vals.clear()


encord_collector = EncordCollector()
