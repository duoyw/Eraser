class GlobalVariable:
    variable = {}

    @classmethod
    def add(cls, name, add_value, default_if_absent=None):
        if name not in cls.variable:
            cls.variable[name] = default_if_absent + add_value
        cls.variable[name] = cls.variable[name] + add_value

    @classmethod
    def get(cls, name, default_if_absent=None):
        if name not in cls.variable:
            return default_if_absent
        return cls.variable[name]
