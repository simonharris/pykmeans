"""Dataset class definition"""


class Dataset():
    """Mimics the sklearn dataset interface"""

    def __init__(self, name, data, target):
        self.name = name
        self.data = data
        self.target = target
