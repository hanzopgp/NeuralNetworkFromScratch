class ActivationStep:

    def __init__(self):
        self.output = []

    def forward(self, inputs):  # outdated most of the time because it makes regression harder
        res = []
        for value in inputs:
            if value > 0:
                res.append(1)
            else:
                res.append(0)
        self.output = res
