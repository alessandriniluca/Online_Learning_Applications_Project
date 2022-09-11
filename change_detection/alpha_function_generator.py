import numpy as np

class AlphaFunctionGenerator:

    def __init__(self):
        self.amplitude = [
            [[0.0, 70.0, 55.0, 5.0, 10.0], [5, 55, 50, 20, 20], [10, 60, 20, 55, 20]],
            [[40.0, 0.0, 0.0, 80.0, 00.0], [55, 0, 0, 60, 10], [50, 0, 20, 10, 80]],
            [[50.0, 70.0, 75.0, 75.0, 60.0], [85, 45, 70, 50, 35], [50, 50, 50, 15, 45]],

        ]
        self.tau = [
            [[-0.034, -0.045, -0.04, -0.044, -0.039], [-0.04, -0.035, -0.049, -0.036, -0.039], [-0.03, -0.04, -0.055, -0.054, -0.045]],
            [[-0.03, -0.04, -0.038, -0.042, -0.036], [-0.038, -0.045, -0.04, -0.039, -0.041], [-0.042, -0.041, -0.05, -0.05, -0.043]],
            [[-0.04, -0.035, -0.045, -0.04, -0.05], [-0.043, -0.039, -0.045, -0.03, -0.029], [-0.038, -0.045, -0.045, -0.044, -0.043]],

        ]

    def get_functions(self, step):

        def alphas_function_class_0(budget):
            increment = [(self.amplitude[step][0][0] * (1.0 - np.exp(self.tau[step][0][0] * (budget[0])))).astype(int),
                        (self.amplitude[step][0][1] * (1.0 - np.exp(self.tau[step][0][1] * (budget[1])))).astype(int),
                        (self.amplitude[step][0][2] * (1.0 - np.exp(self.tau[step][0][2] * (budget[2])))).astype(int),
                        (self.amplitude[step][0][3] * (1.0 - np.exp(self.tau[step][0][3] * (budget[3])))).astype(int),
                        (self.amplitude[step][0][4] * (1.0 - np.exp(self.tau[step][0][4] * (budget[4])))).astype(int)]
            return np.array(increment)


        def alphas_function_class_1(budget):
            increment = [(self.amplitude[step][1][0] * (1.0 - np.exp(self.tau[step][1][0] * (budget[0])))).astype(int),
                        (self.amplitude[step][1][1] * (1.0 - np.exp(self.tau[step][1][1] * (budget[1])))).astype(int),
                        (self.amplitude[step][1][2] * (1.0 - np.exp(self.tau[step][1][2] * (budget[2])))).astype(int),
                        (self.amplitude[step][1][3] * (1.0 - np.exp(self.tau[step][1][3] * (budget[3])))).astype(int),
                        (self.amplitude[step][1][4] * (1.0 - np.exp(self.tau[step][1][4] * (budget[4])))).astype(int)]

            return np.array(increment)


        def alphas_function_class_2(budget):
            increment = [(self.amplitude[step][2][0] * (1.0 - np.exp(self.tau[step][2][0] * (budget[0])))).astype(int),
                        (self.amplitude[step][2][1] * (1.0 - np.exp(self.tau[step][2][1] * (budget[1])))).astype(int),
                        (self.amplitude[step][2][2] * (1.0 - np.exp(self.tau[step][2][2] * (budget[2])))).astype(int),
                        (self.amplitude[step][2][3] * (1.0 - np.exp(self.tau[step][2][3] * (budget[3])))).astype(int),
                        (self.amplitude[step][2][4] * (1.0 - np.exp(self.tau[step][2][4] * (budget[4])))).astype(int)]
            return np.array(increment)

        return [alphas_function_class_0, alphas_function_class_1, alphas_function_class_2]