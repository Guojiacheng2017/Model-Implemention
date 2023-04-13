import numpy as np

# linear classification

# 首先准备数据集
# data
points = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]]) # 自定义的数据集

def compute_error(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b))**2
    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0; w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - (w_current * x - b_current))
        w_gradient += -(2/N) * x * (y - (w_current * x - b_current))
    new_b = b_current - learning_rate * b_gradient
    new_w = w_current - learning_rate * w_gradient
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(0, num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt("../data/data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points)))
    print("Running...")
    print()
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} interations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points)))

if __name__ == '__main__':
    run()