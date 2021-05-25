import pandas as pd

d = {'Predictor I': [2, 4, 6, 6], 'Predictor II': [5, 5, 6, 5], 'Predictor III': [10, 5, 8, 8]}
df = pd.DataFrame(data=d)

d1 = {'Real Value': [6, 5, 6, 7]}
df1 = pd.DataFrame(data=d1)

solution_0 = {'System Disagreement': [3.555556, 0.444444, 0.888889, 1.3333333]}
solution_disagreement = pd.DataFrame(data=solution_0)

solution_1 = {0: [3.666667, 0.666667, 0.666667, 1.000000], 1: [2.666667, 0.333333, 0.666667, 1.333333],
              2: [4.333333, 0.333333, 1.333333, 1.666667]}
solution_predictor = pd.DataFrame(data=solution_1)

list_0 = [1, 3, 4, 100, [1, "test"], 23, [23, "test2"]]

solution_list_0 = [1, 3, 4, 100, 1, 23, 23]

list_1 = [2, 5, 10]
value_2 = 6

solution_list_weights = [0.83333333333333334, 1.3333333333333333, 0.8333333333333334]

solution_consolidation = [5.666666666666667, 4.722222222222222, 7.0, 5.5]

solution_consolidation_memory = [5.666666666666667, 4.694444444444445, 6.7407407407407405, 6.111111111111111]

solution_consolidation_anchor = [6.12, 4.626288659793814, 7.005263157894737, 6.114705882352942]

solution_average_consolidation = [5.666666666666667, 4.666666666666667, 6.666666666666667, 6.333333333333333]



