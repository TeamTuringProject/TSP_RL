import numpy as np
from deap import base, creator, tools, algorithms
import random
import csv

# 유클리드 함수
def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

# csv 파일 읽어서 cities에 load하는 함수
def load_cities(filename):
    cities = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            cities.append([float(coord) for coord in row])
    return cities

# 표준 GA 써서 Value table 초기화 개체 수 50, 세대 100
def initialize_value_table(cities, pop_size=50, generations=100):
    num_cities = len(cities)

    # individual, population 생성
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # toolbox는 유전 알고리즘 연산자를 등록하는 컨테이너
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_cities), num_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation function
    def eval_tsp(individual):
        dist = 0
        for i in range(len(individual) - 1):
            dist += distance(cities[individual[i]], cities[individual[i + 1]])
        dist += distance(cities[individual[-1]], cities[individual[0]]) 
        return dist,
    
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", eval_tsp)
    
    pop = toolbox.population(n=pop_size)

    # 아래 유전 알고리즘에서 초기 개체, 세대 수를 낮춰야 오래 안걸림
    # 초기 개체 50, 세대 수 100일 때 20초 걸림
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, verbose=False)

    # best individual 추출
    best_ind = tools.selBest(pop, 1)[0]
    best_value = eval_tsp(best_ind)[0]
    
    value_table = np.zeros((num_cities, num_cities))  # Value table 생성

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            value_table[i][j] = value_table[j][i] = distance(cities[i], cities[j])
    
    return value_table, best_ind, best_value


# Q-Learning 함수
def q_learning(cities, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=100):
    num_cities = len(cities)
    Q = np.zeros((num_cities, num_cities))

    for episode in range(episodes):
        state = random.randint(0, num_cities - 1)
        visited = [state]
        
        while len(visited) < num_cities:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, num_cities - 1)
            else:
                action = np.argmin(Q[state, :])

            if action not in visited:
                next_state = action
                reward = -distance(cities[state], cities[next_state]) # -distance로 할 것인지, 1/distance 로 할 것인지?
                best_next_action = np.argmin(Q[next_state, :])

                # Temporal Difference 기법으로 각 단계마다 Q 값을 업데이트. 다음 상태로 이동할 때마다 즉시 Q 값을 수정
                td_target = reward + gamma * Q[next_state, best_next_action]
                td_error = td_target - Q[state, action]
                Q[state, action] += alpha * td_error

                visited.append(next_state)
                state = next_state

        # Monte Carlo 기법으로 전체 에피소드가 끝난 후에 반환값을 계산하여 Q 값을 업데이트
        reward = -distance(cities[state], cities[visited[0]])
        Q[state, visited[0]] += alpha * (reward - Q[state, visited[0]])

    return Q

# Policy 추출 함수
def extract_policy_from_q(Q):
    num_cities = Q.shape[0]
    policy = [0]
    current_city = 0
    visited = set(policy)

    while len(policy) < num_cities:
        next_city = np.argmin(Q[current_city, :])
        while next_city in visited:
            Q[current_city, next_city] = float('inf')
            next_city = np.argmin(Q[current_city, :])
        policy.append(next_city)
        visited.add(next_city)
        current_city = next_city

    policy.append(policy[0])
    return policy

# value table(도시 간의 거리를 미리 계산하여 저장한 2차원 배열)에서 Policy 추출
def extract_policy(value_table):
    num_cities = len(value_table)
    visited = [False] * num_cities
    policy = []

    current_city = 0
    policy.append(current_city)
    visited[current_city] = True

    while len(policy) < num_cities:
        next_city = np.argmin([value_table[current_city][j] if not visited[j] else float('inf') for j in range(num_cities)])
        policy.append(next_city)
        visited[next_city] = True
        current_city = next_city
    
    policy.append(policy[0]) 
    return policy

cities = load_cities('./2024_AI_TSP.csv')
value_table, best_solution, best_value = initialize_value_table(cities) # value table 초기화
# policy = extract_policy(value_table) # policy 추출(1번)

# (밑에 2번)
Q = q_learning(cities)  # Q-Learning을 통해 Q 값 학습
policy = extract_policy_from_q(Q)  # Q 값으로부터 최적 policy 추출

# policy 평가
def evaluate_policy(policy, cities):
    total_cost = 0
    for i in range(len(policy) - 1):
        total_cost += distance(cities[policy[i]], cities[policy[i + 1]])
    return total_cost

total_cost = evaluate_policy(policy, cities)

print("최적 경로:", policy)
print("Total cost:", total_cost)