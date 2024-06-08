import numpy as np
import torch
import csv

# 유클리드 거리 계산 함수
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

# 거리 테이블 생성 함수
def create_distance_table(cities):
    num_cities = len(cities)
    distance_table = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance_table[i][j] = distance_table[j][i] = distance(cities[i], cities[j])
    return distance_table

# Load cities and create distance table
cities = load_cities('./2024_AI_TSP.csv')
distance_table = create_distance_table(cities)
num_cities = len(cities)

# 유전 알고리즘 관련 함수들

def nearest_neighbor_search_all(cities, remain_cities_index, query_point):
    min_cost = float('inf')
    nearest_city_index = None
    for idx in remain_cities_index:
        cost = distance(cities[idx], query_point)
        if cost < min_cost:
            min_cost = cost
            nearest_city_index = idx
    return nearest_city_index

def iterative_nn_search(cities):
    solution = []
    remain_cities_index = list(range(len(cities)))
    start_city_index = random.randint(0, len(cities) - 1)
    solution.append(start_city_index)
    query_point = cities[start_city_index]
    remain_cities_index.remove(start_city_index)
    for _ in range(len(cities) - 1):
        nearest_city_index = nearest_neighbor_search_all(cities, remain_cities_index, query_point)
        solution.append(nearest_city_index)
        query_point = cities[nearest_city_index]
        remain_cities_index.remove(nearest_city_index)
    return solution

def initialize_population_nn(cities, population_size):
    population = []
    for _ in range(population_size):
        solution = iterative_nn_search(cities)
        population.append(solution)
    return population

def evaluate_populations(populations, cities):
    scores = []
    for sol in populations:
        total_cost = 0
        for idx in range(len(sol) - 1):
            pos_city_1 = cities[sol[idx]]
            pos_city_2 = cities[sol[idx+1]]
            total_cost += distance(pos_city_1, pos_city_2)
        total_cost += distance(cities[sol[-1]], cities[sol[0]])
        scores.append(total_cost)
    return scores

def selection(population, scores, num_best, population_best_num, population_rest_num):
    best_solutions = select_best(population, scores, num_best, population_best_num)
    rest_solutions = select_rest(population, scores, num_best, population_rest_num)
    return best_solutions + rest_solutions

def select_best(population, scores, num_best, population_best_num):
    sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0])]
    return random.sample(sorted_population[:num_best], population_best_num)

def select_rest(population, scores, num_best, population_rest_num):
    sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0])]
    return random.sample(sorted_population[num_best:], population_rest_num)

def order_crossover(parent_1, parent_2):
    init_node = random.randint(0, num_cities - 1)
    before_end_node = random.randint(init_node, num_cities)
    temp = parent_1[init_node:before_end_node]
    front = parent_2[before_end_node:]
    back = parent_2[:before_end_node]
    front = [val for val in front if val not in temp]
    back = [val for val in back if val not in temp]
    return front + temp + back

def find_distance(parent, child):
    distance = 0
    child_index = 0
    for i in range(len(parent)):
        if parent[0] == child[i]:
            child_index = i
            break
    for i in range(len(parent)):
        if child_index >= len(parent)-1:
            child_index = 0
        if parent[i] != child[child_index]:
            distance += 1
        child_index += 1
    return distance

def get_diff_val(parent1, parent2, child):
    distance_parent1 = find_distance(parent1, child)
    distance_parent2 = find_distance(parent2, child)
    return min(distance_parent1, distance_parent2)

def mutate(parent_1, parent_2, solution):
    diff_val = get_diff_val(parent_1, parent_2, solution)
    mutation_rate1 = 0.03
    mutation_rate2 = 0.01
    mutation_rate = mutation_rate1 if diff_val < 990 else mutation_rate2

    if random.random() < mutation_rate:
        start = random.randint(0, len(solution) - 1)
        end = random.randint(0, len(solution) - 1)
        if start > end:
            start, end = end, start
        solution[start:end + 1] = solution[start:end + 1][::-1]
    elif random.random() < mutation_rate:
        for i in range(len(solution)):
            j = random.randint(1, len(solution) - 1)
            if i == 0:
                continue
            solution[i], solution[j] = solution[j], solution[i]
    return solution

def get_bad_scores(scores):
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    return [indexed_scores[i][0] for i in range(min(5, len(scores)))]

def replace(parent1_idx, parent2_idx, child, scores, population):
    replaced_population = population
    parent1_score = scores[parent1_idx]
    parent2_score = scores[parent2_idx]
    total_cost = 0
    for idx in range(len(child) - 1):
        pos_city_1 = cities[child[idx]]
        pos_city_2 = cities[child[idx + 1]]
        total_cost += distance(pos_city_1, pos_city_2)
    total_cost += distance(cities[child[-1]], cities[child[0]])
    child_score = total_cost

    if child_score < parent1_score or child_score < parent2_score:
        if parent1_score > parent2_score:
            scores[parent1_idx] = child_score
            replaced_population[parent1_idx] = child
        else:
            scores[parent2_idx] = child_score
            replaced_population[parent2_idx] = child
    else:
        bad_scores = get_bad_scores(scores)
        n = random.choice(bad_scores)
        replaced_population[n] = child
        scores[n] = child_score
    return replaced_population

# GA를 이용한 초기 population 설정 및 평가
population_size = 50
POPULATION_BEST_NUM = 8
POPULATION_REST_NUM = 2
population = initialize_population_nn(cities, population_size)
scores = evaluate_populations(population, cities)
generations = 100
num_best = 8

# Genetic Algorithm Main Loop
best_score = float('inf')
best_solution = None
for generation in range(generations + 1):
    new_population = population
    i = 0
    selected_solutions = selection(population, scores, num_best, POPULATION_BEST_NUM, POPULATION_REST_NUM)
    parent1_idx = []
    parent2_idx = []
    child_list = []

    while i < len(population):
        parent1, parent2 = random.sample(selected_solutions, 2)
        parent1_idx.append(population.index(parent1))
        parent2_idx.append(population.index(parent2))
        child = order_crossover(parent1, parent2)
        child = mutate(parent1, parent2, child)
        child_list.append(child)
        i += 1

    for idx in range(len(parent1_idx)):
        new_population = replace(parent1_idx[idx], parent2_idx[idx], child_list[idx], scores, new_population)

    population = new_population

# 빈도 테이블 생성
frequency_table = np.zeros((num_cities, num_cities))
for ind in population:
    for i in range(len(ind) - 1):
        frequency_table[ind[i]][ind[i + 1]] += 1
    frequency_table[ind[-1]][ind[0]] += 1

# value 테이블 생성 함수
def create_value_table(frequency_table, distance_table):
    num_cities = len(frequency_table)
    value_table = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if distance_table[i][j] != 0:
                value_table[i][j] = frequency_table[i][j] / distance_table[i][j]
    return value_table

# value 테이블 생성
value_table = create_value_table(frequency_table, distance_table)

# 정책 추출 함수
def extract_policy(value_table):
    num_cities = len(value_table)
    visited = [False] * num_cities
    policy = []
    current_city = 0
    policy.append(current_city)
    visited[current_city] = True

    while len(policy) < num_cities:
        next_city = np.argmax([value_table[current_city][j] if not visited[j] else float('-inf') for j in range(num_cities)])
        policy.append(next_city)
        visited[next_city] = True
        current_city = next_city
    
    policy.append(policy[0])
    return policy

# 정책 추출 및 평가
policy = extract_policy(value_table)
total_cost = evaluate_policy(policy, cities)

print("최적 경로:", policy)
print("Total cost:", total_cost)
print("빈도수 배열:")
print(frequency_table)
print("\n거리 배열:")
print(distance_table)

# 상태 노드 정보 생성 함수
def create_state_node_info(cities, policy):
    num_cities = len(cities)
    state_node_info = []

    for city in range(num_cities):
        visited = int(city in policy)
        first_node = int(city == policy[0])
        last_node = int(city == policy[-1])
        x, y = cities[city]
        state_node_info.append([visited, first_node, last_node, x, y])
    
    return torch.tensor(state_node_info, dtype=torch.float)

# 데이터셋 생성
batch_size = 32
num_batches = 100  # 학습을 위한 배치 수

xv_list = [create_state_node_info(cities, policy) for _ in range(batch_size * num_batches)]
xv = torch.stack(xv_list)
Ws = torch.tensor(distance_table, dtype=torch.float).unsqueeze(0).repeat(batch_size * num_batches, 1, 1)

# 행동 및 목표 Q값 생성 (랜덤 예시)
actions = torch.randint(0, num_cities, (batch_size * num_batches,))
targets = torch.randn(batch_size * num_batches)

#모델 학습
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QNet(nn.Module):
    def __init__(self, emb_dim, T=4, node_dim=5):
        super(QNet, self).__init__()
        self.emb_dim = emb_dim
        self.T = T
        self.node_dim = node_dim      

        self.theta1 = nn.Linear(self.node_dim, self.emb_dim, True)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta4 = nn.Linear(1, self.emb_dim, True)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True)
        
        self.layer = nn.Linear(self.emb_dim, self.emb_dim, True)
        
    def forward(self, xv, Ws):
        num_nodes = xv.shape[1]   # 전체 도시의 수
        batch_size = xv.shape[0]  # batch size
        
        conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=device)

        s1 = self.theta1(xv)
        s1 = self.layer(F.relu(s1))

        s3_0 = Ws.unsqueeze(3)
        s3_1 = F.relu(self.theta4(s3_0))
        s3_2 = torch.sum(s3_1, dim=1)
        s3 = self.theta3(s3_2)

        for _ in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))
            mu = F.relu(s1 + s2 + s3)

        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        local_action = self.theta7(mu)
            
        out = F.relu(torch.cat([global_state, local_action], dim=2))
        return self.theta5(out).squeeze(dim=2)

class QTrainer:
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.MSELoss()

    def predict(self, state_tsr, W):
        with torch.no_grad():
            estimated_q_value = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_q_value[0]
                
    def get_best_action(self, state_tsr, state):
        W = state.W
        estimated_q_value = self.predict(state_tsr, W)  
        sorted_q_value_idx = estimated_q_value.argsort(descending=True)
        
        solution = state.partial_solution
        already_in = set(solution)
        for idx in sorted_q_value_idx.tolist():
            if (len(solution) == 0 or W[solution[-1], idx] > 0) and idx not in already_in:
                return idx, estimated_q_value[idx].item()

    def batch_update(self, states_tsrs, Ws, actions, targets):
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)
        self.optimizer.zero_grad()
        
        estimated_q_value = self.model(xv, Ws_tsr)[range(len(actions)), actions]
        
        loss = self.loss_fn(estimated_q_value, torch.tensor(targets, device=device))
        loss_val = loss.item()
        
        loss.backward()
        self.optimizer.step()        
        self.lr_scheduler.step()
        
        return loss_val

# 하이퍼파라미터 설정
emb_dim = 128
T = 4
node_dim = 5
model = QNet(emb_dim=emb_dim, T=T, node_dim=node_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
trainer = QTrainer(model, optimizer, lr_scheduler)

# 학습 루프
num_epochs = 1000

for epoch in range(num_epochs):
    for batch in range(num_batches):
        batch_xv = xv[batch * batch_size:(batch + 1) * batch_size]
        batch_Ws = Ws[batch * batch_size:(batch + 1) * batch_size]
        batch_actions = actions[batch * batch_size:(batch + 1) * batch_size]
        batch_targets = targets[batch * batch_size:(batch + 1) * batch_size]

        loss = trainer.batch_update(states_tsrs=batch_xv, Ws=batch_Ws, actions=batch_actions, targets=batch_targets)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    if (epoch + 1) % 100 == 0:
        print(f"Learning rate: {lr_scheduler.get_last_lr()[0]}")
