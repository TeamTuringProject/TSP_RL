import numpy as np
from deap import tools
import random
import csv

# 유클리드 거리 계산 함수
def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

# csv 파일 읽어서 cities에 load하는 함수
cities = []
def load_cities(filename):
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            cities.append([float(coord) for coord in row])
    return cities

#NN
def initialize_population(num_cities, population_size=50):
    population = []
    for _ in range(population_size-1):
        # Start with a list of all city indices
        solution = list(range(num_cities))
        # Remove the first city index to keep it fixed
        first_city = solution.pop(0)
        # Shuffle the remaining cities
        random.shuffle(solution)
        # Prepend the fixed first city back to the start
        solution.insert(0, first_city)
        population.append(solution)
    return population

# 남아있는 도시 배열에서 query_point와 가장 가까운 도시를 찾는 함수 사본
def nearest_neighbor_search_all(cities, remain_cities_index, query_point):
    min_cost = float('inf')
    nearest_city_index = None

    for idx in remain_cities_index:
        cost = distance(cities[idx], query_point)
        if cost < min_cost:
            min_cost = cost
            nearest_city_index = idx

    return nearest_city_index


#NN서치
def iterative_nn_search(cities):
  solution = []

  #남은 도시의 인덱스를 가진 배열 생성
  remain_cities_index = list(range(len(cities)))

  # 랜덤 시작 도시 설정
  start_city_index = random.randint(0, len(cities) - 1)
  solution.append(start_city_index)
  query_point = cities[start_city_index]

  #remain_cities_index에서 시작 도시 인덱스 삭제
  remain_cities_index.remove(start_city_index)

  #NN 서치
  for i in range(len(cities) - 1):
    nearest_city_index = nearest_neighbor_search_all(cities, remain_cities_index, query_point)
    solution.append(nearest_city_index)
    query_point = cities[nearest_city_index]
    remain_cities_index.remove(nearest_city_index)

  return solution

# initilalize population by nn search
def initialize_population_nn(cities, population_size):
    population = []
    solution = None

    # make solutions (population_size) times
    for _ in range(population_size):
        solution = iterative_nn_search(cities)
        population.append(solution)
    return population

# Function to evaluate the cost of each solution in the populations
def evaluate_populations(populations, cities):
    scores = []
    for sol in populations:
        sol.append(sol[len(sol) - 1])
        total_cost = 0
        for idx in range(len(sol) - 1):
            pos_city_1 = cities[sol[idx]]
            pos_city_2 = cities[sol[idx+1]]
            total_cost += distance(pos_city_1, pos_city_2)
        scores.append(total_cost)
        sol.pop()
    return scores

#SELECTION
def selection(population, scores, num_best, population_best_num, population_rest_num):
    # num_best 개수만큼 좋은 해들을 뽑음
    best_solutions = select_best(population, scores, num_best, population_best_num)
    # num_rest 개수만큼 전체 해들을 뽑음
    rest_solutions = select_rest(population, scores, num_best, population_rest_num)
    # num_best+num_rest 전체에서 2개를 랜덤으로 뽑음
    selected_solutions = best_solutions + rest_solutions
    return selected_solutions
def select_best(population, scores, num_best, population_best_num):
    sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0])]
    return random.sample(sorted_population[:num_best], population_best_num)
# best를 제외한 값들을 제외한 값들중에 랜덤으로 선택
def select_rest(population, scores, num_best, population_rest_num):
    sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0])]
    return random.sample(sorted_population[num_best :], population_rest_num)

#CROSSOVER
def order_crossover(parent_1, parent_2):
    init_node = random.randint(0, num_cities - 1)
    before_end_node = random.randint(init_node, num_cities)
    temp = parent_1[init_node:before_end_node]
    front = parent_2[before_end_node:]
    back = parent_2[:before_end_node]
    # temp에 있는 값을 front 및 back에서 제외하고 새로운 리스트 생성
    front = [val for val in front if val not in temp]
    back = [val for val in back if val not in temp]

    child = front + temp + back

    return child

#MUTATION
# 부모 자식간의 distance 계산
def find_distance(parent, child):
    distance = 0
    child_index = 0

   # 자식 해의 첫 번째 도시가 부모 해의 어디에 있는지 찾음
    for i in range(len(parent)):
        if parent[0] == child[i]:
            child_index = i
            break

    # 부모 해와 자식 해 간의 거리 계산
    for i in range(len(parent)):
        # 자식 해의 인덱스가 부모 해의 길이를 초과하면 처음으로 돌아감
        if child_index >= len(parent)-1:
            child_index = 0

        # 부모 해와 자식 해의 도시가 일치하는지 확인하여 거리 계산
        if parent[i] != child[child_index]:
            distance += 1

        child_index += 1
    return distance

def get_diff_val(parent1, parent2, child):
    distance_parent1 = find_distance(parent1, child)
    distance_parent2 = find_distance(parent2, child)
    return min(distance_parent1, distance_parent2)

# mutation rate 설정
mutation_rate1 = 0.03
mutation_rate2 = 0.01

# mutation함수
def mutate(parent_1, parent_2, solution):
    #유전 거리 계산
    diff_val = get_diff_val(parent_1, parent_2, solution)

   #유사도가 높을 수록 더 큰 rate 할당
    if diff_val < 990: #유사도가 높음
        mutation_rate = mutation_rate1
    else:
        mutation_rate = mutation_rate2

    # 랜덤 구간 뒤집기
    if random.random() < mutation_rate:
        start = random.randint(0, len(solution) - 1)
        end = random.randint(0, len(solution) - 1)
        # 구간의 시작과 끝을 올바르게 정렬
        if start > end:
            start, end = end, start
        # 구간을 뒤집기
        solution[start:end + 1] = solution[start:end + 1][::-1]

    #랜덤 swap
    elif random.random() < mutation_rate:
        for i in range(len(solution)):
          j = random.randint(1, len(solution) - 1)
          if i == 0:
            continue
          solution[i], solution[j] = solution[j], solution[i]

    return solution

#REPLACEMENT
def get_bad_scores(scores):
    # temp_scores의 값들을 정렬하고 원래의 위치를 기억할 수 있도록 enumerate 사용
    indexed_scores = list(enumerate(scores))
    # 점수에 따라 내림차순 정렬
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    # 가장 값이 큰 5개의 위치 추출
    bad_scores = [indexed_scores[i][0] for i in range(min(5, len(scores)))]
    return bad_scores

def replace(parent1_idx, parent2_idx, child, scores, population):
    replaced_population = population
    # 부모 경로의 점수 추출
    parent1_score = scores[parent1_idx]
    parent2_score = scores[parent2_idx]
    # 자식 경로의 점수 추출
    total_cost = 0
    for idx in range(len(child) - 1):
        pos_city_1 = cities[child[idx]]
        pos_city_2 = cities[child[idx + 1]]
        total_cost += distance(pos_city_1, pos_city_2)
    total_cost += distance(cities[child[-1]], cities[child[0]])
    child_score = total_cost

    if child_score < parent1_score or child_score < parent2_score:
        # 자식해가 부모해 둘 중 하나보다 품질이 좋으면, 안 좋은 쪽을 대체
        if parent1_score > parent2_score:
            scores[parent1_idx] = child_score
            replaced_population[parent1_idx] = child
        else:
            scores[parent2_idx] = child_score
            replaced_population[parent2_idx] = child
    else:
        # 자식해가 부모해 둘 다보다 품질이 안 좋으면, 가장 안 좋은 품질의 해 5개 중 무작위로 하나 선택해 대체
        bad_scores = get_bad_scores(scores)
        n = random.choice(bad_scores)
        replaced_population[n] = child
        scores[n] = child_score
    return replaced_population

# 1000개 도시 csv파일
cities = load_cities('./2024_AI_TSP.csv')
num_cities = len(cities)

# Initialize population
population_size = 50
POPULATION_BEST_NUM = 8
POPULATION_REST_NUM = 2

#init_population_nn = initialize_population_nn(cities, 20)
population = initialize_population_nn(cities, 20)

# Evaluate initial populations
scores = evaluate_populations(population, cities)

# Genetic algorithm parameters
generations = 100
num_best = 8
solutions_gen = []

# Genetic algorithm main loop
best_score = float('inf')
best_solution = None
for generation in range(generations+1):
    new_population = population
    i = 0
    selected_solutions = selection(population, scores, num_best, POPULATION_BEST_NUM, POPULATION_REST_NUM)
    parent1_idx = []
    parent2_idx = []
    child_list = []

    while i < len(population):
        #selection
        parent1, parent2 = random.sample(selected_solutions, 2)
        parent1_idx.append(population.index(parent1))
        parent2_idx.append(population.index(parent2))
        #crossover
        child = order_crossover(parent1, parent2)
        #mutation
        child = mutate(parent1, parent2, child)
        child_list.append(child)
        i += 1

    #replace
    for idx in range(len(parent1_idx)):
      new_population = replace(parent1_idx[idx], parent2_idx[idx], child_list[idx], scores, new_population)

    #replaced_population을 다음 population으로 지정
    population = new_population

#=========================================================================================================   
# 경로 빈도 수를 세기 위한 테이블
frequency_table = np.zeros((num_cities, num_cities))

# 빈도 테이블 업데이트 (50 개체)
for ind in population:
    for i in range(len(ind) - 1):
        frequency_table[ind[i]][ind[i + 1]] += 1
    frequency_table[ind[-1]][ind[0]] += 1 # 마지막 노드-> 시작 노드로 돌아옴

# 거리 테이블 생성
distance_table = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        distance_table[i][j] = distance_table[j][i] = distance(cities[i], cities[j])
    
# value table 초기화
value_table = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if distance_table[i][j] != 0:  # 0으로 나누는 것을 방지
            value_table[i][j] = frequency_table[i][j] / distance_table[i][j] # 빈도수/거리

# value table에서 Policy 추출
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



cities = load_cities('./2024_AI_TSP.csv')
policy = extract_policy(value_table)

# policy 평가
def evaluate_policy(policy, cities):
    total_cost = 0
    for i in range(len(policy) - 1):
        total_cost += distance(cities[policy[i]], cities[policy[i + 1]])
    return total_cost

total_cost = evaluate_policy(policy, cities)


print("최적 경로:", policy)
print("Total cost:", total_cost)

print("최적경로 길이: ", len(policy))

print("빈도수 배열:")
print(frequency_table)
print("\n거리 배열:")
print(distance_table)

## 검증?
# output_file = f'solution.csv'
# with open(output_file, 'w') as f:
#   for number in best_solution:
#     f.write(f"{number}\n")

# def check_unique_integers(arr):
#     n = len(arr)
#     expected_set = set(range(n))
#     arr_set = set(arr)

#     return arr_set == expected_set