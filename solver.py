import json
import sys
import os
import heapq
import random
from collections import defaultdict, Counter

class WarehouseLocationSolver:
    def __init__(self, json_data):
        self.parse_json(json_data)
        # Cache incompatibilities for faster lookup
        self.incompatibility_map = defaultdict(set)
        for s1, s2 in self.incompatibilities:
            self.incompatibility_map[s1].add(s2)
            self.incompatibility_map[s2].add(s1)
        
        self.efficiencies = self._precompute_efficiencies()
        
    def parse_json(self, json_data):
        """parse the json input data"""
        self.num_warehouses = len(json_data['warehouses'])
        self.num_stores = len(json_data['stores'])
        
        # parse warehouses
        self.capacities = []
        self.fixed_costs = []
        self.warehouse_ids = []
        for wh in json_data['warehouses']:
            self.warehouse_ids.append(wh['id'])
            self.capacities.append(wh['capacity'])
            self.fixed_costs.append(wh['fixed_cost'])
        
        # parse stores and supply costs
        self.goods = []
        self.store_ids = []
        self.supply_costs = []
        for store in json_data['stores']:
            self.store_ids.append(store['id'])
            self.goods.append(store['demand'])
            
            # create supply cost row for this store
            cost_row = []
            for wh_id in self.warehouse_ids:
                cost_key = f"warehouse_{wh_id}"
                cost_row.append(store['supply_costs'][cost_key])
            self.supply_costs.append(cost_row)
        
        # parse incompatibilities
        self.incompatibilities = []
        for inc in json_data.get('incompatibilities', []):
            s1 = self.store_ids.index(inc['store_1'])
            s2 = self.store_ids.index(inc['store_2'])
            self.incompatibilities.append((s1, s2))
    
    def _precompute_efficiencies(self):
        """Precompute efficiency scores for all store-warehouse pairs"""
        efficiencies = []
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                efficiency = -1 * self.supply_costs[s][w]
                efficiencies.append((efficiency, s, w))
        return sorted(efficiencies)
    
    def calculate_initial_solution(self):
        """Create initial solution using improved construction heuristic"""
        solution = [[0 for _ in range(self.num_warehouses)] for _ in range(self.num_stores)]
        remaining_demand = self.goods.copy()
        warehouse_used_capacity = [0] * self.num_warehouses
        open_warehouses = set()
        
        store_priority = [(demand, idx) for idx, demand in enumerate(self.goods)]
        store_priority.sort(reverse=True)
        store_order = [s for _, s in store_priority]
        
        for s in store_order:
            if remaining_demand[s] == 0:
                continue
                
            for _, store, w in self.efficiencies:
                if store != s or warehouse_used_capacity[w] + remaining_demand[s] > self.capacities[w]:
                    continue
                
                if any(other_s in self.incompatibility_map[s] for other_s in range(self.num_stores) 
                       if solution[other_s][w] > 0):
                    continue
                
                solution[s][w] = remaining_demand[s]
                warehouse_used_capacity[w] += remaining_demand[s]
                open_warehouses.add(w)
                remaining_demand[s] = 0
                break
        
        for s in range(self.num_stores):
            if remaining_demand[s] == 0:
                continue
                
            # sort warehouses by cost for this store
            warehouse_costs = [(self.supply_costs[s][w], w) for w in range(self.num_warehouses)]
            warehouse_costs.sort()
            
            for _, w in warehouse_costs:
                if remaining_demand[s] == 0:
                    break
                    
                # check capacity
                available_capacity = self.capacities[w] - warehouse_used_capacity[w]
                if available_capacity <= 0:
                    continue
                    
                # check incompatibilities
                if any(other_s in self.incompatibility_map[s] for other_s in range(self.num_stores) 
                       if solution[other_s][w] > 0):
                    continue
                
                # make partial assignment
                assign_amount = min(remaining_demand[s], available_capacity)
                solution[s][w] = assign_amount
                warehouse_used_capacity[w] += assign_amount
                open_warehouses.add(w)
                remaining_demand[s] -= assign_amount
        
        return solution
    
    def calculate_cost(self, solution):
        """Calculate total cost of a solution with caching for open warehouses"""
        supply_cost = 0
        open_warehouses = set()
        
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                if solution[s][w] > 0:
                    supply_cost += solution[s][w] * self.supply_costs[s][w]
                    open_warehouses.add(w)
        
        opening_cost = sum(self.fixed_costs[w] for w in open_warehouses)
        return supply_cost + opening_cost, open_warehouses
    
    def is_feasible(self, solution):
        """Check if solution satisfies all constraints"""
        # check all stores have their demand met
        for s in range(self.num_stores):
            if sum(solution[s]) != self.goods[s]:
                return False
        
        # check warehouse capacities
        warehouse_usage = [0] * self.num_warehouses
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                warehouse_usage[w] += solution[s][w]
                
        for w in range(self.num_warehouses):
            if warehouse_usage[w] > self.capacities[w]:
                return False
        
        # check incompatibilities using the map
        for w in range(self.num_warehouses):
            stores_at_warehouse = [s for s in range(self.num_stores) if solution[s][w] > 0]
            for s1 in stores_at_warehouse:
                if any(s2 in self.incompatibility_map[s1] for s2 in stores_at_warehouse if s2 != s1):
                    return False
        
        return True
    
    def improved_local_search(self, initial_solution, max_iter=1000, max_no_improve=100):
        """Improved local search with random neighborhood selection and early stopping"""
        current_solution = [row.copy() for row in initial_solution]
        current_cost, open_warehouses = self.calculate_cost(current_solution)
        best_cost = current_cost
        best_solution = [row.copy() for row in current_solution]
        
        no_improve_counter = 0
        iteration = 0
        
        warehouse_usage = [0] * self.num_warehouses
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                warehouse_usage[w] += current_solution[s][w]
        
        stores_at_warehouse = [set(s for s in range(self.num_stores) if current_solution[s][w] > 0) 
                               for w in range(self.num_warehouses)]
        
        while iteration < max_iter and no_improve_counter < max_no_improve:
            iteration += 1
            improved = False
            
            stores_with_demand = [s for s in range(self.num_stores) if sum(current_solution[s]) > 0]
            if not stores_with_demand:
                break
                
            s = random.choice(stores_with_demand)
            
            current_assignments = [(w, current_solution[s][w]) for w in range(self.num_warehouses) if current_solution[s][w] > 0]
            if not current_assignments:
                continue
                
            w_from, amount = random.choice(current_assignments)
            
            potential_targets = [w for w in range(self.num_warehouses) if w != w_from]
            random.shuffle(potential_targets)
            
            for w_to in potential_targets:
                if warehouse_usage[w_to] + amount > self.capacities[w_to]:
                    continue
                    
                if any(other_s in self.incompatibility_map[s] for other_s in stores_at_warehouse[w_to]):
                    continue
                    
                cost_diff = amount * (self.supply_costs[s][w_to] - self.supply_costs[s][w_from])
                
                stores_from_after = stores_at_warehouse[w_from].copy()
                stores_from_after.remove(s)
                
                stores_to_after = stores_at_warehouse[w_to].copy()
                stores_to_after.add(s)
                
                if not stores_from_after and w_from in open_warehouses:
                    cost_diff -= self.fixed_costs[w_from] 
                    
                if not stores_at_warehouse[w_to] and w_to not in open_warehouses:
                    cost_diff += self.fixed_costs[w_to]
                
                if cost_diff < 0 or (random.random() < 0.01 and iteration > max_iter // 2):
                    current_solution[s][w_from] -= amount
                    current_solution[s][w_to] += amount
                    
                    warehouse_usage[w_from] -= amount
                    warehouse_usage[w_to] += amount
                    
                    if current_solution[s][w_from] == 0:
                        stores_at_warehouse[w_from].remove(s)
                    stores_at_warehouse[w_to].add(s)
                    
                    if not stores_at_warehouse[w_from]:
                        open_warehouses.discard(w_from)
                    if stores_at_warehouse[w_to]:
                        open_warehouses.add(w_to)
                    
                    current_cost += cost_diff
                    improved = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_solution = [row.copy() for row in current_solution]
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                        
                    break
            
            if not improved:
                no_improve_counter += 1
        
        return best_solution
    
    def solve(self):
        # generate initial solution
        solution = self.calculate_initial_solution()
        
        if not self.is_feasible(solution):
            # if initial solution isn't feasible, try a different approach
            solution = [[0 for _ in range(self.num_warehouses)] for _ in range(self.num_stores)]
            
            # simple approach: assign each store to warehouses with sufficient capacity
            # and no incompatibilities
            remaining_demand = self.goods.copy()
            for s in range(self.num_stores):
                # sort warehouses by cost
                warehouses = [(self.supply_costs[s][w], w) for w in range(self.num_warehouses)]
                warehouses.sort()
                
                for _, w in warehouses:
                    if remaining_demand[s] == 0:
                        break
                        
                    # check capacity
                    warehouse_used = sum(solution[other_s][w] for other_s in range(self.num_stores))
                    available = self.capacities[w] - warehouse_used
                    
                    # check incompatibilities
                    incompatible = False
                    for other_s in range(self.num_stores):
                        if solution[other_s][w] > 0 and (s in self.incompatibility_map[other_s]):
                            incompatible = True
                            break
                    
                    if incompatible or available <= 0:
                        continue
                        
                    # assign
                    assign_amount = min(remaining_demand[s], available)
                    solution[s][w] = assign_amount
                    remaining_demand[s] -= assign_amount
        
        # apply local search with different parameters for diversification
        solution = self.improved_local_search(solution, max_iter=1000, max_no_improve=100)
        
        # apply focused search on promising solution
        solution = self.improved_local_search(solution, max_iter=500, max_no_improve=50)
        
        return solution
    
    def format_matrix_solution(self, solution):
        """format solution as a matrix of supplied quantities"""
        output = "["
        for s in range(self.num_stores):
            row = "(" + ",".join(map(str, solution[s])) + ")"
            output += row + "\n"
        output = output.rstrip() + "]"
        return output
    
    def format_list_solution(self, solution):
        """format solution as a list of triples"""
        triples = []
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                if solution[s][w] > 0:
                    triples.append(f"({self.store_ids[s]},{self.warehouse_ids[w]},{solution[s][w]})")
        return "{" + ",\n".join(triples) + "}"
    
    def write_solution_file(self, solution, output_file, format_type='list'):
        """write the solution to a file in the specified format"""
        with open(output_file, 'w') as f:
            if format_type == 'matrix':
                f.write(self.format_matrix_solution(solution))
            else:
                f.write(self.format_list_solution(solution))
        
        total_cost, open_warehouses = self.calculate_cost(solution)
        supply_cost = 0
        
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                if solution[s][w] > 0:
                    supply_cost += solution[s][w] * self.supply_costs[s][w]
        
        opening_cost = sum(self.fixed_costs[w] for w in open_warehouses)
        
        print(f"solution written to {output_file}")
        print(f"total cost: {total_cost} = {supply_cost} (supply) + {opening_cost} (opening)")
        print(f"open warehouses: {sorted([self.warehouse_ids[w] for w in open_warehouses])}")

def main():
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} instance.json", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        with open(input_file) as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"error: file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"error: invalid json in file '{input_file}'", file=sys.stderr)
        sys.exit(1)
    
    os.makedirs('solutions', exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"solutions/solution-{base_name}.txt"
    
    solver = WarehouseLocationSolver(json_data)
    solution = solver.solve()
    
    if solver.is_feasible(solution):
        solver.write_solution_file(solution, output_file, format_type='list')
    else:
        print("no feasible solution found", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
