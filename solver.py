import json
import sys
import os
from collections import defaultdict

class WarehouseLocationSolver:
    def __init__(self, json_data):
        self.parse_json(json_data)
        
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
    
    def greedy_construction(self):
        """greedy construction heuristic to build initial solution"""
        solution = [[0 for _ in range(self.num_warehouses)] for _ in range(self.num_stores)]
        open_warehouses = set()
        remaining_demand = self.goods.copy()
        warehouse_used_capacity = [0] * self.num_warehouses
        
        # calculate cost of assigning each store to each warehouse
        assignment_costs = []
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                cost = self.supply_costs[s][w] * self.goods[s]
                if w not in open_warehouses:
                    cost += self.fixed_costs[w]
                assignment_costs.append((cost, s, w))
        
        # sort assignments by cost
        assignment_costs.sort()
        
        # make assignments greedily
        for cost, s, w in assignment_costs:
            if remaining_demand[s] == 0:
                continue
                
            # check capacity and incompatibilities
            if warehouse_used_capacity[w] + remaining_demand[s] > self.capacities[w]:
                continue
                
            # check if this assignment would violate any incompatibilities
            valid = True
            for other_s in range(self.num_stores):
                if solution[other_s][w] > 0:
                    if (s, other_s) in self.incompatibilities or (other_s, s) in self.incompatibilities:
                        valid = False
                        break
            if not valid:
                continue
                
            # make the assignment
            assign_amount = min(remaining_demand[s], self.capacities[w] - warehouse_used_capacity[w])
            solution[s][w] = assign_amount
            remaining_demand[s] -= assign_amount
            warehouse_used_capacity[w] += assign_amount
            open_warehouses.add(w)
            
            if remaining_demand[s] == 0:
                continue
        
        # if any stores still have demand, try to satisfy it
        for s in range(self.num_stores):
            if remaining_demand[s] > 0:
                # try to find any warehouse that can take the remaining demand
                for w in range(self.num_warehouses):
                    if warehouse_used_capacity[w] + remaining_demand[s] <= self.capacities[w]:
                        # check incompatibilities
                        valid = True
                        for other_s in range(self.num_stores):
                            if solution[other_s][w] > 0:
                                if (s, other_s) in self.incompatibilities or (other_s, s) in self.incompatibilities:
                                    valid = False
                                    break
                        if valid:
                            solution[s][w] += remaining_demand[s]
                            warehouse_used_capacity[w] += remaining_demand[s]
                            remaining_demand[s] = 0
                            open_warehouses.add(w)
                            break
        
        return solution
    
    def calculate_cost(self, solution):
        """calculate total cost of a solution"""
        supply_cost = 0
        open_warehouses = set()
        
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                if solution[s][w] > 0:
                    supply_cost += solution[s][w] * self.supply_costs[s][w]
                    open_warehouses.add(w)
        
        opening_cost = sum(self.fixed_costs[w] for w in open_warehouses)
        return supply_cost + opening_cost
    
    def is_feasible(self, solution):
        """check if solution satisfies all constraints"""
        # check all stores have their demand met
        for s in range(self.num_stores):
            if sum(solution[s]) != self.goods[s]:
                return False
        
        # check warehouse capacities
        for w in range(self.num_warehouses):
            total = sum(solution[s][w] for s in range(self.num_stores))
            if total > self.capacities[w]:
                return False
        
        # check incompatibilities
        for w in range(self.num_warehouses):
            stores_served = [s for s in range(self.num_stores) if solution[s][w] > 0]
            for i in range(len(stores_served)):
                for j in range(i+1, len(stores_served)):
                    s1, s2 = stores_served[i], stores_served[j]
                    if (s1, s2) in self.incompatibilities or (s2, s1) in self.incompatibilities:
                        return False
        
        return True
    
    def local_search(self, initial_solution, max_iter=1000):
        """improve solution using local search with interchange moves"""
        current_solution = [row.copy() for row in initial_solution]
        current_cost = self.calculate_cost(current_solution)
        
        for _ in range(max_iter):
            improved = False
            
            # try all possible single moves
            for s in range(self.num_stores):
                for w_from in range(self.num_warehouses):
                    if current_solution[s][w_from] == 0:
                        continue
                        
                    for w_to in range(self.num_warehouses):
                        if w_from == w_to:
                            continue
                            
                        # check if move is possible
                        if sum(current_solution[s2][w_to] for s2 in range(self.num_stores)) + current_solution[s][w_from] > self.capacities[w_to]:
                            continue
                            
                        # check incompatibilities
                        valid = True
                        for other_s in range(self.num_stores):
                            if current_solution[other_s][w_to] > 0:
                                if (s, other_s) in self.incompatibilities or (other_s, s) in self.incompatibilities:
                                    valid = False
                                    break
                        if not valid:
                            continue
                            
                        # create neighbor solution
                        neighbor = [row.copy() for row in current_solution]
                        amount = neighbor[s][w_from]
                        neighbor[s][w_from] = 0
                        neighbor[s][w_to] = amount
                        
                        # calculate cost difference
                        cost_diff = amount * (self.supply_costs[s][w_to] - self.supply_costs[s][w_from])
                        
                        # check if we need to open or close warehouses
                        from_open = any(neighbor[s2][w_from] > 0 for s2 in range(self.num_stores))
                        to_open = any(neighbor[s2][w_to] > 0 for s2 in range(self.num_stores))
                        
                        if not from_open and any(current_solution[s2][w_from] > 0 for s2 in range(self.num_stores)):
                            cost_diff -= self.fixed_costs[w_from]
                        if to_open and not any(current_solution[s2][w_to] > 0 for s2 in range(self.num_stores)):
                            cost_diff += self.fixed_costs[w_to]
                            
                        if cost_diff < 0:  # improvement found
                            current_solution = neighbor
                            current_cost += cost_diff
                            improved = True
                            break
                            
                    if improved:
                        break
                if improved:
                    break
                    
            if not improved:
                break
                
        return current_solution
    
    def solve(self):
        # greedy construction
        solution = self.greedy_construction()
        
        if not self.is_feasible(solution):
            print("warning: initial greedy solution is not feasible", file=sys.stderr)
        
        # local search
        solution = self.local_search(solution)
        
        # additional refinement
        solution = self.local_search(solution, max_iter=500)
        
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
        
        total_cost = self.calculate_cost(solution)
        supply_cost = 0
        opening_cost = 0
        
        open_warehouses = set()
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                if solution[s][w] > 0:
                    supply_cost += solution[s][w] * self.supply_costs[s][w]
                    open_warehouses.add(w)
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