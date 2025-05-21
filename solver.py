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
        """Construction heuristic"""
        solution = [[0]*self.num_warehouses for _ in range(self.num_stores)]
        remaining = self.goods.copy()
        used = [0]*self.num_warehouses
        open_wh = set()
        order = sorted(range(self.num_stores), key=lambda s: self.goods[s], reverse=True)
        for s in order:
            for _, ss, w in self.efficiencies:
                if ss!=s or used[w]+remaining[s]>self.capacities[w]: continue
                if any(other in self.incompatibility_map[s] for other in range(self.num_stores) if solution[other][w]>0): continue
                solution[s][w]=remaining[s]
                used[w]+=remaining[s]
                open_wh.add(w)
                remaining[s]=0
                break
        for s in range(self.num_stores):
            if remaining[s]==0: continue
            costs = sorted([(self.supply_costs[s][w],w) for w in range(self.num_warehouses)])
            for _,w in costs:
                cap_left=self.capacities[w]-used[w]
                if cap_left<=0: continue
                if any(other in self.incompatibility_map[s] for other in range(self.num_stores) if solution[other][w]>0): continue
                amt=min(remaining[s],cap_left)
                solution[s][w]=amt
                used[w]+=amt
                open_wh.add(w)
                remaining[s]-=amt
                if remaining[s]==0: break
        return solution

    def calculate_cost(self, solution):
        supply=0
        open_wh=set()
        for s in range(self.num_stores):
            for w in range(self.num_warehouses):
                if solution[s][w]>0:
                    supply+=solution[s][w]*self.supply_costs[s][w]
                    open_wh.add(w)
        opening=sum(self.fixed_costs[w] for w in open_wh)
        return supply+opening, open_wh

    def is_feasible(self, solution):
        # demand
        for s in range(self.num_stores):
            if sum(solution[s])!=self.goods[s]: return False
        # capacity
        usage=[0]*self.num_warehouses
        for s in range(self.num_stores):
            for w in range(self.num_warehouses): usage[w]+=solution[s][w]
        if any(usage[w]>self.capacities[w] for w in range(self.num_warehouses)): return False
        # incompat
        for w in range(self.num_warehouses):
            stores=[s for s in range(self.num_stores) if solution[s][w]>0]
            for i in range(len(stores)):
                for j in range(i+1,len(stores)):
                    if stores[j] in self.incompatibility_map[stores[i]]: return False
        return True

    def multi_store_relocation(self, sol, open_wh, usage):
        improved=False
        evals=[(self.fixed_costs[w]*(1+usage[w]/self.capacities[w]),w) for w in open_wh]
        for _,w_from in sorted(evals,reverse=True):
            stores=[s for s in range(self.num_stores) if sol[s][w_from]>0]
            for s in sorted(stores, key=lambda x:self.supply_costs[x][w_from], reverse=True):
                amt=sol[s][w_from]
                # try move to another open
                for w_to in list(open_wh):
                    if w_to==w_from: continue
                    if usage[w_to]+amt>self.capacities[w_to]: continue
                    if any(o in self.incompatibility_map[s] for o in [x for x in range(self.num_stores) if sol[x][w_to]>0]): continue
                    diff=amt*(self.supply_costs[s][w_to]-self.supply_costs[s][w_from])
                    if not any(sol[x][w_to]>0 for x in range(self.num_stores)): diff+=self.fixed_costs[w_to]
                    if sum(sol[x][w_from] for x in range(self.num_stores) if x!=s)==0: diff-=self.fixed_costs[w_from]
                    if diff<0:
                        sol[s][w_from]=0; sol[s][w_to]=amt
                        usage[w_from]-=amt; usage[w_to]+=amt
                        if not any(sol[x][w_from]>0 for x in range(self.num_stores)): open_wh.discard(w_from)
                        open_wh.add(w_to)
                        return True
        return False

    def swap_warehouse_operator(self, sol, open_wh, usage):
        """Swap two stores' allocations"""
        assigns=[(s,w) for s in range(self.num_stores) for w in range(self.num_warehouses) if sol[s][w]>0]
        for _ in range(len(assigns)):
            (s1,w1),(s2,w2)=random.sample(assigns,2)
            if w1==w2: continue
            a1=sol[s1][w1]; a2=sol[s2][w2]
            if usage[w1]-a1+a2>self.capacities[w1] or usage[w2]-a2+a1>self.capacities[w2]: continue
            if any(o in self.incompatibility_map[s1] for o in [x for x in range(self.num_stores) if sol[x][w2]>0]): continue
            if any(o in self.incompatibility_map[s2] for o in [x for x in range(self.num_stores) if sol[x][w1]>0]): continue
            diff=a1*(self.supply_costs[s2][w1]-self.supply_costs[s2][w2])+a2*(self.supply_costs[s1][w2]-self.supply_costs[s1][w1])
            if not any(sol[x][w1]>0 for x in range(self.num_stores)): diff+=self.fixed_costs[w1]
            if not any(sol[x][w2]>0 for x in range(self.num_stores)): diff+=self.fixed_costs[w2]
            if sum(sol[x][w1] for x in range(self.num_stores) if x!=s1)==0: diff-=self.fixed_costs[w1]
            if sum(sol[x][w2] for x in range(self.num_stores) if x!=s2)==0: diff-=self.fixed_costs[w2]
            if diff<0:
                sol[s1][w1]=0; sol[s2][w2]=0; sol[s1][w2]=a1; sol[s2][w1]=a2
                usage[w1]=usage[w1]-a1+a2; usage[w2]=usage[w2]-a2+a1
                return True
        return False

    def harmony_search(self, initial_solution, harmony_memory_size=50, max_iterations=5000, 
                      harmony_memory_consideration_rate=0.85, pitch_adjustment_rate=0.2):
        # initialize harmony memory with feasible solutions
        harmony_memory = []
        for _ in range(harmony_memory_size):
            solution = [[0] * self.num_warehouses for _ in range(self.num_stores)]
            remaining_demand = self.goods.copy()
            warehouse_usage = [0] * self.num_warehouses
            
            # create a random feasible solution
            for s in range(self.num_stores):
                available_warehouses = [w for w in range(self.num_warehouses) 
                                     if (warehouse_usage[w] + remaining_demand[s] <= self.capacities[w] and
                                         not any(other in self.incompatibility_map[s] 
                                               for other in range(self.num_stores) 
                                               if solution[other][w] > 0))]
                if available_warehouses:
                    w = random.choice(available_warehouses)
                    solution[s][w] = remaining_demand[s]
                    warehouse_usage[w] += remaining_demand[s]
                    remaining_demand[s] = 0
            
            if self.is_feasible(solution):
                cost, _ = self.calculate_cost(solution)
                harmony_memory.append((cost, solution))
        
        if not harmony_memory:
            harmony_memory = [(self.calculate_cost(initial_solution)[0], initial_solution)]
        harmony_memory.sort()

        best_solution = harmony_memory[0][1]
        best_cost = harmony_memory[0][0]

        for iteration in range(max_iterations):
            # create new harmony
            new_solution = [[0] * self.num_warehouses for _ in range(self.num_stores)]
            remaining_demand = self.goods.copy()
            warehouse_usage = [0] * self.num_warehouses
            
            for s in range(self.num_stores):
                if random.random() < harmony_memory_consideration_rate:
                    memory_solution = random.choice(harmony_memory)[1]
                    for w in range(self.num_warehouses):
                        if memory_solution[s][w] > 0:
                            if (warehouse_usage[w] + remaining_demand[s] <= self.capacities[w] and
                                not any(other in self.incompatibility_map[s] 
                                      for other in range(self.num_stores) 
                                      if new_solution[other][w] > 0)):
                                new_solution[s][w] = remaining_demand[s]
                                warehouse_usage[w] += remaining_demand[s]
                                remaining_demand[s] = 0
                                break
                            
                            # pitch adjustment
                            if random.random() < pitch_adjustment_rate:
                                for w2 in range(self.num_warehouses):
                                    if (w2 != w and 
                                        warehouse_usage[w2] + remaining_demand[s] <= self.capacities[w2] and
                                        not any(other in self.incompatibility_map[s] 
                                              for other in range(self.num_stores) 
                                              if new_solution[other][w2] > 0)):
                                        new_solution[s][w] = 0
                                        new_solution[s][w2] = remaining_demand[s]
                                        warehouse_usage[w] -= remaining_demand[s]
                                        warehouse_usage[w2] += remaining_demand[s]
                                        remaining_demand[s] = 0
                                        break
                
                if remaining_demand[s] > 0:
                    available_warehouses = [w for w in range(self.num_warehouses) 
                                         if (warehouse_usage[w] + remaining_demand[s] <= self.capacities[w] and
                                             not any(other in self.incompatibility_map[s] 
                                                   for other in range(self.num_stores) 
                                                   if new_solution[other][w] > 0))]
                    if available_warehouses:
                        w = random.choice(available_warehouses)
                        new_solution[s][w] = remaining_demand[s]
                        warehouse_usage[w] += remaining_demand[s]
                        remaining_demand[s] = 0

            for s in range(self.num_stores):
                if remaining_demand[s] > 0:
                    for w in range(self.num_warehouses):
                        if (warehouse_usage[w] + remaining_demand[s] <= self.capacities[w] and
                            not any(other in self.incompatibility_map[s] 
                                  for other in range(self.num_stores) 
                                  if new_solution[other][w] > 0)):
                            new_solution[s][w] = remaining_demand[s]
                            warehouse_usage[w] += remaining_demand[s]
                            remaining_demand[s] = 0
                            break

            # if solution is not feasible, skip this iteration
            if not self.is_feasible(new_solution):
                continue

            # calculate cost of new solution
            new_cost, _ = self.calculate_cost(new_solution)

            # update harmony memory
            if new_cost < harmony_memory[-1][0]:
                harmony_memory[-1] = (new_cost, new_solution)
                harmony_memory.sort()
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_solution = [row[:] for row in new_solution]

        return best_solution

    def solve(self):
        sol = self.calculate_initial_solution()
        if not self.is_feasible(sol):
            sol = [[0]*self.num_warehouses for _ in range(self.num_stores)]
            rem = self.goods.copy()
            for s in range(self.num_stores):
                for cost,w in sorted([(self.supply_costs[s][w],w) for w in range(self.num_warehouses)]):
                    if rem[s]==0: break
                    used_cap=sum(sol[x][w] for x in range(self.num_stores))
                    if used_cap+1>self.capacities[w]: continue
                    if any(x in self.incompatibility_map[s] for x in range(self.num_stores) if sol[x][w]>0): continue
                    amt=min(rem[s],self.capacities[w]-used_cap)
                    sol[s][w]=amt; rem[s]-=amt

        # harmony search
        print("Applying Harmony Search...")
        harmony_solution = self.harmony_search(sol)
        cost_h, _ = self.calculate_cost(harmony_solution)
        print(f"Cost after Harmony Search: {cost_h}")

        # multi-store relocation
        cost, open_wh = self.calculate_cost(harmony_solution)
        usage = [sum(harmony_solution[x][w] for x in range(self.num_stores)) for w in range(self.num_warehouses)]
        
        for _ in range(500):
            if not self.multi_store_relocation(harmony_solution, open_wh, usage): break
        print(f"Cost after multi_store_relocation: {self.calculate_cost(harmony_solution)[0]}")
        
        # swap warehouse operator
        for _ in range(500):
            if not self.swap_warehouse_operator(harmony_solution, open_wh, usage): break
        print(f"Cost after swap_warehouse_operator: {self.calculate_cost(harmony_solution)[0]}")
        
        return harmony_solution

    def format_matrix_solution(self, solution):
        out="["
        for s in range(self.num_stores): out+=f"({','.join(map(str,solution[s]))})\n"
        return out.rstrip()+"]"

    def format_list_solution(self, solution):
        triples=[f"({self.store_ids[s]},{self.warehouse_ids[w]},{solution[s][w]})" for s in range(self.num_stores) for w in range(self.num_warehouses) if solution[s][w]>0]
        return "{"+",\n".join(triples)+"}"

    def write_solution_file(self, solution, output_file, format_type='list'):
        new_cost,_=self.calculate_cost(solution)
        if os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    lines=f.readlines(); line=next((l for l in lines if "Total Cost:" in l),None)
                    if line and int(line.split(":")[1])<=new_cost: print(f"New cost ({new_cost})>=existing; skip"); return
            except: pass
        with open(output_file,'w') as f:
            f.write(self.format_list_solution(solution) if format_type=='list' else self.format_matrix_solution(solution))
            tot,open_wh=self.calculate_cost(solution); sup=sum(solution[s][w]*self.supply_costs[s][w] for s in range(self.num_stores) for w in range(self.num_warehouses) if solution[s][w]>0)
            opn=sum(self.fixed_costs[w] for w in open_wh)
            f.write(f"\nTotal Cost: {tot}\nSupply Cost: {sup}\nOpening Cost: {opn}\nOpen Warehouses: {sorted([self.warehouse_ids[w] for w in open_wh])}\n")
        print(f"Solution written to {output_file}")
        print(f"Total cost: {tot} = {sup} + {opn}")


def main():
    if len(sys.argv)!=2:
        print(f"usage: python {sys.argv[0]} instance.json",file=sys.stderr); sys.exit(1)
    infile=sys.argv[1]
    try:
        data=json.load(open(infile))
    except FileNotFoundError:
        print(f"error: file '{infile}' not found",file=sys.stderr); sys.exit(1)
    except json.JSONDecodeError:
        print(f"error: invalid json in file '{infile}'",file=sys.stderr); sys.exit(1)
    os.makedirs('solutions',exist_ok=True)
    base=os.path.splitext(os.path.basename(infile))[0]
    outfile=f"solutions/solution-{base}.txt"
    solver=WarehouseLocationSolver(data)
    sol=solver.solve()
    if solver.is_feasible(sol):
        solver.write_solution_file(sol,outfile)
    else:
        print("no feasible solution found",file=sys.stderr); sys.exit(1)

if __name__=="__main__":
    main()