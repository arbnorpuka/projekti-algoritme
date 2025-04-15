import json
import sys
import re
from collections import defaultdict

class SolutionValidator:
    def __init__(self, instance_file, solution_file):
        self.instance = self.load_instance(instance_file)
        self.solution = self.load_solution(solution_file)
        self.violations = 0
        self.costs = {
            'total': 0,
            'supply': 0,
            'opening': 0
        }
        self.usage = {
            'warehouses': defaultdict(int),
            'stores': defaultdict(int),
            'assignments': defaultdict(set)
        }

    def load_instance(self, filename):
        with open(filename) as f:
            data = json.load(f)
        
        warehouses = {wh['id']: wh for wh in data['warehouses']}
        stores = {s['id']: s for s in data['stores']}
        
        for store in stores.values():
            store['supply_costs'] = {
                int(k.split('_')[1]): v 
                for k, v in store['supply_costs'].items()
            }
        
        return {
            'warehouses': warehouses,
            'stores': stores,
            'incompatibilities': [
                (inc['store_1'], inc['store_2'])
                for inc in data.get('incompatibilities', [])
            ]
        }

    def load_solution(self, filename):
        with open(filename) as f:
            content = f.read().strip()
        
        if content.startswith('['):
            return self.parse_matrix(content)
        elif content.startswith('{'):
            return self.parse_list(content)
        else:
            raise ValueError("Unknown solution format")

    def parse_matrix(self, content):
        solution = defaultdict(dict)
        rows = content[1:-1].strip().split('\n')
        for store_idx, row in enumerate(rows):
            store_id = list(self.instance['stores'].keys())[store_idx]
            values = [int(x.strip()) for x in row[1:-1].split(',')]
            for wh_idx, amount in enumerate(values):
                if amount > 0:
                    wh_id = list(self.instance['warehouses'].keys())[wh_idx]
                    solution[store_id][wh_id] = amount
        return solution

    def parse_list(self, content):
        solution = defaultdict(dict)
        triples = re.findall(r'\((\d+),(\d+),(\d+)\)', content)
        for store_id, wh_id, amount in triples:
            solution[int(store_id)][int(wh_id)] = int(amount)
        return solution

    def validate(self):
        self.check_store_demands()
        self.check_warehouse_capacities()
        self.check_incompatibilities()
        self.calculate_costs()
        self.print_report()

    def check_store_demands(self):
        for store_id, store in self.instance['stores'].items():
            delivered = sum(self.solution.get(store_id, {}).values())
            if delivered != store['demand']:
                self.violations += 1
                print(f"Violation: Store {store_id} received {delivered}, expected {store['demand']}")

    def check_warehouse_capacities(self):
        for wh_id, wh in self.instance['warehouses'].items():
            used = sum(
                amount 
                for store in self.solution.values() 
                for w, amount in store.items() 
                if w == wh_id
            )
            self.usage['warehouses'][wh_id] = used
            if used > wh['capacity']:
                self.violations += 1
                print(f"Violation: Warehouse {wh_id} capacity exceeded ({used} > {wh['capacity']})")

    def check_incompatibilities(self):
        for store_id, assignments in self.solution.items():
            for wh_id in assignments:
                self.usage['assignments'][wh_id].add(int(store_id))
        
        for s1, s2 in self.instance['incompatibilities']:
            for wh_id, stores in self.usage['assignments'].items():
                if {s1, s2}.issubset(stores):
                    self.violations += 1
                    print(f"Violation: Incompatible stores {s1} and {s2} both served by warehouse {wh_id}")

    def calculate_costs(self):
        for store_id, assignments in self.solution.items():
            for wh_id, amount in assignments.items():
                unit_cost = self.instance['stores'][store_id]['supply_costs'][wh_id]
                self.costs['supply'] += amount * unit_cost
        
        opened_warehouses = set(
            wh_id
            for assignments in self.solution.values()
            for wh_id in assignments.keys()
        )
        self.costs['opening'] = sum(
            self.instance['warehouses'][wh_id]['fixed_cost']
            for wh_id in opened_warehouses
        )
        self.costs['total'] = self.costs['supply'] + self.costs['opening']

    def print_report(self):
        print(f"Total cost: {self.costs['total']} = {self.costs['supply']} (supply) + {self.costs['opening']} (opening)")
        print(f"Open warehouses: {sorted(self.usage['assignments'].keys())}")
        print(f"Number of violations: {self.violations}")
        print("Status:", "VALID" if self.violations == 0 else "INVALID")

def main():
    if len(sys.argv) != 3:
        print("Usage: python validator.py instance.json solution.txt")
        sys.exit(1)
    
    validator = SolutionValidator(sys.argv[1], sys.argv[2])
    validator.validate()
    sys.exit(1 if validator.violations > 0 else 0)

if __name__ == "__main__":
    main()