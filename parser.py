import re
import json
from collections import defaultdict

class Instance:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.content = f.read()  
            
        self.data = {
            "warehouses": [],
            "stores": [],
            "incompatibilities": []
        }
        
        # parse basic parameters
        self.num_warehouses = self._parse_value(r'Warehouses\s*=\s*(\d+);', int)
        self.num_stores = self._parse_value(r'Stores\s*=\s*(\d+);', int)
        
        # parse warehouses
        capacities = self._parse_array(r'Capacity\s*=\s*\[([^\]]+)')
        fixed_costs = self._parse_array(r'FixedCost\s*=\s*\[([^\]]+)')
        self.data["warehouses"] = [
            {"id": i+1, "capacity": c, "fixed_cost": f}
            for i, (c, f) in enumerate(zip(capacities, fixed_costs))
        ]
        
        # parse stores
        demands = self._parse_array(r'Goods\s*=\s*\[([^\]]+)')
        supply_matrix = self._parse_matrix(
            r'SupplyCost\s*=\s*\[(.*?)\];',
            self.num_stores,
            self.num_warehouses
        )
        self.data["stores"] = [
            {
                "id": i+1,
                "demand": demands[i],
                "supply_costs": {
                    f"warehouse_{j+1}": cost
                    for j, cost in enumerate(row)
                }
            } for i, row in enumerate(supply_matrix)
        ]
        
        # parse incompatibilities
        incompat_str = re.search(
            r'IncompatiblePairs\s*=\s*\[\|(.*?)\|]',
            self.content, 
            re.DOTALL
        ).group(1)
        
        self.data["incompatibilities"] = [
            {"store_1": int(s1), "store_2": int(s2)}
            for s1, s2 in re.findall(r'(\d+)\s*,\s*(\d+)', incompat_str)
        ]

    def _parse_value(self, pattern, converter):
        match = re.search(pattern, self.content)
        return converter(match.group(1)) if match else None
        
    def _parse_array(self, pattern):
        match = re.search(pattern, self.content)
        return list(map(int, re.findall(r'\d+', match.group(1)))) if match else []
        
    def _parse_matrix(self, pattern, rows, cols):
        match = re.search(pattern, self.content, re.DOTALL)
        values = list(map(int, re.findall(r'\d+', match.group(1)))) if match else []
        return [values[i*cols : (i+1)*cols] for i in range(rows)]
        
    def to_json(self):
        return json.dumps(self.data, indent=2)

def main():
    instance = Instance('instances/toy.dzn')
    print(instance.to_json())

if __name__ == "__main__":
    main()
