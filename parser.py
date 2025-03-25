import re
import json
import sys
import os
from collections import defaultdict

class Instance:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.content = f.read()  
            
        self.data = {
            "warehouses": [],
            "stores": [],
            "incompatibilities": [],
            "warehouse_stats": {},
            "global_stats": {}
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
        
        # Parse incompatibilities
        incompat_str = re.search(
            r'IncompatiblePairs\s*=\s*\[\|(.*?)\|]',
            self.content, 
            re.DOTALL
        ).group(1)
        
        self.data["incompatibilities"] = [
            {"store_1": int(s1), "store_2": int(s2)}
            for s1, s2 in re.findall(r'(\d+)\s*,\s*(\d+)', incompat_str)
        ]

        # add precomputed fields
        self._add_precomputed_fields()

    def _add_precomputed_fields(self):
        # warehouse stats
        capacities = [w["capacity"] for w in self.data["warehouses"]]
        fixed_costs = [w["fixed_cost"] for w in self.data["warehouses"]]
        demands = [s["demand"] for s in self.data["stores"]]
        
        total_capacity = sum(capacities)
        avg_fixed_cost = sum(fixed_costs) / self.num_warehouses
        total_demand = sum(demands)
        capacity_utilization = (total_demand / total_capacity) * 100
        
        self.data["warehouse_stats"] = {
            "total_capacity": total_capacity,
            "avg_fixed_cost": avg_fixed_cost,
            "total_demand": total_demand,
            "capacity_utilization": capacity_utilization
        }

        # store stats
        for store in self.data["stores"]:
            supply_costs = store["supply_costs"]
            cheapest_warehouse = min(supply_costs, key=lambda k: supply_costs[k])
            total_supply_cost = sum(supply_costs.values())
            
            store["cheapest_warehouse"] = cheapest_warehouse
            store["total_supply_cost"] = total_supply_cost

        # incompatibility stats
        incompat_count = defaultdict(int)
        incompat_list = defaultdict(list)

        for pair in self.data["incompatibilities"]:
            s1, s2 = pair["store_1"], pair["store_2"]
            incompat_count[s1] += 1
            incompat_count[s2] += 1
            incompat_list[s1].append(s2)
            incompat_list[s2].append(s1)

        for store in self.data["stores"]:
            store_id = store["id"]
            store["incompatibility_count"] = incompat_count.get(store_id, 0)
            store["incompatible_stores"] = incompat_list.get(store_id, [])

        # stats
        total_fixed_cost = sum(fixed_costs)
        total_supply_cost = sum(
            sum(store["supply_costs"].values())
            for store in self.data["stores"]
        )
        cost_per_unit_demand = (total_fixed_cost + total_supply_cost) / total_demand

        self.data["global_stats"] = {
            "total_fixed_cost": total_fixed_cost,
            "total_supply_cost": total_supply_cost,
            "cost_per_unit_demand": cost_per_unit_demand
        }

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
    if len(sys.argv) < 2:
        print("Usage: python3 parser.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        instance = Instance(filename)
        json_output = instance.to_json()
        
        print(json_output)

        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{base_name}-output.json")
        with open(output_file, 'w') as f:
            f.write(json_output)

        print(f"Output saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()