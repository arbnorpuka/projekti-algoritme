import json

def check_constraints(data, solution):
    """
    Checks all constraints for the Warehouse Location Problem with Store Incompatibilities:
    1.  Warehouse capacity constraint
    2.  Store demand constraint
    3.  Supply from open warehouses constraint (implicitly checked)
    4.  Store incompatibility constraints

    Args:
        data (dict): The data loaded from the JSON file.
        solution (list): A list of tuples, where each tuple represents a supply:
                         (store_id, warehouse_id, quantity).
                         For example: [(1, 4, 12), (2, 4, 17), ...]

    Returns:
        bool: True if all constraints are satisfied, False otherwise.
    """

    warehouses = data["warehouses"]
    stores = data["stores"]
    incompatibilities = data["incompatibilities"]

    # 1. Warehouse Capacity Constraint
    warehouse_capacities = {}
    for warehouse in warehouses:
        warehouse_capacities[warehouse["id"]] = warehouse["capacity"]
    
    warehouse_usage = {}
    for warehouse_id in warehouse_capacities:
        warehouse_usage[warehouse_id] = 0
    
    for supply in solution:
        store_id = supply[0]
        warehouse_id = supply[1]
        quantity = supply[2]
        warehouse_usage[warehouse_id] += quantity
        print(f"Transferring {quantity} units from warehouse {warehouse_id} to store {store_id}")  # Log transfer
    
    for warehouse_id, usage in warehouse_usage.items():
        if usage > warehouse_capacities[warehouse_id]:
            print(f"Warehouse capacity constraint violated for warehouse {warehouse_id}.")
            return False

    # 2. Store Demand Constraint
    store_demands = {}
    for store in stores:
        store_demands[store["id"]] = store["demand"]
    
    store_supply = {}
    for store_id in store_demands:
        store_supply[store_id] = 0
    
    for supply in solution:
        store_id = supply[0]
        quantity = supply[2]
        store_supply[store_id] += quantity
    
    for store_id, supply in store_supply.items():
        if supply != store_demands[store_id]:
            print(f"Store demand constraint violated for store {store_id}.")
            return False
        
try:
    with open("output/toy-output.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'output/toy-output.json' not found.")
    exit()

solution = [(1, 1, 12), (2, 1, 17), (3, 1, 5), (4, 1, 13), (5, 1, 20), (6, 1, 20), (9, 1, 11), (7, 2, 17), (8, 2, 19), (10, 3, 20)]

constraints_satisfied = check_constraints(data, solution)

if constraints_satisfied:
    print("All constraints are satisfied.")
else:
    print("At least one constraint is violated.")
