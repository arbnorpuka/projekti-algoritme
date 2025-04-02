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

solution =[(1, 1, 18), (2, 1, 9), (3, 1, 20), (5, 1, 12), (6, 1, 11), (7, 1, 9), (8, 1, 5), (34, 1, 5), (4, 2, 13), (9, 2, 9), (10, 2, 10), (17, 2, 7), (11, 3, 17), (12, 3, 20), (13, 4, 17), (14, 4, 17), (15, 4, 17), (16, 4, 12), (18, 4, 7), (19, 4, 5), (20, 4, 9), (49, 4, 5), (21, 5, 7), (22, 5, 9), (23, 5, 12), (24, 5, 12), (27, 5, 10), (25, 6, 17), (26, 6, 13), (28, 6, 19), (29, 6, 11), (30, 6, 15), (31, 6, 7), (36, 6, 8), (39, 6, 9), (32, 7, 7), (33, 7, 19), (35, 7, 20), (37, 7, 14), (38, 8, 12), (40, 8, 19), (41, 8, 9), (44, 8, 7), (42, 9, 16), (43, 9, 19), (45, 9, 10), (46, 9, 5), (47, 9, 14), (48, 9, 8), (50, 9, 20), (54, 9, 8), (51, 10, 13), (52, 10, 18), (53, 10, 5), (57, 10, 5), (72, 10, 7), (55, 11, 16), (56, 11, 17), (58, 11, 14), (60, 11, 17), (61, 11, 15), (67, 11, 10), (59, 12, 11), (62, 12, 7), (66, 12, 11), (63, 13, 18), (76, 13, 6), (77, 13, 6), (64, 14, 19), (65, 14, 18), (68, 15, 16), (69, 15, 18), (71, 15, 16), (70, 16, 18), (73, 16, 18), (74, 16, 19), (75, 16, 14), (78, 17, 20), (79, 17, 13), (80, 17, 16), (81, 17, 12), (82, 17, 11), (83, 17, 18), (84, 17, 10), (85, 18, 10), (86, 18, 8), (87, 18, 18), (88, 18, 5), (90, 18, 17), (89, 19, 15), (91, 19, 15), (92, 20, 19), (94, 20, 6), (95, 20, 20), (93, 21, 17), (96, 21, 12), (97, 21, 15), (98, 21, 10), (99, 21, 14), (103, 21, 12), (100, 22, 15), (101, 22, 16), (102, 22, 11), (104, 22, 10), (106, 22, 14), (105, 23, 6), (107, 23, 16), (108, 23, 13), (109, 23, 6), (111, 23, 20), (112, 23, 10), (113, 23, 8), (115, 23, 16), (110, 24, 10), (114, 24, 14)]

constraints_satisfied = check_constraints(data, solution)

if constraints_satisfied:
    print("All constraints are satisfied.")
else:
    print("At least one constraint is violated.")
