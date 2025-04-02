import json

def generate_simplified_solution(data):
    """
    Generates a simplified (and likely not optimal) solution to the 
    Warehouse Location Problem.  This is for illustrative purposes only.

    Args:
        data (dict): The data loaded from the JSON file.

    Returns:
        list: A list of tuples, where each tuple represents a supply:
              (store_id, warehouse_id, quantity).
    """

    warehouses = data["warehouses"]
    stores = data["stores"]
    incompatibilities = data["incompatibilities"]

    solution = []
    opened_warehouses = set()

    # 1.  A very basic strategy:
    #     -   Open warehouses one by one.
    #     -   For each store, assign it to the first opened warehouse that can 
    #         fulfill its demand and doesn't violate incompatibilities.

    for warehouse in warehouses:
        warehouse_id = warehouse["id"]
        opened_warehouses.add(warehouse_id)  # Open the warehouse

        for store in stores:
            store_id = store["id"]
            demand = store["demand"]

            # Check if the store is already supplied
            if any(supply[0] == store_id for supply in solution):
                continue

            # Try to supply the store from the current warehouse
            remaining_demand = demand  # Initially, the whole demand

            # Check capacity and incompatibilities
            warehouse_capacity = warehouse["capacity"]
            used_capacity = sum(
                supply[2] for supply in solution if supply[1] == warehouse_id
            )

            # Incompatibility check
            incompatible_warehouse = False
            for inc_pair in incompatibilities:
                store_1 = inc_pair["store_1"]
                store_2 = inc_pair["store_2"]

                #If either store in the incompatible pair is the current store
                #and the other store is already supplied by the current warehouse
                if (store_1 == store_id or store_2 == store_id):
                  other_store = store_1 if store_2 == store_id else store_2
                  if any(supply[0] == other_store and supply[1] == warehouse_id for supply in solution):
                    incompatible_warehouse = True
                    break

            available_capacity = warehouse_capacity - used_capacity

            if (
                remaining_demand <= available_capacity and not incompatible_warehouse
            ):  
                solution.append((store_id, warehouse_id, remaining_demand))
                
    return solution

try:
    with open("output/toy-output.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'output/toy-output-solution.json' not found.")
    exit()

solution = generate_simplified_solution(data)

print("Generated Solution:")
print(solution)

