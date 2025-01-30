import time
start_time = time.time()

from optimizer import *

# Question 10 & 14
def main():
    """Main function for testing optimization diet"""

    file_path_excel = "diet_data.xlsx"
    #file_path_csv = ["Table_1.csv", "Table_2.csv"] file_path2=file_path_csv[1],
    #file_path_txt = "diet_data.txt"
    group_cible = "Children (4-8 years)"
    categories_not_recommended_at_dinner = ["Appetizer", "Side dish", "Sauce"]

    solver = DietOptimization(
        file_path=file_path_excel,
        group_cible=group_cible, 
        categories_not_recommended_at_dinner=categories_not_recommended_at_dinner
    )

    # Question 10
    local_solution, local_cost = solver.local_search()
    print("*"*41 + "Local Search" + "*"*41)
    solver.print_solution(local_solution)

    print("\n" + "-"*40 + "\n")
    
    # Question 14
    interior_solution, interior_cost = solver.interior_search()
    print("*"*41 + "Interior Search" + "*"*41)
    solver.print_solution(interior_solution)

    print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()
    
    #Calculate excution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")