import numpy as np
import pandas as pd
import random
import math

class DietOptimization:
    def __init__(self, food_data_path, not_recommended_path):
        # Load data
        self.food_data = pd.read_csv(food_data_path)
        self.not_recommended = pd.read_csv(not_recommended_path)
        
        # Constants and Parameters
        self.N = len(self.food_data)  # Number of food items (I)
        self.MAX_TOTAL_QUANTITY = 60  # Maximum total food quantity per day
        self.MEALS = ['breakfast', 'lunch', 'dinner'] # (K)
        self.CATEGORIES = self.food_data['Category'].unique() # Get unique Category (G)
        
        # Nutritional requirements (Bj)
        self.NUTRIENT_REQUIREMENTS = {
            'Calories': 2000,
            'Protein': 50,
            'VitaminA': 900
        }
        
        # Penalty for violating soft constraints (𝜆)
        self.LAMBDA = 0.5 #random
    
    def feasible_solution(self):
        """Function Generate a random feasible solution. Q3"""
        solution = np.zeros((self.N, len(self.MEALS)))
        
        # Meal Composition (Categories) (Constrainte 4)
        for meal_index, meal in enumerate(self.MEALS):
            for cat in self.CATEGORIES:
                cat_foods = self.food_data[self.food_data['Category'] == cat].index
                if len(cat_foods) > 0:
                    food_index = random.choice(cat_foods)
                    solution[food_index, meal_index] = random.uniform(0.1, 3)# start with 0.1 -> 3 after that we multipy by 100g
        
        # Randomly generate remaining quantity
        remaining_quantity = self.MAX_TOTAL_QUANTITY - solution.sum()
        while remaining_quantity > 0:
            food_index = random.randint(0, self.N - 1)
            meal_index = random.randint(0, len(self.MEALS) - 1)
            
            # Check not recommended constraints
            if not self.verify_recommanded(food_index, meal_index):
                continue
            
            # Make sure the maximum quantity is not exceeded.
            max_quantity = self.food_data.loc[food_index, 'MaxQuantity(q100g)']
            current_quantity = solution[food_index, meal_index]
            
            add_quantity = min(random.uniform(0.1, 3), remaining_quantity, max_quantity - current_quantity)
            solution[food_index, meal_index] += add_quantity
            remaining_quantity -= add_quantity
        return solution
    
    def verify_recommanded(self, food_index, meal_index):
        """Check if a food is allowed in a specific meal"""
        not_rec_row = self.not_recommended[
            (self.not_recommended['FoodID'] == food_index + 1) & 
            (self.not_recommended['MealID'] == meal_index + 1)
        ]
        return not_rec_row['NotRecommended'].values[0] == 0
    
    #Q4. Cost function of any solution X
    def calculate_cost(self, solution):
        """Calculate total cost of the solution"""
        total_cost = 0
        for food_index in range(self.N):
            for meal_index in range(len(self.MEALS)):
                quantity = solution[food_index, meal_index]
                cost_per_100g = self.food_data.loc[food_index, 'Cost(q100g)']
                total_cost += quantity * cost_per_100g
        return total_cost
    
    def calculate_nutrient_penalty(self, solution):
        """Calculate penalty for not meeting nutritional requirements"""
        nutrient_penalty = 0
        for nutrient in ['Nutrient_Calories', 'Nutrient_Protein', 'Nutrient_VitaminA']:
            total_nutrient = sum(
                solution[food_index, meal_index] * self.food_data.loc[food_index, nutrient]
                for food_index in range(self.N)
                for meal_index in range(len(self.MEALS))
            )
            required = self.NUTRIENT_REQUIREMENTS.get(nutrient.replace('Nutrient_', ''), 0)
            if total_nutrient < required:
                nutrient_penalty += (required - total_nutrient)
        return nutrient_penalty
    
    #5. Non-recommended Items at Dinner
    def calculate_dinner_penalty(self, solution):
        """Calculate penalty for non-recommended foods at dinner"""
        dinner_penalty = 0
        for food_index in range(self.N):
            if not self.verify_recommanded(food_index, 2):  # Dinner is index 2
                dinner_penalty += solution[food_index, 2]
        return dinner_penalty

    #Q5. Objective Function
    def objective_solution(self, solution):
        """Evaluate solution with penalties for soft constraints"""
        base_cost = self.calculate_cost(solution)
        nutrient_penalty = self.calculate_nutrient_penalty(solution)
        dinner_penalty = self.calculate_dinner_penalty(solution)
        
        return base_cost + self.LAMBDA * (nutrient_penalty + dinner_penalty)

    #6. Neighborhood Function and Q8. Evaluate comparison F(x)=f(x) + α
    def neighborhood(self, solution):
        """Generate a neighbor solution by making small modifications"""
        neighbor = solution.copy()
        
        # select a food item and meal randomly
        # We can select many food or all
        food_index = random.randint(0, self.N - 1)
        meal_index = random.randint(0, len(self.MEALS) - 1)
        
        # Small perturbation (ISA)
        perturbation = random.uniform(-1, 1)
        
        # Validation nw quantity
        current_quantity = neighbor[food_index, meal_index]
        max_quantity = self.food_data.loc[food_index, 'MaxQuantity(q100g)']
        
        new_quantity = max(0, min(current_quantity + perturbation, max_quantity))
        
        # Ensure food is allowed in this meal
        if self.verify_recommanded(food_index, meal_index):
            neighbor[food_index, meal_index] = new_quantity
        
        return neighbor
    
    # Q9. Local search algo
    def local_search(self, max_iterations=1000, max_no_improvement=100):
        """Local search to optimize diet"""
        # Random solution
        current_solution = self.feasible_solution()
        current_cost = self.objective_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            # Generate a neighbor solution
            neighbor_solution = self.neighborhood(current_solution)
            neighbor_cost = self.objective_solution(neighbor_solution)
            
            # If neighbor is better, update current solution
            if neighbor_cost < current_cost:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                no_improvement_count = 0
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            else:
                no_improvement_count += 1
            
            # Stop if no improvement for a while
            if no_improvement_count >= max_no_improvement:
                break
        
        return best_solution, best_cost
    
    #Q13. ISA
    def interior_search_algorithm(self, max_iteration=1000, population_size=10):
        population = self.initialize_population(population_size)
        fitness = np.array([self.objective_solution(index) for index in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        for iteration in range(max_iteration):
            for i in range(population_size):
                new_solution = population[i] + np.random.uniform(-1, 1, 3) * (best_solution - population[i])
                new_fitness = self.objective_solution(new_solution)
                
                # Update solution
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                
                # Update Best solution
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness

        return best_solution, best_fitness
        
    # Q11. Display the best solution, its cost, and the evaluation of its nutrients
    def print_solution(self, solution, cost):
        """Print the detailed solution"""
        print("\nBest Diet Solution:")
        print(f"Total Cost: {cost:.2f} DA")
        
        # Print quantities for each food in each meal
        for meal_index, meal_name in enumerate(self.MEALS):
            print(f"\n{meal_name.capitalize()} Composition:")
            for food_index, row in self.food_data.iterrows():
                quantity = solution[food_index, meal_index]
                if quantity > 0:
                    print(f"{row['FoodName']}: {quantity:.2f} * 100g")
        
        # Print total nutrient intake
        print("\nNutrient Intake:")
        for nutrient in ['Calories', 'Protein', 'VitaminA']:
            total_intake = sum(
                solution[food_index, meal_index] * self.food_data.loc[food_index, f'Nutrient_{nutrient}']
                for food_index in range(self.N)
                for meal_index in range(len(self.MEALS))
            )
            print(f"Total {nutrient}: {total_intake:.2f}")

    #Q12. Generates population of size(10 or 20)
    def initialize_population(self, population_size):
        """Generates a random solution."""
        return [np.random.permutation(self.feasible_solution()) for _ in range(population_size)]

def main():
    food_data_path = 'data/food_data.csv'
    not_recommended_path = 'data/not_recommended.csv'

    optimizer = DietOptimization(food_data_path, not_recommended_path)
    # Optimize solution
    best_solution, best_cost = optimizer.interior_search_algorithm()
    
    # Print results
    optimizer.print_solution(best_solution, best_cost)

if __name__ == "__main__":
    main()

"""
Problems:
    - Nutrient are Fixed in ['Calories', 'Protein', 'VitaminA'].
    - result is printing in Terminal.
    - Don't put name of each Categories.
"""