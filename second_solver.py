import pandas as pd
import numpy as np
import random

class DietOptimization:
    def __init__(self, csv_path):
        # Read the diet data
        self.data = pd.read_csv(csv_path)
        
        # Normalize the 'Type' column to have consistent capitalization
        self.data['Type'] = self.data['Type'].str.capitalize()
        
        # Constants and parameters
        self.N = len(self.data)  # Number of food items
        self.MEALS = 3  # Breakfast, Lunch, Dinner
        self.EPSILON = 0.5  # Minimum required amount per category per meal
        self.MAX_TOTAL_QUANTITY = 60  # Maximum total quantity (6 kg)
        self.LAMBDA = 100  # Penalty for non-recommended dinner items
        
        # Predefined categories (with capitalized first letter)
        self.CATEGORIES = ['Drink', 'Main', 'Appetizer', 'Side dish', 'Dessert', 'Sauce']
        
        # Define daily nutritional requirements (these are approximations)
        self.DAILY_REQUIREMENTS = {
            'Calories(kcal)': 2000,
            'Protein (g)': 50,
            'Carbohydrates (g)': 250,
            'Fat (g)': 70,
            'Fiber (%\AQR)': 25,
            'Iron (mg)': 8,
            'Folate (mcg)': 400,
            'Sodium (mg)': 2300,
            'VitaminA (UI)': 5000,
            'VitaminC (mg)': 90
        }
        
        # Max values for some nutrients (soft constraint)
        self.MAX_REQUIREMENTS = {
            'Calories(kcal)': 2500,
            'Sodium (mg)': 2500,
            'Fat (g)': 90
        }
    
    def generate_initial_solution(self):
        """
        Generate a random feasible initial solution
        Solution is a 2D array: [food_item_index, meal_index, quantity]
        """
        solution = []
        total_quantity = 0
        
        # Ensure at least one item from each category in each meal
        for meal in range(self.MEALS):
            # Track category coverage
            category_coverage = {cat: 0 for cat in self.CATEGORIES}
            
            # Add items to the meal
            attempts = 0
            while not self._check_category_coverage(category_coverage, meal) and attempts < 100:
                attempts += 1
                
                # Randomly select a food item
                item_index = random.randint(0, self.N - 1)
                item = self.data.iloc[item_index]
                
                # Check if item is recommended for the meal
                if self._is_item_recommended(item_index, meal):
                    # Limit quantity to prevent exceeding total limit
                    max_quantity = min(
                        self.MAX_TOTAL_QUANTITY - total_quantity, 
                        item['budget (da)'] / 100,  # Convert budget to quantity
                        self._get_max_item_quantity(item_index)
                    )
                    
                    if max_quantity > 0:
                        quantity = random.uniform(0.1, max_quantity)
                        
                        solution.append([item_index, meal, quantity])
                        total_quantity += quantity
                        
                        # Update category coverage
                        category_coverage[item['Type']] += quantity
            
            # If unable to cover categories, raise an error
            if attempts >= 100:
                raise ValueError(f"Unable to generate a solution with category coverage for meal {meal}")
        
        return solution
    
    def _is_item_recommended(self, item_index, meal):
        """
        Check if an item is recommended for a specific meal
        """
        # For dinner (meal index 2), some items are not recommended
        if meal == 2:
            non_dinner_types = ['Appetizer', 'Dessert']
            return self.data.iloc[item_index]['Type'] not in non_dinner_types
        return True
    
    def _get_max_item_quantity(self, item_index):
        """
        Get maximum allowed quantity for a specific food item
        """
        # Default max is 3 (300g) if not specified
        return min(3.0, self.data.iloc[item_index]['budget (da)'] / 100)
    
    def _check_category_coverage(self, category_coverage, meal):
        """
        Ensure each category has at least minimum required quantity
        """
        return all(qty >= self.EPSILON for qty in category_coverage.values())
    
    def calculate_cost(self, solution):
        """
        Calculate the total cost of the solution
        """
        total_cost = sum(
            self.data.iloc[int(item[0])]['budget (da)'] * item[2] 
            for item in solution
        )
        
        # Add penalty for non-recommended dinner items
        dinner_penalty = sum(
            self.LAMBDA * item[2] 
            for item in solution 
            if item[1] == 2 and not self._is_item_recommended(int(item[0]), 2)
        )
        
        return total_cost + dinner_penalty
    
    def evaluate_solution(self, solution):
        """
        Evaluate nutritional constraints of the solution
        """
        # Aggregate nutrients
        nutrients = {nutrient: 0 for nutrient in self.DAILY_REQUIREMENTS.keys()}
        
        for item_index, _, quantity in solution:
            item = self.data.iloc[item_index]
            for nutrient in nutrients.keys():
                # Multiply nutrient value by quantity (converting to 100g basis)
                nutrients[nutrient] += item[nutrient] * quantity
        
        # Check nutritional requirements
        violations = 0
        for nutrient, value in nutrients.items():
            # Check both minimum and maximum requirements
            if value < self.DAILY_REQUIREMENTS.get(nutrient, 0):
                violations += self.DAILY_REQUIREMENTS[nutrient] - value
            
            if nutrient in self.MAX_REQUIREMENTS and value > self.MAX_REQUIREMENTS[nutrient]:
                violations += value - self.MAX_REQUIREMENTS[nutrient]
        
        return nutrients, violations
    
    def local_search(self, max_iterations=100):
        """
        Implement local search algorithm
        """
        # Generate initial solution
        best_solution = self.generate_initial_solution()
        best_cost = self.calculate_cost(best_solution)
        best_nutrients, best_violations = self.evaluate_solution(best_solution)
        
        for _ in range(max_iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(best_solution)
            
            # Calculate neighbor cost
            neighbor_cost = self.calculate_cost(neighbor)
            neighbor_nutrients, neighbor_violations = self.evaluate_solution(neighbor)
            
            # Improvement criteria: lower cost and fewer violations
            if (neighbor_cost < best_cost) or (neighbor_violations < best_violations):
                best_solution = neighbor
                best_cost = neighbor_cost
                best_nutrients = neighbor_nutrients
                best_violations = neighbor_violations
        
        return best_solution, best_cost, best_nutrients, best_violations
    
    def _generate_neighbor(self, solution):
        """
        Generate a neighbor solution by making small modifications
        """
        neighbor = solution.copy()
        
        # Randomly modify 1-2 items
        for _ in range(random.randint(1, 2)):
            # Randomly choose an item to modify
            index = random.randint(0, len(neighbor) - 1)
            
            # Slightly perturb the quantity
            neighbor[index][2] *= random.uniform(0.8, 1.2)
            
            # Ensure quantity remains within constraints
            neighbor[index][2] = min(
                neighbor[index][2], 
                self._get_max_item_quantity(int(neighbor[index][0]))
            )
        
        return neighbor

# Main execution
def main():
    # Initialize the diet optimization problem
    optimizer = DietOptimization('data/diet_data.csv')
    
    # Run local search
    best_solution, best_cost, best_nutrients, best_violations = optimizer.local_search()
    
    # Print results
    print("Best Solution:")
    for item_index, meal, quantity in best_solution:
        food_item = optimizer.data.iloc[item_index]
        print(f"Meal {meal+1}: {food_item['Food']} - {quantity:.2f} * 100g")
    
    print("\nTotal Cost:", best_cost)
    
    print("\nNutrient Intake:")
    for nutrient, value in best_nutrients.items():
        print(f"{nutrient}: {value:.2f} (Target: {optimizer.DAILY_REQUIREMENTS.get(nutrient, 'N/A')})")
    
    print("\nViolations:", best_violations)

if __name__ == "__main__":
    main()