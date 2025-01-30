"""
Diet Optimization Project - TP Solution
Author: Mouhadjer Aissa
Date Started: 18-12-2024
Date Completed: 22-12-2024

Description:
This project focuses on optimizing diet plans using local and interior search algorithms.
The goal is to create balanced meal plans that meet nutritional requirements, minimize costs, and respect constraints.

Key Features:

    - Implements feasible and correct solutions.

    - Supports nutrient ranges and category constraints.

    - Uses penalty-based cost evaluation for violations.

    - Provides local and interior search optimization approaches.

Contact:
Email: mouhadjeraissa49@gmail.com
Github: https://github.com/Mouhadjer20/
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class UnitConverter:
    # Convert any unit to (g)
    TO_GRAM = {
        'g': 1.0,
        'mg': 0.001,
        'mcg': 0.000001,
        'µg': 0.000001,
        'kg': 1000.0,
        'ui': 0.000025,
        '%aqr': 1.0,  # Keep as is, will be handled separately
        'kcal': 1.0,  # Keep as is
        'None': 1.0
    }

    # Convert (g) to any unit
    GRAM_TO = {
        'g': 1.0,
        'mg': 1000,
        'mcg': 1000000,
        'µg': 100000,
        'kg': 0.0001,
        'ui': 250000,
        '%aqr': 1.0,  # Keep as is, will be handled separately
        'kcal': 1.0,  # Keep as is
        'None': 1.0
    }

    @classmethod
    def get_default_unit(cls, nutrient: str) -> str:
        if '(' or ')' in nutrient:
            unit = nutrient.split('(')[-1].split(')')[0].lower()
            if unit in cls.TO_GRAM.keys():
                return unit
        return 'None'
    
    @classmethod
    def convert_to_gram(cls, value: float, nutrient: str) -> float:
        return value * cls.TO_GRAM[cls.get_default_unit(nutrient)]

    @classmethod
    def convert_from_gram_to(cls, value: float, nutrient: str) -> Tuple[str, float]:
        unit = cls.get_default_unit(nutrient)
        return str(round(value * cls.GRAM_TO[unit],4)) + ' (' + unit + ')', value * cls.GRAM_TO[unit]

class DataReader:
    def __init__(self, file_path: str, file_path2: str = ''):
        self.file_path = file_path
        self.file_path2 = file_path2
        self.file_type = self.file_path.split('.')[-1].lower()
        self.file_type2 = self.file_path2.split('.')[-1].lower()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.file_type == 'xlsx':
            return self.read_excel()
        elif self.file_type == 'txt':
            return self.read_txt()
        elif self.file_type == 'csv' and self.file_type2 == 'csv':
            return self.read_csv()
        raise FileExistsError("We need file with type 'txt' or 'xlsx' or 2 file'csv'")

    @staticmethod
    def parse_nutrient_value(value: str, nutrient: str) -> Dict[str, float]:
        """Parse nutrient requirement values that may be ranges or comparisons."""

        if isinstance(value, (int, np.int64, float)):
            return {'min': UnitConverter.convert_to_gram(float(value), nutrient), 'max': float('inf')}

        if isinstance(value, str):
            if value in ("", "N/A", "None"):
                return {'min': 0, 'max': float('inf')}
            
            value = value.replace(',', '.')
            # Handle format "1600-2000"
            if '-' in value:
                min_val, max_val = map(float, value.split('-'))
                min_val = UnitConverter.convert_to_gram(min_val, nutrient)
                max_val = UnitConverter.convert_to_gram(max_val, nutrient)
                return {'min': min_val, 'max': max_val}

            # Handle format "<100" or ">100"
            if '<' in value:
                return {'min': 0, 'max': UnitConverter.convert_to_gram(float(value.replace('<', '')), nutrient)}
            if '>' in value:
                return {'min': UnitConverter.convert_to_gram(float(value.replace('>', '')), nutrient), 'max': float('inf')}

        # Handle numbers
        try:
            num_value = float(value)
            num_value = UnitConverter.convert_to_gram(num_value, nutrient)
            return {'min': num_value, 'max': float('inf')}
        except ValueError:
            return {'min': 0, 'max': float('inf')}

    @staticmethod
    def parse_value_food(value: str, nutrient: str) -> float:
        """Parse nutrient values in food."""

        if isinstance(value, (int, np.int64, float)):
            if np.isnan(value):
                return 0
            return UnitConverter.convert_to_gram(float(value), nutrient)

        if isinstance(value, str):
            if value in ("", "None", "N/A"):
                return 0

            value = value.replace(',', '.')
            # Handle format "1600-2000"
            if '-' in value:
                min_val, max_val = map(float, value.split('-'))
                return np.random.uniform(min_val, max_val)

            # Handle format "<100" or ">100"
            if '<' or '>' in value:
                return UnitConverter.convert_to_gram(float(value.replace('<', '').replace('>', '')), nutrient)

        # Handle numbers
        try:
            if np.isnan(value):
                return 0
            num_value = float(value)
            return UnitConverter.convert_to_gram(num_value, nutrient)
        except ValueError:
            raise ValueError(f"You declared unexepted value : '{value}'")

    def read_txt(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read and parse data from text file."""
        with open(self.file_path, 'r') as file:
            content = file.read().strip().split('\n\n')
        
        if len(content) != 2:
            raise ValueError("Expected 2 tables in text file")
            
        tables = [
            [row.split('\t') for row in table.strip().split('\n')]
            for table in content
        ]

        # Convert to DataFrames and Return
        return (
            pd.DataFrame(tables[0][1:], columns=tables[0][0]),
            pd.DataFrame(tables[1][1:], columns=tables[1][0])
        )

    def read_excel(self, sheet_names: List[str] = ['Table_1', 'Table_2']) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from Excel file with error handling."""
        try:
            return(
                pd.DataFrame(pd.read_excel(self.file_path, sheet_name=sheet_names[0])),
                pd.DataFrame(pd.read_excel(self.file_path, sheet_name=sheet_names[1]))
            )
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def read_csv(self):
        """Load data from CSV file with error handling."""
        try:
            return(
                pd.read_csv(self.file_path),
                pd.read_csv(self.file_path2)
            )
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

class DietOptimization:
    """Diet optimization using local search algorithms or interior search alogrithm."""
    # Question 1 & 2
    def __init__(self, file_path: str, file_path2_csv: str='', group_cible: str='diabetic adult women', categories_not_recommended_at_dinner: List[str] = None):
        """
        Initialize sets & parametars the diet optimizer.
        
        Args:
            file_path: Path to data file
            file_path2_csv: Path to data file for csv
            group_cible: Target group for nutritional requirements
            categories_not_recommended_at_dinner: Categories not recommended at dinner
        """
        try:
            # Read data
            self.reader = DataReader(file_path, file_path2_csv)
            self.food_items, self.nutritional_req = self.reader.load_data()

            group_req = self.nutritional_req.loc[self.nutritional_req['Group'] == group_cible].iloc[0] # Bj: Daily requirement for nutrient j, j ∈ {1, ... , M}.
            self.daily_req = {
                nutrient: self.reader.parse_nutrient_value(group_req[nutrient], nutrient)
                for nutrient in group_req.index if nutrient != 'Group'
            }

            # Initialize sets
            self.N = len(self.food_items)   # I : Set of all N food items, I = {1,2, ... , N}.
            self.NUTRIENTS = [n for n in self.daily_req.keys()] # J: set of M nutrients (calories, protein, vitamin A...)
            self.MEALS = ['breakfast', 'lunch', 'dinner']   # K : Set of 3 meals
            self.CATEGORIES = self.food_items['Type'].str.capitalize().unique()   # G : Set of food categories, G = {1,2, ... , Gmax }. Drink, Principal item, Dessert

            # Initialize parameters
            self.category_indices = {# cat (i, g): get when execute any constraint or Objective function (إذا food ينتمي إلى gategory) => 0 or 1
                cat: np.where(self.food_items['Type'].str.capitalize() == cat)[0]
                for cat in self.CATEGORIES
            }
            self.food_costs = self.food_items['budget (da)'].astype(float).values    # ci: like 'Aij' and 'cat(i,g)', (سعر food i in 100g)
            self.food_qmax = self.food_items['qmax (g)'].astype(float).values
            self.MAX_TOTAL_FOOD_QUANTITY = 60   # Quantity of all food items per day (6 kg)
            self.LAMBDA = 100   #  λ : notRecommended(i, 3), Random value for penaltie non-recommended items in dinner (k == 3)
            # notRecommended(i, k): same thing to 'ci' ( food i not recommended in meal k) => 0 or 1
            self.CATEGORIES_NOT_RECOMMENDED_AT_DINNER = set(categories_not_recommended_at_dinner or ['Appetizer', 'Side dish', 'Sauce']) # Example of categories not recommender at dinner. after that i change to user put what is your specefic categories
            self.EPSILON = 0.5   # ε = 0.5

            self.nutrient_matrix = np.zeros((len(self.NUTRIENTS), self.N))# Aij: also get in constriant or objective fun (نسبة 'nutrient' j in food i in 100 g)
            for nutrient_idx, nutrient in enumerate(self.NUTRIENTS):
                for food in range(self.N):
                    self.nutrient_matrix[nutrient_idx, food] = self.reader.parse_value_food(self.food_items.loc[food, nutrient], nutrient)
        except Exception as e:
            raise ValueError(f"Error initializing diet optimization: {str(e)}")
            raise TypeError(f"Error initializing diet optimization: {str(e)}")

    # Question 3
    def __feasible_solution(self) -> np.ndarray:
        """
        Generate a random feasible solution.
        
        Returns:
            Array of food quantities per meal
        """
        solution = np.zeros((self.N, len(self.MEALS)))

        for meal_idx in range(len(self.MEALS)):
            for category in self.CATEGORIES:
                category_foods = self.category_indices[category]
                if len(category_foods) > 0:
                    food_idx = np.random.choice(category_foods)
                    solution[food_idx, meal_idx] = np.random.uniform(0, min(self.food_qmax[food_idx], self.EPSILON)) #0.04 g, 0.5
        return solution

    # Question 4
    def __evaluate_cost(self, solution: np.ndarray) -> float:
        """
        Calculate total cost of the solution.
        
        Args:
            solution: Current food quantities
            
        Returns:
            Total cost including penalties
        """
        return np.sum(self.food_costs[:, np.newaxis] * solution) / 100

    def __dinner_penalitie(self, solution: np.ndarray) -> int:
        """calculate somme of not recommended at Dinner for any solution X"""
        dinner_penalties = 0
        for food_idx in range(len(solution)):
            category_food = self.food_items.loc[food_idx, 'Type'] # Get category of food
            dinner_penalties += 1 if category_food.capitalize() in self.CATEGORIES_NOT_RECOMMENDED_AT_DINNER else 0
        
        return dinner_penalties

    # Question 5
    def objective_function(self, solution: np.ndarray) -> float:
        """
        Calculate objective function value.
        
        Args:
            solution: Current food quantities
            
        Returns:
            Objective value including penalties
        """
        return self.__evaluate_cost(solution) + self.LAMBDA * self.__dinner_penalitie(solution)

    # Question 6
    def __neighborhood(self, solution: np.ndarray) -> np.ndarray:
        """
        Generate a neighbor solution with small modifications.
        
        Args:
            solution: Current solution
            
        Returns:
            Modified neighbor solution
        """
        neighbor = solution.copy()  # Generate neighborhood solution
        
        # Select Randomly food and meal
        food_idx = np.random.randint(0, self.N)
        meal_idx = np.random.randint(0, len(self.MEALS))
        
        # Modify quantity
        current_qty = neighbor[food_idx, meal_idx] # [0, min(self.Epsilon, self.qmaxi)]
        max_qty = self.food_qmax[food_idx]
        perturbation = np.random.uniform(-self.EPSILON, self.EPSILON) #espsilon = 0.5 (-1, 1)
        new_qty = max(0, min(current_qty + perturbation, max_qty, self.EPSILON)) #np.clip(current_qty + perturbation, 0, min(max_qty, self.EPSILON))
        
        neighbor[food_idx, meal_idx] = new_qty
        return self.__correct_solution(neighbor)

    # Question 7
    def __correct_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Adjusts the solution to meet nutrient requirements within the given ranges.
        
        Args:
            solution: Current solution
            
        Returns:
            Corrected solution
        """
        corrected = solution.copy()

        for _ in range(4):  # Random range for just limit modifications
            nutrient_totals = self.__calculate_nutrient_totals(tuple(map(tuple, corrected)))
            violations = self.__get_nutrient_violations(nutrient_totals) #{protein: 45g, cl: 60kcal, vitA: 65, vitC: 0}

            if not violations:
                break

            for nutrient, value in violations.items():
                if value < 0:  # We need reject food to low quantity of this nutrient
                    self.__remove_food_for_nutrient(solution, nutrient, abs(value))
                elif value > 0:  # We need more from this nutrient
                    self.__add_food_for_nutrient(solution, nutrient, value)

            if self.__is_solution_valid(solution):
                break

        return solution
    
    #****************************************functions uses in correct_solution********************************************
    def __calculate_nutrient_totals(self, solution_tuple: tuple) -> Dict[str, float]:
        """Calculates total nutrient intake from the solution."""
        solution = np.array(solution_tuple)
        return {
            nutrient: np.sum(solution * self.nutrient_matrix[i, :, np.newaxis])
            for i, nutrient in enumerate(self.NUTRIENTS)
        }

    def __get_nutrient_violations(self, totals: Dict[str, float]) -> Dict[str, float]:
        """Optimized nutrient violation calculation."""
        violations = {}
        for nutrient, total in totals.items():
            req = self.daily_req[nutrient]
            if total < req['min']:
                violations[nutrient] = req['min'] - total
            elif req['max'] != float('inf') and total > req['max']:
                violations[nutrient] = req['max'] - total
        return violations

    def __add_food_for_nutrient(self, solution: np.ndarray, nutrient: str, value: float):
        """Adds food to the solution to increase a specific nutrient."""
        nutrient_idx = self.NUTRIENTS.index(nutrient)
        nutrient_values = self.nutrient_matrix[nutrient_idx]
        
        # Sort indices by nutrient content
        sorted_indices = np.argsort(-nutrient_values)
        
        for idx in sorted_indices:
            meal_idx = np.random.randint(0, len(self.MEALS))
            current_qty = solution[idx, meal_idx]
            try:

                max_addition = min(self.food_qmax[idx] - current_qty, value / nutrient_values[idx])
                
                if max_addition > 0:
                    test_solution = solution.copy()
                    test_solution[idx, meal_idx] += max_addition
                    
                    if self.__comparison_function(test_solution) < self.__comparison_function(solution):
                        solution[idx, meal_idx] += max_addition
                        break
            except:
                continue
    
    def __remove_food_for_nutrient(self, solution: np.ndarray, nutrient: str, value: float):
        """Removes food from the solution to decrease a specific nutrient."""
        nutrient_idx = self.NUTRIENTS.index(nutrient)
        nutrient_values = self.nutrient_matrix[nutrient_idx]
        
        # Sort indices by nutrient content (ascending)
        sorted_indices = np.argsort(nutrient_values)
        
        for idx in sorted_indices:
            meal_idx = np.random.randint(0, len(self.MEALS))
            current_qty = solution[idx, meal_idx]
            if current_qty >= self.EPSILON:
                try:
                    reduction = min(current_qty - self.EPSILON, value / nutrient_values[idx])
                
                    if reduction > 0:
                        test_solution = solution.copy()
                        test_solution[idx, meal_idx] -= reduction
                        
                        if self.__comparison_function(test_solution) < self.__comparison_function(solution):
                            solution[idx, meal_idx] -= reduction
                            break
                except:
                    break

    def __is_solution_valid(self, solution: np.ndarray) -> bool:
        """Checks if the solution is valid against all constraints."""
        nutrient_totals = self.__calculate_nutrient_totals(tuple(map(tuple, solution)))
        return not self.__get_nutrient_violations(nutrient_totals)
    #***********************************************************************************************************************

    # Question 8
    def __comparison_function(self, solution: np.ndarray, alpha: float = 10.0) -> float:
        """
        Evaluate solution quality with stronger penalties for nutrient violations.
                
        Args:
            solution: Current solution
            alpha: Penalty weight
            
        Returns:
            Comparison function value
        """
        obj_value = self.objective_function(solution)
        violations = 0
        
        # Constraint 1
        nutrients = self.__calculate_nutrient_totals(tuple(map(tuple, solution))) 
        for nutrient, total in nutrients.items():
            req = self.daily_req[nutrient]
            if total < req['min']:
                # Stronger penalty for being below minimum
                violations += (req['min'] - total) / req['min']
            elif total > req['max'] and req['max'] != float('inf'):
                violations += (total - req['max']) / req['max']
        
        # Constraint 2
        total_qty = np.sum(solution)
        if total_qty > self.MAX_TOTAL_FOOD_QUANTITY:
            violations += (total_qty - self.MAX_TOTAL_FOOD_QUANTITY)

        # Constriant 3
        violations += np.sum(
            np.maximum(0, solution - self.food_qmax[:, np.newaxis])
        )

        # Constraint 4
        for meal_idx, meal in enumerate(self.MEALS):
            for category in self.CATEGORIES:
                items_in_category = self.category_indices[category]
                total_quantity = np.sum(solution[items_in_category, meal_idx])
                if total_quantity < self.EPSILON:
                    violations += (self.EPSILON - total_quantity)
        
        return obj_value + alpha * violations

    # Question 9
    def local_search(self, max_iterations: int = 500, max_no_improve: int = 100) -> Tuple[np.ndarray, float]:
        """
        Implements a local search algorithm to find an improved solution.
        
        Parameters:
            - max_iterations: Maximum number of iterations
            - max_iterations_without_improvement: Stop if no improvement found after this many iterations
        
        Returns:
            - Tuple containing best solution found and its cost
        """
        current_solution = self.__correct_solution(self.__feasible_solution())
        best_solution = current_solution.copy()
        best_cost = self.__comparison_function(best_solution)
        no_improve = 0
        
        for _ in range(max_iterations):
            if no_improve >= max_no_improve:
                break

            neighbor = self.__neighborhood(current_solution)
            neighbor_cost = self.__comparison_function(neighbor)
            
            # accept solutions that meet minimum nutrient req
            nutrients = self.__calculate_nutrient_totals(tuple(map(tuple, neighbor)))
            nutrients_met = all(
                nutrients[n] >= self.daily_req[n]['min'] 
                for n in self.NUTRIENTS 
                if not np.isnan(nutrients[n])
            )
            
            if nutrients_met and neighbor_cost <= self.__comparison_function(current_solution):
                current_solution = neighbor.copy()
                if neighbor_cost < best_cost:
                    best_solution = neighbor.copy()
                    best_cost = neighbor_cost
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        return best_solution, self.objective_function(best_solution)

    # Question 11
    def print_solution(self, solution: np.ndarray) -> None:
        """Print the solution details."""
        print("\nOptimal Diet Plan:")
        total_cost = self.objective_function(solution)
        for k, meal in enumerate(self.MEALS):
            print(f"\n{meal.capitalize()}:")
            for i in range(self.N):
                if solution[i, k] > 0:
                    food_name = self.food_items.iloc[i]['Food']
                    quantity = solution[i, k]
                    print(f"  - {food_name}: {quantity:.2f}g")
        
        print("\nNutritional Summary:")
        nutrients = self.__calculate_nutrient_totals(solution)
        for nutrient, amount_g in nutrients.items():
            req = self.daily_req[nutrient]
            status = ""
            amount_original_unit_str, amount_original_unit = UnitConverter.convert_from_gram_to(amount_g, nutrient)
            if amount_g < req['min'] - (10 * req['min'] / 100): # Minuse 10%
                status = "\t => BELOW minimum"
            elif req['max'] != float('inf') and amount_g > req['max'] + (10 * req['min'] / 100): # add 10%
                status = "\t => ABOVE maximum"
            
            # For get Beautifull display
            if UnitConverter.get_default_unit(amount_original_unit_str) in ['kcal', 'g', 'None', '%aqr']:
                print(f"{nutrient}: {amount_original_unit_str} {status}")
            else:    
                print(f"{nutrient}: {amount_g:.6f}(g), {amount_original_unit_str} {status}")
        
        print(f"\nTotal Cost: {total_cost:.2f} DA")

    # Question 12
    def __initialize_population(self, population_size: int=10):
        """enerates a population of size=10 or 20 initial solutions using function created in 3"""
        return np.array([self.__correct_solution(self.__feasible_solution()) for _ in range(population_size)])
    
    # Question 13
    def interior_search(self, population_size: int=10, max_iterations: int=500) -> Tuple[np.ndarray, float]:
        """Implements an interior search algorithm."""
        # Generate initial population
        population = self.__initialize_population(population_size)
        fitness = np.array([self.__comparison_function(solution) for solution in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        best_cost = self.objective_function(best_solution)

        for iteration in range(max_iterations):
            for i in range(population_size):
                # Generate a new solution based on the current best
                new_solution = population[i] + np.random.uniform(-self.EPSILON, self.EPSILON, (self.N, len(self.MEALS))) * (best_solution - population[i])
                
                # Evaluate the new solution
                new_fitness = self.__comparison_function(new_solution)
                
                # Update the individual if the new solution is better
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                
                # Update the global best solution
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    best_cost = self.objective_function(best_solution)

            #print(f"Iteration {iteration + 1}/{max_iterations}, Best Cost: {best_cost}")
        return best_solution, best_cost

# Positive Points:
    # 1. Better complexity (runs quickly)
    # 2. Achieves the best cost Rq: (self.LAMBDA = 100 and violating is 10 in self.comparaison_function)
    # 3. Use POO

# Incv:
    # 1. qmaxi is added manually
    # 2. it hasn't been tested so potential errors in the future 
    # 3. Values in clories req like: '1,200' be '1.2'
        # Is fixed manually
    # 4. POO uncapsulation variable not uses
    # 5. don't handlinlg problems like in food itmes there are 'VitamineA' but in requirement there are 'VitaminA'
        # Is fixed manually

# Improvements:
    # 1. Added GUI
    # 2. Incorporated V(x) and E(x) in Constraint 1
    # 3. Add random function
