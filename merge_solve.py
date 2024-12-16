import numpy as np      # 
import pandas as pd
import random
import math

class DietOptimization:
    def __init__(self, food_data_path):
        # Load data
        self.food_data = pd.read_csv(food_data_path)
        
        # Constants and Parameters
        self.N = len(self.food_data)  # Number of food items (I)
        self.MAX_TOTAL_QUANTITY = 60  # Maximum total food quantity per day
        self.MEALS = 3 # (K)
        self.CATEGORIES = (self.food_data['Type'].str.capitalize()).unique() # Get unique Category (G)
        self.LAMBDA = 100 # Penalty for violating soft constraints (𝜆)
        
        # Nutritional requirements (Bj)
        self.NUTRIENT_REQUIREMENTS = {
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

        self.MAX_REQUIREMENTS = {
            'Calories(kcal)': 2500,
            'Protein (g)': 150,
            'Carbohydrates (g)': 350,
            'Fat (g)': 100,
            'Fiber (%\AQR)': 45,
            'Iron (mg)': 18,
            'Folate (mcg)': 500,
            'Sodium (mg)': 2900,
            'VitaminA (UI)': 5500,
            'VitaminC (mg)': 140
        }

    def feasible_solution(self):
        """
        Function Generate a random feasible solution. Q3
        2D array: [food_item_index, meal_index, quantity]
        """