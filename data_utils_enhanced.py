# Enhanced Data Utilities with Professional Functions
# File: data_utils_enhanced.py

import pandas as pd
import re
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import json

def get_conversion_factor(units_df: pd.DataFrame, from_unit: str, to_unit: str, food_name: str = "") -> float:
    """
    Enhanced conversion factor calculation with comprehensive unit support
    """
    try:
        if from_unit.lower() == to_unit.lower():
            return 1.0
        
        # Comprehensive conversion table
        conversions = {
            # Volume conversions
            ('tsp', 'g'): 5.0,
            ('tbsp', 'g'): 15.0,
            ('cup', 'g'): 240.0,  # for water/liquids
            ('cup', 'ml'): 240.0,
            ('ml', 'g'): 1.0,  # for water
            ('l', 'ml'): 1000.0,
            ('fl_oz', 'ml'): 29.5735,
            
            # Weight conversions
            ('kg', 'g'): 1000.0,
            ('lb', 'kg'): 0.453592,
            ('oz', 'g'): 28.35,
            
            # Indian measurements
            ('pav', 'g'): 250.0,  # 1/4 kg
            ('ser', 'kg'): 0.933,  # Traditional Indian measure
            ('maund', 'kg'): 37.32,
            
            # Cooking measurements
            ('pinch', 'g'): 0.5,
            ('dash', 'ml'): 0.6,
            ('drop', 'ml'): 0.05,
            
            # Spice measurements
            ('sprig', 'g'): 2.0,  # Fresh herbs
            ('bunch', 'g'): 100.0,  # Leafy vegetables
            ('clove', 'g'): 3.0,  # Garlic clove
        }
        
        # Direct conversion check
        conversion_key = (from_unit.lower(), to_unit.lower())
        if conversion_key in conversions:
            return conversions[conversion_key]
        
        # Reverse conversion check
        reverse_key = (to_unit.lower(), from_unit.lower())
        if reverse_key in conversions:
            return 1.0 / conversions[reverse_key]
        
        # Food-specific conversions
        food_conversions = get_food_specific_conversions(food_name, from_unit, to_unit)
        if food_conversions:
            return food_conversions
        
        # Try to use units_df if available
        if units_df is not None and not units_df.empty:
            conversion = lookup_in_units_table(units_df, from_unit, to_unit, food_name)
            if conversion:
                return conversion
        
        # Default fallback
        print(f"Warning: No conversion found for {from_unit} to {to_unit} for {food_name}. Using 1.0")
        return 1.0
        
    except Exception as e:
        print(f"Error in conversion: {e}")
        return 1.0

def get_food_specific_conversions(food_name: str, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Get food-specific conversion factors based on density and properties
    """
    food_name_lower = food_name.lower()
    
    # Food density mappings (cups to grams)
    food_densities = {
        # Grains and cereals
        'rice': {'cup': 195, 'tbsp': 12},
        'wheat_flour': {'cup': 120, 'tbsp': 7.5},
        'besan': {'cup': 110, 'tbsp': 7},
        'oats': {'cup': 80, 'tbsp': 5},
        'quinoa': {'cup': 170, 'tbsp': 11},
        
        # Legumes
        'moong_dal': {'cup': 200, 'tbsp': 12.5},
        'chana_dal': {'cup': 190, 'tbsp': 12},
        'masoor_dal': {'cup': 192, 'tbsp': 12},
        'toor_dal': {'cup': 200, 'tbsp': 12.5},
        
        # Oils and fats
        'oil': {'cup': 216, 'tbsp': 13.5, 'tsp': 4.5},
        'ghee': {'cup': 226, 'tbsp': 14, 'tsp': 4.7},
        'butter': {'cup': 227, 'tbsp': 14.2},
        
        # Spices and seasonings
        'turmeric': {'tsp': 2.2, 'tbsp': 6.6},
        'cumin': {'tsp': 2.1, 'tbsp': 6.3},
        'coriander': {'tsp': 1.8, 'tbsp': 5.4},
        'salt': {'tsp': 6, 'tbsp': 18},
        
        # Vegetables (approximate)
        'onion': {'cup': 160, 'medium': 150},
        'tomato': {'cup': 180, 'medium': 120},
        'potato': {'cup': 150, 'medium': 150},
        
        # Dairy
        'milk': {'cup': 244, 'tbsp': 15.25},
        'yogurt': {'cup': 245, 'tbsp': 15.3},
        'paneer': {'cup': 250, 'tbsp': 15.6},
    }
    
    # Find matching food
    matched_food = None
    for food_key in food_densities.keys():
        if food_key in food_name_lower or any(word in food_name_lower for word in food_key.split('_')):
            matched_food = food_key
            break
    
    if not matched_food:
        return None
    
    densities = food_densities[matched_food]
    
    # Convert from volume to weight
    if from_unit.lower() in densities and to_unit.lower() == 'g':
        return densities[from_unit.lower()]
    
    # Convert from weight to volume
    if from_unit.lower() == 'g' and to_unit.lower() in densities:
        return 1.0 / densities[to_unit.lower()]
    
    return None

def lookup_in_units_table(units_df: pd.DataFrame, from_unit: str, to_unit: str, food_name: str) -> Optional[float]:
    """
    Look up conversion in the units DataFrame
    """
    try:
        # This would depend on the actual structure of your units.csv
        # Adapt based on your file structure
        
        # Example implementation - adjust based on your CSV structure
        if 'from_unit' in units_df.columns and 'to_unit' in units_df.columns:
            matches = units_df[
                (units_df['from_unit'].str.lower() == from_unit.lower()) &
                (units_df['to_unit'].str.lower() == to_unit.lower())
            ]
            
            if not matches.empty:
                return float(matches.iloc[0]['conversion_factor'])
        
        return None
        
    except Exception as e:
        print(f"Error looking up in units table: {e}")
        return None

def calculate_recipe_nutrition(recipe_ingredients: List[Dict], food_databases: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate comprehensive nutritional information for a recipe
    """
    try:
        total_nutrition = {
            'energy_kcal': 0,
            'protein_g': 0,
            'carb_g': 0,
            'fat_g': 0,
            'fiber_g': 0,
            'calcium_mg': 0,
            'iron_mg': 0,
            'vitc_mg': 0,
            'vita_ug': 0
        }
        
        for ingredient in recipe_ingredients:
            food_code = ingredient.get('food_code')
            amount_g = ingredient.get('amount_g', 0)
            
            if not food_code or amount_g <= 0:
                continue
            
            # Find nutritional data for this ingredient
            nutrition_data = find_food_nutrition(food_code, food_databases)
            
            if nutrition_data:
                # Calculate proportional nutrition (nutrition per 100g * actual amount / 100)
                factor = amount_g / 100.0
                
                for nutrient in total_nutrition.keys():
                    if nutrient in nutrition_data:
                        total_nutrition[nutrient] += nutrition_data[nutrient] * factor
        
        return total_nutrition
        
    except Exception as e:
        print(f"Error calculating recipe nutrition: {e}")
        return total_nutrition

def find_food_nutrition(food_code: str, food_databases: Dict[str, pd.DataFrame]) -> Optional[Dict]:
    """
    Find nutritional data for a food item across multiple databases
    """
    try:
        # Search in order: NIN FCT, UK FCT, US FCT, INDB
        search_order = ['nin_fct', 'indb', 'uk_fct', 'us_fct']
        
        for db_name in search_order:
            if db_name in food_databases:
                df = food_databases[db_name]
                
                if df is not None and not df.empty and 'food_code' in df.columns:
                    matches = df[df['food_code'] == food_code]
                    
                    if not matches.empty:
                        return matches.iloc[0].to_dict()
        
        return None
        
    except Exception as e:
        print(f"Error finding food nutrition: {e}")
        return None

def get_ayurvedic_properties(food_name: str, ayurvedic_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Get Ayurvedic properties for a food item
    """
    try:
        properties = {
            'rasa': 'Unknown',
            'virya': 'Unknown', 
            'vipaka': 'Unknown',
            'vata_effect': 'Neutral',
            'pitta_effect': 'Neutral',
            'kapha_effect': 'Neutral',
            'digestibility': 'Moderate'
        }
        
        # If ayurvedic DataFrame is provided, search it first
        if ayurvedic_df is not None and not ayurvedic_df.empty:
            if 'food_name' in ayurvedic_df.columns:
                matches = ayurvedic_df[
                    ayurvedic_df['food_name'].str.contains(food_name, case=False, na=False)
                ]
                
                if not matches.empty:
                    match = matches.iloc[0]
                    for prop in properties.keys():
                        if prop in match:
                            properties[prop] = str(match[prop])
                    return properties
        
        # Fallback to default Ayurvedic properties
        return get_default_ayurvedic_properties(food_name)
        
    except Exception as e:
        print(f"Error getting Ayurvedic properties: {e}")
        return properties

def get_default_ayurvedic_properties(food_name: str) -> Dict[str, str]:
    """
    Get default Ayurvedic properties based on food categorization
    """
    food_lower = food_name.lower()
    
    # Grain properties
    if any(grain in food_lower for grain in ['rice', 'wheat', 'bread', 'roti']):
        return {
            'rasa': 'Sweet',
            'virya': 'Cooling',
            'vipaka': 'Sweet',
            'vata_effect': 'Balancing',
            'pitta_effect': 'Balancing',
            'kapha_effect': 'Increasing',
            'digestibility': 'Easy'
        }
    
    # Vegetable properties
    elif any(veg in food_lower for veg in ['spinach', 'kale', 'bitter', 'gourd']):
        return {
            'rasa': 'Bitter, Astringent',
            'virya': 'Cooling',
            'vipaka': 'Pungent',
            'vata_effect': 'Increasing',
            'pitta_effect': 'Balancing',
            'kapha_effect': 'Balancing',
            'digestibility': 'Moderate'
        }
    
    # Legume properties
    elif any(dal in food_lower for dal in ['dal', 'lentil', 'bean', 'chickpea']):
        return {
            'rasa': 'Sweet, Astringent',
            'virya': 'Cooling',
            'vipaka': 'Sweet',
            'vata_effect': 'Neutral',
            'pitta_effect': 'Balancing',
            'kapha_effect': 'Neutral',
            'digestibility': 'Moderate'
        }
    
    # Spice properties
    elif any(spice in food_lower for spice in ['ginger', 'garlic', 'chili', 'pepper']):
        return {
            'rasa': 'Pungent',
            'virya': 'Heating',
            'vipaka': 'Pungent',
            'vata_effect': 'Balancing',
            'pitta_effect': 'Increasing',
            'kapha_effect': 'Balancing',
            'digestibility': 'Easy'
        }
    
    # Default properties
    else:
        return {
            'rasa': 'Sweet',
            'virya': 'Neutral',
            'vipaka': 'Sweet',
            'vata_effect': 'Neutral',
            'pitta_effect': 'Neutral', 
            'kapha_effect': 'Neutral',
            'digestibility': 'Moderate'
        }

def format_nutrition_display(nutrition: Dict[str, float]) -> str:
    """
    Format nutrition data for user-friendly display
    """
    try:
        formatted = "🍎 **Nutritional Information (per serving):**\n\n"
        
        # Macronutrients
        formatted += "**Macronutrients:**\n"
        formatted += f"• Energy: {nutrition.get('energy_kcal', 0):.0f} kcal\n"
        formatted += f"• Protein: {nutrition.get('protein_g', 0):.1f}g\n"
        formatted += f"• Carbohydrates: {nutrition.get('carb_g', 0):.1f}g\n"
        formatted += f"• Fat: {nutrition.get('fat_g', 0):.1f}g\n"
        
        if nutrition.get('fiber_g', 0) > 0:
            formatted += f"• Fiber: {nutrition.get('fiber_g', 0):.1f}g\n"
        
        # Micronutrients
        formatted += "\n**Key Micronutrients:**\n"
        if nutrition.get('calcium_mg', 0) > 0:
            formatted += f"• Calcium: {nutrition.get('calcium_mg', 0):.0f}mg\n"
        if nutrition.get('iron_mg', 0) > 0:
            formatted += f"• Iron: {nutrition.get('iron_mg', 0):.1f}mg\n"
        if nutrition.get('vitc_mg', 0) > 0:
            formatted += f"• Vitamin C: {nutrition.get('vitc_mg', 0):.1f}mg\n"
        if nutrition.get('vita_ug', 0) > 0:
            formatted += f"• Vitamin A: {nutrition.get('vita_ug', 0):.0f}μg\n"
        
        return formatted
        
    except Exception as e:
        print(f"Error formatting nutrition display: {e}")
        return "Nutrition information unavailable"

def validate_user_input(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate user input data for consistency and reasonableness
    """
    errors = []
    
    try:
        # Age validation
        if 'age' in input_data and input_data['age']:
            age = input_data['age']
            if not isinstance(age, (int, float)) or age < 1 or age > 120:
                errors.append("Age must be between 1 and 120 years")
        
        # Weight validation
        if 'weight' in input_data and input_data['weight']:
            weight = input_data['weight']
            if not isinstance(weight, (int, float)) or weight < 10 or weight > 300:
                errors.append("Weight must be between 10 and 300 kg")
        
        # Height validation (basic)
        if 'height' in input_data and input_data['height']:
            height_str = str(input_data['height']).lower()
            if not re.search(r'\d+', height_str):
                errors.append("Height must contain numeric values")
        
        # Activity level validation
        if 'activity_level' in input_data and input_data['activity_level']:
            valid_levels = ['sedentary', 'moderate', 'active', 'very_active']
            if input_data['activity_level'].lower() not in valid_levels:
                errors.append(f"Activity level must be one of: {', '.join(valid_levels)}")
        
        # Dosha validation
        if 'primary_dosha' in input_data and input_data['primary_dosha']:
            valid_doshas = ['vata', 'pitta', 'kapha']
            if input_data['primary_dosha'].lower() not in valid_doshas:
                errors.append(f"Dosha must be one of: {', '.join(valid_doshas)}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors

def extract_recipe_ingredients(recipe_text: str) -> List[Dict[str, str]]:
    """
    Extract ingredients from recipe text using pattern matching
    """
    try:
        ingredients = []
        
        # Pattern to match ingredients with quantities
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(cup|cups|tbsp|tsp|kg|g|ml|l|piece|pieces|medium|large|small)\s+(.+)',
            r'(\d+(?:\.\d+)?)\s+(.+)',  # Just number and ingredient
            r'(a pinch of|a dash of|few)\s+(.+)',  # Qualitative amounts
        ]
        
        lines = recipe_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or len(line) < 5:
                continue
            
            # Try each pattern
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 3:
                        amount, unit, ingredient = match.groups()
                        ingredients.append({
                            'amount': amount,
                            'unit': unit.lower(),
                            'ingredient': ingredient.strip(),
                            'original_line': line
                        })
                        break
                    elif len(match.groups()) == 2:
                        amount_or_qual, ingredient = match.groups()
                        ingredients.append({
                            'amount': amount_or_qual,
                            'unit': 'piece' if amount_or_qual.isdigit() else 'qualitative',
                            'ingredient': ingredient.strip(),
                            'original_line': line
                        })
                        break
        
        return ingredients
        
    except Exception as e:
        print(f"Error extracting recipe ingredients: {e}")
        return []

def clean_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize data in nutritional databases
    """
    try:
        cleaned_df = df.copy()
        
        # Handle common data issues
        # Replace trace values and missing data indicators
        trace_values = ['Tr', 'tr', 'N', 'n', 'NA', 'na', '']
        for trace in trace_values:
            cleaned_df = cleaned_df.replace(trace, 0)
        
        # Convert object columns to numeric where appropriate
        numeric_patterns = ['_g', '_mg', '_ug', '_kcal', '_kj']
        
        for col in cleaned_df.columns:
            if any(pattern in col for pattern in numeric_patterns):
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)
        
        # Standardize food names
        if 'food_name' in cleaned_df.columns:
            cleaned_df['food_name'] = cleaned_df['food_name'].str.strip().str.title()
        
        # Remove duplicate entries based on food_code
        if 'food_code' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=['food_code'], keep='first')
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return df

# Export main functions
__all__ = [
    'get_conversion_factor',
    'calculate_recipe_nutrition', 
    'find_food_nutrition',
    'get_ayurvedic_properties',
    'format_nutrition_display',
    'validate_user_input',
    'extract_recipe_ingredients',
    'clean_and_standardize_data'
]