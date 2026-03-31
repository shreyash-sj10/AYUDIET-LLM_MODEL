# Fixed Enhanced Data Ingestion with Ayurvedic Integration
# File: data_ingestion_fixed_final.py

import os
import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Dict, List, Tuple, Optional
import json

class AyurvedicDataIngestion:
    """Enhanced data ingestion with Ayurvedic properties integration"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = Path(data_directory)
        self.datasets = {}
        self.knowledge_retriever = None
        
    def get_data_directory(self):
        """Get the data directory path, making it configurable"""
        data_dir = os.environ.get('DATA_DIRECTORY', str(self.data_directory))
        return Path(data_dir)

    def validate_csv_structure(self, df: pd.DataFrame, filename: str, required_columns: List[str] = None) -> bool:
        """Validate CSV structure and log information"""
        if df.empty:
            print(f"Warning: {filename} is empty")
            return False
        
        print(f"✓ {filename}: {len(df)} rows, {len(df.columns)} columns")
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                print(f"Warning: {filename} missing columns: {missing_cols}")
        
        return True

    def load_nutrition_databases(self) -> Dict[str, pd.DataFrame]:
        """Load all nutrition databases with enhanced validation"""
        data_dir = self.get_data_directory()
        
        csv_files = {
            "nin_fct": {
                "file": "NIN_fct.csv", 
                "required_cols": ["food_code", "food_name", "energy_kcal", "protein_g", "carb_g", "fat_g"]
            },
            "indb": {
                "file": "INDB.csv", 
                "required_cols": ["food_code", "food_name"]
            },
            "uk_fct": {
                "file": "UK_fct.csv", 
                "required_cols": ["food_code", "food_name"]
            },
            "us_fct": {
                "file": "US_fct.csv", 
                "required_cols": ["food_code", "food_name"]
            },
            "recipes": {
                "file": "recipes.csv", 
                "required_cols": ["recipe_code", "food_code"]
            },
            "recipes_names": {
                "file": "recipes_names.csv", 
                "required_cols": ["recipe_code", "recipe_name"]
            },
            "recipes_servingsize": {
                "file": "recipes_servingsize.csv", 
                "required_cols": ["recipe_code"]
            },
            "units": {
                "file": "Units.csv", 
                "required_cols": None
            },
            "indian_food": {
                "file": "IndianFoodDatasetXLS.csv", 
                "required_cols": None
            },
            "indb_ayurvedic": {
                "file": "INDB_ayurvedic.csv", 
                "required_cols": None
            }
        }
        
        loaded_datasets = {}
        
        try:
            print("Loading CSV datasets...")
            for key, config in csv_files.items():
                file_path = data_dir / config["file"]
                
                if not file_path.exists():
                    print(f"Warning: {config['file']} not found, skipping...")
                    continue
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Validate structure
                    if self.validate_csv_structure(df, config["file"], config["required_cols"]):
                        # Clean and preprocess common issues
                        if key in ['nin_fct', 'indb', 'uk_fct', 'us_fct']:
                            df = df.replace(['Tr', 'N', 'tr', 'n'], 0).fillna(0)
                            
                            # Convert numeric columns
                            numeric_columns = df.select_dtypes(include=['object']).columns
                            for col in numeric_columns:
                                if col not in ['food_name', 'food_code', 'recipe_name', 'recipe_code']:
                                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        
                        loaded_datasets[key] = df
                        
                except Exception as e:
                    print(f"Error loading {config['file']}: {e}")
                    continue
            
            if not loaded_datasets:
                print("Error: No CSV files could be loaded successfully.")
                return {}
            
            print(f"✓ Successfully loaded {len(loaded_datasets)} datasets")
            
        except Exception as e:
            print(f"Critical error during data loading: {e}")
            return {}
        
        return loaded_datasets

    def load_ayurvedic_knowledge_base(self) -> Optional[object]:
        """Load and process PDF documents for Ayurvedic knowledge"""
        data_dir = self.get_data_directory()
        
        print("\nLoading PDF documents...")
        docs = []
        
        try:
            # List of PDF files to load
            pdf_files = [
                "Charaka-Samhita-Acharya-Charaka.pdf",
                "Ayurveda_Nutrition_Guide.pdf",  # If available
                "Sushruta_Samhita.pdf",  # If available
                "Ashtanga_Hridaya.pdf"   # If available
            ]
            
            for pdf_file in pdf_files:
                pdf_path = data_dir / pdf_file
                
                if not pdf_path.exists():
                    print(f"Warning: {pdf_file} not found, skipping...")
                    continue
                
                try:
                    loader = PyPDFLoader(str(pdf_path))
                    pdf_docs = loader.load()
                    docs.extend(pdf_docs)
                    print(f"✓ Loaded {pdf_file}: {len(pdf_docs)} pages")
                    
                except Exception as e:
                    print(f"Error loading {pdf_file}: {e}")
                    continue
            
            if not docs:
                print("Warning: No PDF documents loaded. Knowledge retrieval may be limited.")
                # Create a minimal retriever with empty docs
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                from langchain_core.documents import Document
                dummy_doc = Document(page_content="Ayurveda knowledge base not available.")
                db = FAISS.from_documents([dummy_doc], embeddings)
                retriever = db.as_retriever()
                return retriever
            
            # Split documents into chunks - larger chunks for medical texts
            text_splitter = CharacterTextSplitter(
                chunk_size=1500,  # Increased for better context
                chunk_overlap=200,  # Increased overlap
                separator="\n\n"
            )
            
            final_documents = text_splitter.split_documents(docs)
            print(f"✓ Split into {len(final_documents)} text chunks")
            
            # Create embeddings and build the FAISS vector store
            print("Creating vector embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(final_documents, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 5})  # Return top 5 results
            
            print("✓ Vector store created successfully")
            return retriever
            
        except Exception as e:
            print(f"Error during PDF processing or embedding: {e}")
            return None

    def enhance_with_ayurvedic_properties(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhance food databases with Ayurvedic properties"""
        try:
            # Load Ayurvedic properties
            ayurvedic_properties = self._load_ayurvedic_properties()
            
            # Enhance main nutrition databases
            enhanced_datasets = {}
            for db_name, df in datasets.items():
                if db_name in ['nin_fct', 'indb', 'uk_fct', 'us_fct']:
                    enhanced_datasets[db_name] = self._add_ayurvedic_columns(df, ayurvedic_properties)
                else:
                    enhanced_datasets[db_name] = df
            
            # Copy other datasets as-is
            for db_name, df in datasets.items():
                if db_name not in enhanced_datasets:
                    enhanced_datasets[db_name] = df
            
            return enhanced_datasets
            
        except Exception as e:
            print(f"Error enhancing with Ayurvedic properties: {e}")
            return datasets

    def _load_ayurvedic_properties(self) -> Dict[str, Dict]:
        """Load comprehensive Ayurvedic properties for foods"""
        return {
            # Grains
            "rice": {
                "rasa": ["sweet"],
                "virya": "cooling",
                "vipaka": "sweet",
                "vata_effect": "balancing",
                "pitta_effect": "balancing", 
                "kapha_effect": "increasing",
                "digestibility": "easy",
                "best_season": "all",
                "meal_time": ["lunch", "dinner"]
            },
            "wheat": {
                "rasa": ["sweet"],
                "virya": "cooling",
                "vipaka": "sweet",
                "vata_effect": "balancing",
                "pitta_effect": "neutral",
                "kapha_effect": "increasing",
                "digestibility": "moderate",
                "best_season": "winter,spring",
                "meal_time": ["breakfast", "lunch"]
            },
            # Legumes
            "moong_dal": {
                "rasa": ["sweet", "astringent"],
                "virya": "cooling",
                "vipaka": "sweet",
                "vata_effect": "neutral",
                "pitta_effect": "balancing",
                "kapha_effect": "balancing",
                "digestibility": "easy",
                "best_season": "all",
                "meal_time": ["lunch", "dinner"]
            },
            "toor_dal": {
                "rasa": ["sweet", "astringent"],
                "virya": "cooling",
                "vipaka": "sweet",
                "vata_effect": "neutral",
                "pitta_effect": "neutral",
                "kapha_effect": "neutral",
                "digestibility": "moderate",
                "best_season": "all",
                "meal_time": ["lunch", "dinner"]
            },
            # Vegetables
            "spinach": {
                "rasa": ["sweet", "bitter", "astringent"],
                "virya": "cooling",
                "vipaka": "pungent",
                "vata_effect": "increasing",
                "pitta_effect": "balancing",
                "kapha_effect": "balancing",
                "digestibility": "easy",
                "best_season": "winter,spring",
                "meal_time": ["lunch", "dinner"]
            },
            "carrot": {
                "rasa": ["sweet"],
                "virya": "neutral",
                "vipaka": "sweet",
                "vata_effect": "balancing",
                "pitta_effect": "neutral",
                "kapha_effect": "neutral",
                "digestibility": "easy",
                "best_season": "winter",
                "meal_time": ["lunch", "dinner"]
            },
            # Spices
            "turmeric": {
                "rasa": ["bitter", "pungent"],
                "virya": "heating",
                "vipaka": "pungent",
                "vata_effect": "balancing",
                "pitta_effect": "neutral",
                "kapha_effect": "balancing",
                "digestibility": "easy",
                "best_season": "all",
                "meal_time": ["all"]
            },
            "ginger": {
                "rasa": ["pungent"],
                "virya": "heating",
                "vipaka": "sweet",
                "vata_effect": "balancing",
                "pitta_effect": "increasing",
                "kapha_effect": "balancing",
                "digestibility": "easy",
                "best_season": "winter,monsoon",
                "meal_time": ["all"]
            },
            # Fruits
            "apple": {
                "rasa": ["sweet", "astringent"],
                "virya": "cooling",
                "vipaka": "sweet",
                "vata_effect": "neutral",
                "pitta_effect": "balancing",
                "kapha_effect": "neutral",
                "digestibility": "easy",
                "best_season": "autumn,winter",
                "meal_time": ["breakfast", "snack"]
            },
            "banana": {
                "rasa": ["sweet"],
                "virya": "cooling",
                "vipaka": "sweet",
                "vata_effect": "balancing",
                "pitta_effect": "balancing",
                "kapha_effect": "increasing",
                "digestibility": "easy",
                "best_season": "all",
                "meal_time": ["breakfast", "snack"]
            }
        }

    def _add_ayurvedic_columns(self, df: pd.DataFrame, properties: Dict[str, Dict]) -> pd.DataFrame:
        """Add Ayurvedic property columns to dataframe"""
        enhanced_df = df.copy()
        
        # Add Ayurvedic property columns
        ayurvedic_columns = [
            'rasa', 'virya', 'vipaka', 'vata_effect', 'pitta_effect', 'kapha_effect',
            'digestibility', 'best_season', 'meal_time', 'ayurvedic_category'
        ]
        
        for col in ayurvedic_columns:
            enhanced_df[col] = None
        
        # Map foods to Ayurvedic properties
        for idx, row in enhanced_df.iterrows():
            food_name = str(row.get('food_name', '')).lower().strip()
            
            # Find matching Ayurvedic properties
            food_properties = self._find_food_properties(food_name, properties)
            
            if food_properties:
                enhanced_df.at[idx, 'rasa'] = ', '.join(food_properties.get('rasa', []))
                enhanced_df.at[idx, 'virya'] = food_properties.get('virya', '')
                enhanced_df.at[idx, 'vipaka'] = food_properties.get('vipaka', '')
                enhanced_df.at[idx, 'vata_effect'] = food_properties.get('vata_effect', '')
                enhanced_df.at[idx, 'pitta_effect'] = food_properties.get('pitta_effect', '')
                enhanced_df.at[idx, 'kapha_effect'] = food_properties.get('kapha_effect', '')
                enhanced_df.at[idx, 'digestibility'] = food_properties.get('digestibility', '')
                enhanced_df.at[idx, 'best_season'] = food_properties.get('best_season', '')
                enhanced_df.at[idx, 'meal_time'] = ', '.join(food_properties.get('meal_time', []))
                enhanced_df.at[idx, 'ayurvedic_category'] = self._categorize_food(food_properties)
        
        return enhanced_df

    def _find_food_properties(self, food_name: str, properties: Dict[str, Dict]) -> Optional[Dict]:
        """Find Ayurvedic properties for a food item"""
        # Direct match
        if food_name in properties:
            return properties[food_name]
        
        # Partial match
        for key, props in properties.items():
            if key in food_name or food_name in key:
                return props
        
        # Category-based default properties
        return self._get_default_properties(food_name)

    def _get_default_properties(self, food_name: str) -> Dict:
        """Get default Ayurvedic properties based on food category"""
        defaults = {
            "rasa": ["sweet"],
            "virya": "neutral",
            "vipaka": "sweet",
            "vata_effect": "neutral",
            "pitta_effect": "neutral",
            "kapha_effect": "neutral",
            "digestibility": "moderate",
            "best_season": "all",
            "meal_time": ["lunch"]
        }
        
        # Adjust defaults based on food type
        if any(grain in food_name for grain in ['rice', 'wheat', 'bread', 'pasta']):
            defaults['kapha_effect'] = 'increasing'
        elif any(veg in food_name for veg in ['spinach', 'kale', 'bitter']):
            defaults['rasa'] = ['bitter']
            defaults['virya'] = 'cooling'
        elif any(fruit in food_name for fruit in ['apple', 'orange', 'fruit']):
            defaults['rasa'] = ['sweet']
            defaults['virya'] = 'cooling'
        
        return defaults

    def _categorize_food(self, properties: Dict) -> str:
        """Categorize food based on Ayurvedic properties"""
        rasa = properties.get('rasa', [])
        virya = properties.get('virya', '')
        
        if 'sweet' in rasa and virya == 'cooling':
            return 'nourishing'
        elif 'bitter' in rasa or 'astringent' in rasa:
            return 'cleansing'
        elif 'pungent' in rasa and virya == 'heating':
            return 'stimulating'
        else:
            return 'balancing'

    def load_and_process_all_data(self) -> Tuple[Dict[str, pd.DataFrame], Optional[object]]:
        """
        Main method to load all datasets and create knowledge retriever.
        Returns a tuple of (data_dictionary, faiss_retriever).
        """
        try:
            # Load nutrition databases
            datasets = self.load_nutrition_databases()
            
            if not datasets:
                print("Error: Failed to load nutrition datasets")
                return {}, None
            
            # Enhance with Ayurvedic properties
            datasets = self.enhance_with_ayurvedic_properties(datasets)
            
            # Load knowledge base
            knowledge_retriever = self.load_ayurvedic_knowledge_base()
            
            # Store in instance
            self.datasets = datasets
            self.knowledge_retriever = knowledge_retriever
            
            return datasets, knowledge_retriever
            
        except Exception as e:
            print(f"Critical error in data processing: {e}")
            return {}, None

    def get_dataset_info(self):
        """Print information about loaded datasets"""
        if not self.datasets:
            print("No datasets loaded")
            return
        
        print(f"\n=== LOADED DATASETS INFO ===")
        for name, df in self.datasets.items():
            print(f"📊 {name.upper()}")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            
            # Show Ayurvedic columns if present
            ayurvedic_cols = [col for col in df.columns if col in ['rasa', 'virya', 'vipaka', 'vata_effect']]
            if ayurvedic_cols:
                print(f"   Ayurvedic Properties: {ayurvedic_cols}")
            print()

# Compatibility function for existing code
def load_and_process_data():
    """
    Compatibility function that matches the existing interface.
    Returns a tuple of (data_dictionary, faiss_retriever).
    """
    ingestion = AyurvedicDataIngestion()
    return ingestion.load_and_process_all_data()

if __name__ == "__main__":
    print("Testing enhanced data ingestion...")
    
    ingestion = AyurvedicDataIngestion()
    datasets, knowledge_retriever = ingestion.load_and_process_all_data()
    
    if datasets:
        ingestion.get_dataset_info()
        
    if knowledge_retriever:
        print("✓ Knowledge retriever ready")
    else:
        print("✗ Knowledge retriever failed")