from src.data_processing import load_and_preprocess_data
from src.model_trainer import train_and_evaluate

def main():
    print("🚀 Initializing Course Recommendation ML Pipeline...")
    df = load_and_preprocess_data()
    train_and_evaluate(df)
    
if __name__ == "__main__":
    main()
