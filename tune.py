import optuna
import yaml
import copy
from train import train_model

def objective(trial):
    # 1. Load Base Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Suggest Hyperparameters (Round 2: Structure - Regression)
    # Fixed Best Params from Round 1
    config['learning_rate'] = 8.4e-4
    config['num_layers'] = 1
    config['dim_feedforward'] = 128
    config['d_model'] = 64
    
    # Tuning Targets
    config['kernel_size'] = trial.suggest_categorical('kernel_size', [1, 3, 5, 7])
    config['window_size'] = trial.suggest_categorical('window_size', [10, 20, 30, 40, 60])
    # Re-tune dropout slightly
    config['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    
    # Force Regression Task
    config['task_type'] = 'regression'
    config['epochs'] = 20  # Shorten epochs for tuning
    
    print(f"\n--- Trial {trial.number} ---")
    print(f"Params: {trial.params}")
    
    # 3. Run Training
    # Disable Neptune logging for individual trials to avoid spamming runs
    best_val_loss, best_val_acc = train_model(config, run_neptune=False)
    
    # Optimization Goal: Minimize Validation Loss (MSE)
    return best_val_loss

def main():
    print("Starting Hyperparameter Tuning (Regression - Round 2) with Optuna...")
    
    study = optuna.create_study(direction='minimize') # Minimize Loss
    study.optimize(objective, n_trials=20)
    
    print("\n--- Tuning Complete ---")
    print("Best Params:", study.best_params)
    print("Best Value (Val Loss):", study.best_value)
    
    # Save best params to a file
    with open('best_params_regression_r1.yaml', 'w') as f:
        yaml.dump(study.best_params, f)

if __name__ == "__main__":
    main()
