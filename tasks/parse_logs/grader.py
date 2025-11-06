import os
import json

def check_solution():
    """
    Grader for the 'parse_logs' task.
    Checks if 'results.json' exists and contains the correct data.
    """
    
    if not os.path.exists("results.json"):
        return False, "Failure: `results.json` file was not created."

    try:
        with open("results.json", 'r') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failure: `results.json` is not valid JSON. Error: {e}"

    required_keys = ["best_epoch", "train_loss", "val_loss"]
    if not all(key in data for key in required_keys):
        return False, f"Failure: `results.json` is missing one or more keys. Required: {required_keys}"
        
    
    expected_epoch = 3
    expected_train_loss = 0.22
    expected_val_loss = 0.15
    
    try:
        if not isinstance(data["best_epoch"], int):
            return False, "Failure: `best_epoch` is not an integer."
            
        if data["best_epoch"] != expected_epoch:
            return False, f"Failure: `best_epoch` is incorrect. Expected {expected_epoch}, got {data['best_epoch']}."

        
        if not abs(data["train_loss"] - expected_train_loss) < 0.001:
            return False, f"Failure: `train_loss` is incorrect. Expected {expected_train_loss}, got {data['train_loss']}."
            
        if not abs(data["val_loss"] - expected_val_loss) < 0.001:
            return False, f"Failure: `val_loss` is incorrect. Expected {expected_val_loss}, got {data['val_loss']}."

    except Exception as e:
        return False, f"Failure: Error during value validation. {e}"

    
    return True, "Success: `results.json` is correct."