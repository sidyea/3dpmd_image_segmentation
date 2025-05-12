
import json
import pandas as pd
import datetime

def append_results(results, metrics, i, left, right):
    """
    Append the results to the list.
    
    Args:
        metrics (dict): Dictionary containing intensity metrics.
        i (int): Frame index.
        left (float): Left edge location.
        right (float): Right edge location.
    
    Returns:
        list: Updated results list.
    """

    results.append({
        "frame": i,
        "mean": metrics["mean"],
        "std": metrics["std"],
        "max": metrics["max"],
        "min": metrics["min"],
        "median": metrics["median"],
        "left": left,
        "right": right
    })
    
    return results

def save_results_to_csv(results, path, filename):
    """
    Save the results to a CSV file.
    
    Args:
        results (list): List of results to save.
        path (str): Path to save the CSV file.
        filename (str): Name of the CSV file.
    """
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}"

    # Save the results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(f"{path}/{filename}.csv", index=False)
    print(f"Results saved to {path}/{filename}.csv")


def save_results_to_json(results, path, filename):
    """
    Save the results to a JSON file.
    
    Args:
        results (list): List of results to save.
        path (str): Path to save the JSON file.
        filename (str): Name of the JSON file.
    """
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}"

    # Save the results to a JSON file
    with open(f"{path}/{filename}.json", "w") as f:
        json.dump(results, f)
    print(f"Results saved to {path}/{filename}.json")