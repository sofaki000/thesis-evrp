import os
import datetime

def get_day():
    now = datetime.datetime.now()
    return now.day

losses_dir = f"C:\\Users\\Lenovo\\Desktop\\my_evrp_final\\thesis-evrp\\training_results\\losses\\{get_day()}\\"
rewards_dir = f"C:\\Users\\Lenovo\\Desktop\\my_evrp_final\\thesis-evrp\\training_results\\rewards\\{get_day()}\\"

os.makedirs(losses_dir, exist_ok=True)
os.makedirs(rewards_dir, exist_ok=True)