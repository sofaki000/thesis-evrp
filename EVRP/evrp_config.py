import os
import datetime

def get_day():
    now = datetime.datetime.now()
    return now.day

evrp_losses_dir = f"losses\\{get_day()}\\"
evrp_rewards_dir = f"rewards\\{get_day()}\\"

os.makedirs(evrp_losses_dir, exist_ok=True)
os.makedirs(evrp_rewards_dir, exist_ok=True)