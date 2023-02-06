import datetime

from matplotlib import pyplot as plt

from config import losses_dir,rewards_dir


def get_filename_time():
    now = datetime.datetime.now()
    return f'm={now.month}_d={now.day}_h={now.hour}_m={now.minute}'



def plot_losses_and_rewards(losses_per_epochs, rewards_per_epochs):
    time = get_filename_time()
    # epochs finished
    plt.plot(losses_per_epochs)
    plt.savefig(f"{losses_dir}losses{time}.png")
    plt.clf()
    plt.plot(rewards_per_epochs)
    plt.savefig(f"{rewards_dir}rewards{time}.png")