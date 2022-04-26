import matplotlib.pyplot as plt
import numpy as np

def my_plot(epochs, loss):
    plt.plot(epochs, loss)

def merge_log_files(path1, path2):
    log1 = read_log_file(path1)
    log2 = read_log_file(path2)
    log1.extend(log2)
    return log1

def read_log_file(log_path):
    log = []
    with open(log_path) as file:
        for line in file:
            line_info = line.split('\t')
            index, train_loss, val_loss = int(line_info[0]), float(line_info[1]), float(line_info[2].strip())
            log.append((train_loss, val_loss))
    return log



def visualize_train_log():
    log = read_log_file('important_logs/log1.txt')
    train_loss = [item[0] for item in log]
    val_loss = [item[1] for item in log]
    num_epochs = len(log)
    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), train_loss)
    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), val_loss)


    plt.show()


visualize_train_log()