from tensorflow.keras.callbacks import Callback
from pandas import DataFrame
import matplotlib.pyplot as plt


class HistorySaver(Callback):
    def __init__(self):
        super(HistorySaver, self).__init__()
        self.history_data = []

    def on_epoch_end(self, epoch, logs=None):
        current_logs = {"epoch": epoch}
        current_logs.update(logs)
        self.history_data.append(current_logs)
        history_df = DataFrame(self.history_data)
        history_df.loc[:, ["loss", "val_loss"]].plot()
        plt.savefig("loss.png")
        history_df.loc[:, ["accuracy", "val_accuracy"]].plot()
        plt.savefig("accuracy.png")
        plt.close()
