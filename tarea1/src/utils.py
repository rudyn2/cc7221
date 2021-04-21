import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def plot_metrics(train_loss: list, train_acc: list, test_loss: list, test_acc: list, fig_name: str):
    assert len(train_loss) == len(train_acc)
    assert len(train_acc) == len(test_loss)
    assert len(test_loss) == len(test_acc)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    epochs = list(range(1, len(train_loss) + 1))
    axs[0].plot(epochs, train_loss, '--o', label="Train")
    axs[0].plot(epochs, test_loss, '--o', label="Validation")
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].plot(epochs, train_acc, '--o', label="Train")
    axs[1].plot(epochs, test_acc, '--o', label="Validation")
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(fig_name, dpi=200)
    plt.show()

    return fig


if __name__ == '__main__':
    import numpy as np
    t1 = np.random.rand(10)
    t2 = np.random.rand(10)
    t3 = np.random.rand(10)
    t4 = np.random.rand(10)

    plot_metrics(t1, t2, t3, t4, 'test_fig.png')