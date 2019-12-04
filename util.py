
def find_closest(imgs, dataset):
    closest_imgs = [None] * len(imgs)
    min_l2 = [None] * len(imgs)
    for img in imgs:
        for sample in dataset:

def flatten(x):
    return x.view(x.size(0), -1)

def unflatten(x, imsize):
    return x.view(x.size(0), imsize[0], imsize[1])

def show_imgs(imgs, in_row=3, figsize=(5,5), save=False, name=None):
    n = len(imgs)
    fig, axs = plt.subplots((n + in_row - 1)// in_row, in_row, figsize=figsize)
    for i in range(n):
        if (n + in_row - 1) // in_row == 1:
            cur_ax = axs[i % in_row]
        else:
            cur_ax = axs[i//in_row, i % in_row]
        cur_ax.axis("off")
        cur_ax.imshow(imgs[i].numpy().squeeze(), cmap="gray")
    plt.show()

def plot_history(train_history, eval_history):
    if len(train_history) == 0:
        return
    metrics = train_history.keys()
    n = len(metrics)

    fig, ax = plt.subplots((n + 1) // 2, 2, figsize=(15, 10))

    for i, metric in enumerate(metrics):
        train_arr = train_history[metric]
        eval_arr = eval_history[metric]

        step = len(train_arr) // len(eval_arr)
        start = step - 1
        end = len(train_arr)


        ax[i // 2, i % 2].plot(list(range(len(train_arr))), train_arr, label='train')
        ax[i // 2, i % 2].scatter(list(range(start, end, step)),
                                  eval_arr, c='r', zorder=3, label='eval')
        ax[i // 2, i % 2].set_title(metric)
        ax[i // 2, i % 2].set_xlabel('train steps')
        ax[i // 2, i % 2].set_ylabel(metric)
        ax[i // 2, i % 2].legend()
    plt.show()
