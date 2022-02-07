import matplotlib.pyplot as plt

def plot_images(images, res, fname=""):
    scales = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 8, 256: 4, 512: 2, 1024: 1}
    scale = scales[res]

    grid_col = min(images.shape[0], int(32 // scale))
    grid_row = 1

    f, axarr = plt.subplots(
        grid_row, grid_col, figsize=(grid_col * scale, grid_row * scale)
    )

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col], vmin=-1, vmax=1)
            ax[col].axis("off")
    plt.show()
    if fname:
        f.savefig(fname)