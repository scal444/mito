import matplotlib.pyplot as plt


def plot_3_regimes(cyl_x, cyl_y, junc_x, junc_y, flat_x, flat_y, mito_shape):
    fig, axarr = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0)

    axarr[0].scatter(cyl_x, cyl_y)
    axarr[0].set_xlim(0, mito_shape.l_cylinder / 2)
    axarr[0].set_xlabel("cylinder coordinate (nm)")
    axarr[0].set_ylabel("force (kJ/mol)")

    axarr[1].scatter(junc_x, junc_y)
    axarr[1].set_xlim(0, 90)
    axarr[1].set_xlabel("junctino coordinate (degrees)")

    axarr[2].scatter(flat_x, flat_y)
    axarr[2].set_xlim(0, mito_shape.l_cylinder / 2)
    axarr[2].set_xlabel("cylinder coordinate (nm)")

    plt.show()
