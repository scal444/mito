import matplotlib.pyplot as plt


def plot_dummy_forces_spatial(inner_data, outer_data, cmax, cmap='Reds', title=None):
    plt.figure()
    plt.scatter(inner_data.coordinates.rho, inner_data.coordinates.z, vmin=0, vmax=cmax, c=4 * inner_data.force, cmap=cmap)
    plt.scatter(outer_data.coordinates.rho, outer_data.coordinates.z, vmin=0, vmax=cmax, c=4 * outer_data.force, cmap=cmap)
    plt.xlabel("rho (nm)")
    plt.ylabel("z (nm)")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


def plot_lipid_occupancy_spatial(inner_data, inner_coords, outer_data, outer_coords, mindelta=-5, maxdelta=5, cmap='seismic',
                                 save=None):
    plt.figure()
    plt.scatter(inner_coords.rho, inner_coords.z, c=inner_data, vmin=mindelta, vmax=maxdelta, cmap=cmap)
    plt.scatter(outer_coords.rho, outer_coords.z, c=outer_data, vmin=mindelta, vmax=maxdelta, cmap=cmap)
    plt.xlabel("Rho (nm)")
    plt.ylabel("z (nm)")
    plt.colorbar(label="% change")

    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_lipid_occupancy_heatmap(data, mindelta=-5, maxdelta=5, cmap='seismic',
                                 xticks=None, xticklabels=None,
                                 yticks=None, yticklabels=None,
                                 xdelimiters=None, ydelimiters=None,
                                 xlabel=None, ylabel=None, save=None):
    '''
        mindelta, maxdelta - set the color scaling min and max for colormap
        xticks, yticks - where to mark labels - be aware of what the x and y are displaying, ie whether you've inverted
        xlabels, ylabels - same size as xticks, yticks - the text that is labeled
        x/y delimiters - draws a dotted line at the locations specified, for e.g. delimiting sections of the mitochondria
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=mindelta, vmax=maxdelta, cmap=cmap, origin='lower', extent=[0, data.shape[1], 0, data.shape[0]])
    # setting ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xdelimiters is not None:
        for delim in xdelimiters:
            ax.plot([delim, delim], [0, data.shape[0]], 'k--')

    if ydelimiters is not None:
        for delim in ydelimiters:
            ax.plot([0, data.shape[1]], [ delim, delim], 'k--')

    ax.figure.colorbar(im, ax=ax, label="% change")

    if save is not None:
        plt.savefig(save)
    plt.show()
