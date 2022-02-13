import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
################################################################################
def explanation(
    flux:'np.array',
    explanation:'lime-explanation',
    positive_only:'bool'=True,
    number_features:'int'=5,
    hide_rest:'bool'=False,
    show=False
    ):
####################################################################
    #Select the same class explained on the figures above.
    top_label =  explanation.top_labels[0]
    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[top_label])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
####################################################################
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=positive_only,
        num_features=number_features,
        hide_rest=hide_rest
        )
####################################################################
    fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
########################################################
    ax[0].imshow(flux)
    ax[0].set_title('GALAXY')
########################################################
    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title('Explanation')
########################################################
    ax[2].imshow(
        heatmap[:], cmap='RdBu',
        vmin=-heatmap.max(), vmax=heatmap.max()
        )

    ax[2].set_title('Heatmap')
########################################################
    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()

    if show:
        plt.show()
################################################################################
# #Plot. The visualization makes more sense if a symmetrical colorbar is used.
# plt.imshow(
#     heatmap[:],
#     cmap='RdBu',
#     vmin=-heatmap.max(),
#     vmax=heatmap.max()
#     )
#
# plt.colorbar()
# plt.show()
################################################################################
