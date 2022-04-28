import os
import time
import torch
import torch.utils.data as D
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from mlxtend.utils import check_Xy, format_kwarg_dictionaries
from mlxtend.plotting.decision_regions import get_feature_range_mask
import warnings
from itertools import cycle
from math import floor, ceil
from PIL import Image
from catalyst.data.sampler import BatchBalanceClassSampler, DistributedSamplerWrapper


class Clustered_Dataset(D.Dataset):
    def __init__(
        self,
        folder_path="./data/pasc/pasc-align-final-train",
        meta_path="data/pasc/test_meta.npy",
        transform=None,
        file_path=False,
        selected_class=None,
        transform_aux=None,
    ):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.data = np.load(meta_path)
        if selected_class:
            self.__filter__(selected_class)
        self.num_classes = len(np.unique(self.data["class"]))
        self.num_subset = len(np.unique(self.data["subset"]))
        self.num_subcluster = len(np.unique(self.data["subcluster"]))

        self.data_len = len(self.data)

        self.transform = transform
        self.transform_aux = transform_aux
        self.folder_path = folder_path
        self.file_path = file_path

    def __getitem__(self, index):
        file_path, cls_idx, subset_idx, subcluster_idx = self.data[index]
        img = Image.open(os.path.join(self.folder_path, file_path)).convert("RGB")
        label = np.array((cls_idx, subset_idx, subcluster_idx))

        if self.transform is not None:
            img = self.transform(img)

        if self.transform_aux is not None:
            img = self.transform_aux(img)

        if self.file_path:
            return (img, label, file_path)
        else:
            return (img, label)

    def __len__(self):
        return self.data_len

    def __filter__(self, selected_class):
        new_data = []
        for i in selected_class:
            new_data.append(self.data[self.data["class"] == i])
        self.data = np.concatenate(new_data)


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "model" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "batchnorm" in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def get_dataloader(args):
    # Data loading code
    print("Loading data")
    st = time.time()

    transform = T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = Clustered_Dataset(
        folder_path=args.folder_path,
        meta_path=args.train_meta_path,
        transform=transform,
        file_path=False,
    )
    dataset_test = Clustered_Dataset(
        folder_path=args.folder_path,
        meta_path=args.test_meta_path,
        transform=transform,
        file_path=False,
    )
    print("Took", time.time() - st)

    print("Creating data loaders")
    if args.distributed:
        labels = dataset.data["class"]
        train_sampler = DistributedSamplerWrapper(
            BatchBalanceClassSampler(
                labels, num_classes=8, num_samples=args.batch_size // 8
            ),
            num_replicas=2,
            shuffle=False,
        )
        test_sampler = D.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        labels = dataset.data["class"]
        assert args.batch_size % 8 == 0, "batch size can be divisible by 8"
        train_sampler = BatchBalanceClassSampler(
            labels, num_classes=8, num_samples=args.batch_size // 8
        )
        test_sampler = D.SequentialSampler(dataset_test)

    dataloader = D.DataLoader(
        dataset, batch_sampler=train_sampler, num_workers=args.workers, pin_memory=True
    )

    dataloader_test = D.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )

    return dataloader, dataloader_test


def serialize_target(target, num_subcluster=None, num_subset=None):
    return (
        target[:, 0] * num_subcluster * num_subset
        + target[:, 1] * num_subcluster
        + target[:, 2]
    )


def evaluate_l2(features, centers, targets, subcenters_available=False):
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim) or (num_classes, feat_dim)
        batch_size = targets.size(0)
        if subcenters_available:
            num_classes, num_subcenters, feat_dim = centers.shape
            num_centers = num_classes * num_subcenters
        else:
            num_classes, feat_dim = centers.shape
            num_centers = num_classes

        serialized_centers = centers.view(-1, feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = (
            torch.pow(features, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, num_centers)
            + torch.pow(serialized_centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(num_centers, batch_size)
            .t()
        )
        distmat.addmm_(features, serialized_centers.t(), beta=1, alpha=-2)
        # distmat in shape (batch_size,num_centers)

        if subcenters_available:
            preds = distmat.argmin(1) // num_subcenters
        else:
            preds = distmat.argmin(1)

        correct = preds.eq(targets)
        accuracy = correct.flatten().sum(dtype=torch.float32) * (100.0 / batch_size)
        return preds, accuracy


def visualize(feat, labels, centers, num_classes, epoch):
    colors = [
        "#ff0000",
        "#ffff00",
        "#00ff00",
        "#00ffff",
        "#0000ff",
        "#ff00ff",
        "#990000",
        "#999900",
        "#009900",
        "#009999",
        "#004499",
        "#009944",
    ]

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    for i in range(num_classes):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(
            feat[labels == i, 0].mean(),
            feat[labels == i, 1].mean(),
            str(i),
            color="black",
            fontsize=12,
        )
        ax.text(centers[i, 0], centers[i, 1], "c" + str(i), color="black", fontsize=12)
    # ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    # ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()

    fig.savefig("./epoch_%d.jpg" % epoch)


def plot_features(
    X,
    y,
    feature_index=None,
    filler_feature_values=None,
    filler_feature_ranges=None,
    ax=None,
    X_highlight=None,
    res=None,
    zoom_factor=1.0,
    legend=1,
    hide_spines=False,
    markers="sss^^^oooxxxvvv<>",
    colors=(
        "#d62728,#3ca02c,#1f77b4,"
        "#9467bd,#ff7f0e,#8c564b,#e377c2,"
        "#7f7f7f,#bcbd22,#17becf"
    ),
    scatter_kwargs=None,
    contourf_kwargs=None,
    scatter_highlight_kwargs=None,
):
    """Plot decision regions of a classifier.

    Please note that this functions assumes that class labels are
    labeled consecutively, e.g,. 0, 1, 2, 3, 4, and 5. If you have class
    labels with integer labels > 4, you may want to provide additional colors
    and/or markers as `colors` and `markers` arguments.
    See http://matplotlib.org/examples/color/named_colors.html for more
    information.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Feature Matrix.
    y : array-like, shape = [n_samples]
        True class labels.
    clf : Classifier object.
        Must have a .predict method.
    feature_index : array-like (default: (0,) for 1D, (0, 1) otherwise)
        Feature indices to use for plotting. The first index in
        `feature_index` will be on the x-axis, the second index will be
        on the y-axis.
    filler_feature_values : dict (default: None)
        Only needed for number features > 2. Dictionary of feature
        index-value pairs for the features not being plotted.
    filler_feature_ranges : dict (default: None)
        Only needed for number features > 2. Dictionary of feature
        index-value pairs for the features not being plotted. Will use the
        ranges provided to select training samples for plotting.
    ax : matplotlib.axes.Axes (default: None)
        An existing matplotlib Axes. Creates
        one if ax=None.
    X_highlight : array-like, shape = [n_samples, n_features] (default: None)
        An array with data points that are used to highlight samples in `X`.
    res : float or array-like, shape = (2,) (default: None)
        This parameter was used to define the grid width,
        but it has been deprecated in favor of
        determining the number of points given the figure DPI and size
        automatically for optimal results and computational efficiency.
        To increase the resolution, it's is recommended to use to provide
        a `dpi argument via matplotlib, e.g., `plt.figure(dpi=600)`.
    zoom_factor : float (default: 1.0)
        Controls the scale of the x- and y-axis of the decision plot.
    hide_spines : bool (default: True)
        Hide axis spines if True.
    legend : int (default: 1)
        Integer to specify the legend location.
        No legend if legend is 0.
    markers : str (default: 's^oxv<>')
        Scatterplot markers.
    colors : str (default: 'red,blue,limegreen,gray,cyan')
        Comma separated list of colors.
    scatter_kwargs : dict (default: None)
        Keyword arguments for underlying matplotlib scatter function.
    contourf_kwargs : dict (default: None)
        Keyword arguments for underlying matplotlib contourf function.
    scatter_highlight_kwargs : dict (default: None)
        Keyword arguments for underlying matplotlib scatter function.

    Returns
    ---------
    ax : matplotlib.axes.Axes object

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/

    """

    check_Xy(X, y, y_int=True)  # Validate X and y arrays
    dim = X.shape[1]

    if ax is None:
        ax = plt.gca()

    if res is not None:
        warnings.warn(
            "The 'res' parameter has been deprecated."
            "To increase the resolution, it's is recommended"
            "to use to provide a `dpi argument via matplotlib,"
            "e.g., `plt.figure(dpi=600)`.",
            DeprecationWarning,
        )

    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError("X_highlight must be a NumPy array or None")
        else:
            plot_testdata = False
    elif len(X_highlight.shape) < 2:
        raise ValueError("X_highlight must be a 2D array")

    if feature_index is not None:
        # Unpack and validate the feature_index values
        if dim == 1:
            raise ValueError("feature_index requires more than one training feature")
        try:
            x_index, y_index = feature_index
        except ValueError:
            raise ValueError(
                "Unable to unpack feature_index. Make sure feature_index "
                "only has two dimensions."
            )
        try:
            X[:, x_index], X[:, y_index]
        except IndexError:
            raise IndexError(
                "feature_index values out of range. X.shape is {}, but "
                "feature_index is {}".format(X.shape, feature_index)
            )
    else:
        feature_index = (0, 1)
        x_index, y_index = feature_index

    # Extra input validation for higher number of training features
    if dim > 2:
        if filler_feature_values is None:
            raise ValueError(
                "Filler values must be provided when "
                "X has more than 2 training features."
            )

        if filler_feature_ranges is not None:
            if not set(filler_feature_values) == set(filler_feature_ranges):
                raise ValueError(
                    "filler_feature_values and filler_feature_ranges must "
                    "have the same keys"
                )

        # Check that all columns in X are accounted for
        column_check = np.zeros(dim, dtype=bool)
        for idx in filler_feature_values:
            column_check[idx] = True
        for idx in feature_index:
            column_check[idx] = True
        if not all(column_check):
            missing_cols = np.argwhere(~column_check).flatten()
            raise ValueError(
                "Column(s) {} need to be accounted for in either "
                "feature_index or filler_feature_values".format(missing_cols)
            )

    marker_gen = cycle(list(markers))

    n_classes = np.unique(y).shape[0]
    colors = colors.split(",")
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]

    # Get minimum and maximum
    x_min, x_max = (
        X[:, x_index].min() - 1.0 / zoom_factor,
        X[:, x_index].max() + 1.0 / zoom_factor,
    )
    if dim == 1:
        y_min, y_max = -1, 1
    else:
        y_min, y_max = (
            X[:, y_index].min() - 1.0 / zoom_factor,
            X[:, y_index].max() + 1.0 / zoom_factor,
        )

    xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
    xnum, ynum = floor(xnum), ceil(ynum)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, num=xnum), np.linspace(y_min, y_max, num=ynum)
    )

    contourf_kwargs_default = {"alpha": 0.45, "antialiased": True}
    contourf_kwargs = format_kwarg_dictionaries(
        default_kwargs=contourf_kwargs_default,
        user_kwargs=contourf_kwargs,
        protected_keys=["colors", "levels"],
    )
    ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])

    # Scatter training data samples
    # Make sure scatter_kwargs has backwards compatible defaults
    scatter_kwargs_default = {"alpha": 0.8, "edgecolor": "black"}
    scatter_kwargs = format_kwarg_dictionaries(
        default_kwargs=scatter_kwargs_default,
        user_kwargs=scatter_kwargs,
        protected_keys=["c", "marker", "label"],
    )
    for idx, c in enumerate(np.unique(y)):
        if dim == 1:
            y_data = [0 for i in X[y == c]]
            x_data = X[y == c]
        elif dim == 2:
            y_data = X[y == c, y_index]
            x_data = X[y == c, x_index]
        elif dim > 2 and filler_feature_ranges is not None:
            class_mask = y == c
            feature_range_mask = get_feature_range_mask(
                X,
                filler_feature_values=filler_feature_values,
                filler_feature_ranges=filler_feature_ranges,
            )
            y_data = X[class_mask & feature_range_mask, y_index]
            x_data = X[class_mask & feature_range_mask, x_index]
        else:
            continue

        ax.scatter(
            x=x_data,
            y=y_data,
            c=colors[idx],
            marker=next(marker_gen),
            label=c,
            **scatter_kwargs
        )

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if dim == 1:
        ax.axes.get_yaxis().set_ticks([])

    if plot_testdata:
        if dim == 1:
            x_data = X_highlight
            y_data = [0 for i in X_highlight]
        elif dim == 2:
            x_data = X_highlight[:, x_index]
            y_data = X_highlight[:, y_index]
        else:
            feature_range_mask = get_feature_range_mask(
                X_highlight,
                filler_feature_values=filler_feature_values,
                filler_feature_ranges=filler_feature_ranges,
            )
            y_data = X_highlight[feature_range_mask, y_index]
            x_data = X_highlight[feature_range_mask, x_index]

        # Make sure scatter_highlight_kwargs backwards compatible defaults
        scatter_highlight_defaults = {
            "c": "",
            "edgecolor": "black",
            "alpha": 1.0,
            "linewidths": 1,
            "marker": "o",
            "s": 80,
        }
        scatter_highlight_kwargs = format_kwarg_dictionaries(
            default_kwargs=scatter_highlight_defaults,
            user_kwargs=scatter_highlight_kwargs,
        )
        ax.scatter(x_data, y_data, **scatter_highlight_kwargs)

    if legend:
        if dim > 2 and filler_feature_ranges is None:
            pass
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, framealpha=0.3, scatterpoints=1, loc=legend)

    return ax


def plot_features_3d(
    X,
    y,
    num_classes,
    ax=None,
    X_highlight=None,
    zoom_factor=1.0,
    legend=1,
    hide_spines=False,
    markers="s^ox^^^oooxxxvvv<>",
    colors=(
        "#d62728,#3ca02c,#e377c2,"
        "#bcbd22,#ff7f0e,#8c564b,#e377c2,"
        "#7f7f7f,#bcbd22,#bcbd22,#17cccf,#9e571b,#9f57fb"
    ),
    scatter_kwargs=None,
    contourf_kwargs=None,
    scatter_highlight_kwargs=None,
):
    """Plot decision regions of a classifier.

    Please note that this functions assumes that class labels are
    labeled consecutively, e.g,. 0, 1, 2, 3, 4, and 5. If you have class
    labels with integer labels > 4, you may want to provide additional colors
    and/or markers as `colors` and `markers` arguments.
    See http://matplotlib.org/examples/color/named_colors.html for more
    information.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Feature Matrix.
    y : array-like, shape = [n_samples]
        True class labels.
    ax : matplotlib.axes.Axes (default: None)
        An existing matplotlib Axes. Creates
        one if ax=None.
    X_highlight : array-like, shape = [n_samples, n_features] (default: None)
        An array with data points that are used to highlight samples in `X`.
    res : float or array-like, shape = (2,) (default: None)
        This parameter was used to define the grid width,
        but it has been deprecated in favor of
        determining the number of points given the figure DPI and size
        automatically for optimal results and computational efficiency.
        To increase the resolution, it's is recommended to use to provide
        a `dpi argument via matplotlib, e.g., `plt.figure(dpi=600)`.
    zoom_factor : float (default: 1.0)
        Controls the scale of the x- and y-axis of the decision plot.
    hide_spines : bool (default: True)
        Hide axis spines if True.
    legend : int (default: 1)
        Integer to specify the legend location.
        No legend if legend is 0.
    markers : str (default: 's^oxv<>')
        Scatterplot markers.
    colors : str (default: 'red,blue,limegreen,gray,cyan')
        Comma separated list of colors.
    scatter_kwargs : dict (default: None)
        Keyword arguments for underlying matplotlib scatter function.
    contourf_kwargs : dict (default: None)
        Keyword arguments for underlying matplotlib contourf function.
    scatter_highlight_kwargs : dict (default: None)
        Keyword arguments for underlying matplotlib scatter function.

    Returns
    ---------
    ax : matplotlib.axes.Axes object

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/

    """

    check_Xy(X, y, y_int=True)  # Validate X and y arrays
    dim = X.shape[1]

    x_index, y_index, z_index = (0, 1, 2)

    marker_gen = cycle(list(markers))

    n_classes = num_classes
    colors = colors.split(",")
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]
    markers = [next(marker_gen) for c in range(n_classes)]

    contourf_kwargs_default = {"alpha": 0.45, "antialiased": True}
    contourf_kwargs = format_kwarg_dictionaries(
        default_kwargs=contourf_kwargs_default,
        user_kwargs=contourf_kwargs,
        protected_keys=["colors", "levels"],
    )

    # Scatter training data samples
    # Make sure scatter_kwargs has backwards compatible defaults
    scatter_kwargs_default = {"alpha": 0.8, "edgecolor": "black"}
    scatter_kwargs = format_kwarg_dictionaries(
        default_kwargs=scatter_kwargs_default,
        user_kwargs=scatter_kwargs,
        protected_keys=["c", "marker", "label"],
    )
    available_classes = np.unique(y)
    for idx, c in enumerate(range(num_classes)):
        if c in available_classes:
            pass
        else:
            continue

        z_data = X[y == c, z_index]
        y_data = X[y == c, y_index]
        x_data = X[y == c, x_index]

        ax.scatter(
            xs=x_data,
            ys=y_data,
            zs=z_data,
            c=colors[idx],
            marker=markers[idx],
            label=c,
            **scatter_kwargs
        )

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')

    # if legend:
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.legend(handles, labels,
    #                 framealpha=0.3, scatterpoints=1, loc='center left')

    return ax
