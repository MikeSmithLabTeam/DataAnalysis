import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from matplotlib.path import Path
import numpy as np

class SelectPts:
    """
    Select indices from a matplotlib plot using Lasso, rectangle or ellipse

    Selected indices are saved in the `ind_store` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    type: 'Lasso', 'Ellipse' or 'Rectangle'
    
    """

    def __init__(self, ax, collection, alpha_other=0.3, selector_type='Lasso',enter_closes=False):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.enter_closes=enter_closes
        self.alpha_other = alpha_other
        
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        if selector_type == 'Lasso':
            self.tool = LassoSelector(ax, onselect=self.onselect)
        elif selector_type == 'Rectangle':
            self.tool = RectangleSelector(ax, onselect=self.onselect_rect)
        elif type =='Ellipse':
            self.tool = EllipseSelector(ax, onselect=self.onselect_rect)

        self.ind = []

    def onselect_rect(self,click,release):
        x1, y1 = click.xdata, click.ydata
        x2, y2 = release.xdata, release.ydata
        verts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        self.onselect(verts)

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        if not hasattr(self, 'ind_store'):
            self.ind_store = self.ind.copy()
        else:
            self.ind_store = np.unique(np.concatenate((self.ind,self.ind_store)))
        
        
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind_store, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.tool.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        if self.enter_closes:
            plt.close()
    
    def reset(self):
        delattr(self, 'ind_store')
        self.fc[:, -1] = self.alpha_orig
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def get_pts(ax, pts_handle, selector_type='Lasso',enter_closes=False):
    """
    Call this function on a matplotlib axes to enable interactive
    selection of data.
    
    ax  : Matplotlib axes
    fig : Matplotlib figure
    pts_handle  : handle to some plotted data. Must be a collection. ax.scatter works but ax.plot won't
    
    Example Usage:
    #Create random data
    np.random.seed(19680801)
    data = np.random.rand(100, 2)
    #Plot data
    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)
    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    #Launch interactive data collection
    ind, values = get_pts(ax, pts,type='Ellipse',enter_closes=True)    
    
    
    """
    selector = SelectPts(ax, pts_handle, selector_type=selector_type, enter_closes=enter_closes)
    fig = ax.get_figure()
    def accept(event):
        if event.key == "enter":           
            pt_indices = selector.ind_store
            pt_coords = selector.xys[selector.ind_store]
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()
        elif event.key == "x":
            selector.reset()
            fig.canvas.mpl_connect("key_press_event", accept)

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    plt.show()
    pt_indices = selector.ind_store
    pt_values = selector.xys[selector.ind_store]
    return pt_indices, pt_values

