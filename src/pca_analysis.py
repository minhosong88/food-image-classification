from sklearn.decomposition import PCA
import numpy as np


def perform_pca(images, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(images.copy())
    eigenfood = pca.components_.reshape((n_components, 512, 512))
    return pca, eigenfood


def plot_explained_variance(pca, title):
    import plotly
    import plotly.graph_objects as go
    from plotly.graph_objs import Bar, Scatter, Layout
    from plotly.graph_objs.layout import XAxis, YAxis, Annotation

    explained_var = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(explained_var)
    final_annotation = go.layout.Annotation(
        x=len(explained_var),
        y=cum_var_exp[-1],
        text='{:.2f}'.format(cum_var_exp[-1] * 100),
        showarrow=False,
        xanchor='left',
        yanchor='bottom'
    )

    plotly.offline.iplot({
        "data": [
            Bar(y=explained_var, name='individual explained variance'),
            Scatter(y=cum_var_exp, name='cumulative explained variance')
        ],
        "layout": Layout(
            title=title,
            xaxis=XAxis(title='Principal components'),
            yaxis=YAxis(title='Explained variance ratio'),
            annotations=[final_annotation]
        )
    })
