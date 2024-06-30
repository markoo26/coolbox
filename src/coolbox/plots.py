def make_html_filename(plot_title):
    """
        Setup a filename for .write_html method of Plotly.

        Parameters
        ----------
        plot_title : string
            User friendly plot name to be shown in Plotly figure

        Returns
        ---------

            Suitable, space-free title with .html extension
    """
    return plot_title.lower().replace(' ', '_') + '.html'
