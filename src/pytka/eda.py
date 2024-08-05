from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import pandas as pd
from .plots import make_html_filename, calculate_boxplot_stats
from .logging import step_time_calculation
class AutoEDA():

    def __init__(self, skip_cols, metrics, aggs, target_col, dataframe):
        self.skip_cols = skip_cols
        self.metrics = metrics
        self.aggs = aggs
        self.target_col = target_col
        self.dataframe = dataframe
        self.desired_cols = [col for col in self.dataframe.columns if
                             col not in self.skip_cols and col != self.target_col]

    @step_time_calculation('global_descriptive_stats')
    def global_descriptive_stats(self):

        plot_title = f'[EDA_0] Global feature descriptive stats dashboard'
        stats_df = self.dataframe[[col for col in self.dataframe.columns if col not in self.skip_cols]].describe().T[
            self.aggs]
        stats_df['feature'] = stats_df.index
        stats_df['n_unique'] = stats_df.apply(lambda x: len(self.dataframe[x['feature']].unique()), axis=1)
        stats_df['nan_count'] = stats_df.apply(lambda x: sum(self.dataframe[x['feature']].isna()), axis=1)
        del stats_df['feature']
        fig0 = px.bar(stats_df, barmode='group', title=plot_title)
        self.fig0 = fig0
        fig0.write_html(make_html_filename(plot_title))
        fig0.show()
        return fig0
        print(f'Plot {plot_title} generated successfully')

    @step_time_calculation('feature_boxplots')
    def feature_boxplots(self):

        plot_title = f'[EDA_1] Feature boxplots'

        total_cols = int(pow(len(self.desired_cols), 0.5))
        total_rows = int(len(self.desired_cols) / total_cols) + 1
        print(f"""Total features: {len(self.desired_cols)}""")
        print(f"""Grid size: {total_rows} x {total_cols}""")

        fig1 = make_subplots(rows=total_rows, cols=total_cols, subplot_titles=self.desired_cols)

        for feature_index, feature in enumerate(self.desired_cols):
            print(f"""Processing feature: {feature} which is {feature_index + 1} out of {len(self.desired_cols)}""")

            row_index, col_index = divmod(feature_index, total_cols)
            row_index += 1
            col_index += 1

            fig1.add_trace(calculate_boxplot_stats(self.dataframe[feature]),
                           row=row_index, col=col_index
                           )

        fig1.update_layout(height=total_rows * 400, width=total_cols * 400, title_text=plot_title)
        self.fig1 = fig1
        fig1.show()
        fig1.write_html(make_html_filename(plot_title))
        print(f'Plot {plot_title} generated successfully')

    @step_time_calculation('feature_correlations')
    def feature_correlations(self):
        plot_title = f'[EDA_2] Feature correlations'
        fig2 = px.imshow(self.dataframe[self.desired_cols].corr(), title=plot_title)
        self.fig2 = fig2
        fig2.show()
        fig2.write_html(make_html_filename(plot_title))
        print(f'Plot {plot_title} generated successfully')

    @step_time_calculation('feature_mean_target')
    def feature_mean_target(self):
        plot_title = f'[EDA_3] Feature mean target'

        feature_mean_target_summary = pd.DataFrame(columns=['mean_target', 'feature', 'feature_value'])

        cols = [col for col in self.dataframe.columns if col not in self.skip_cols and col != self.target_col]

        for feature in cols:
            temp_summary = pd.concat([self.dataframe[feature], self.dataframe[self.target_col]], axis=1).groupby(
                by=feature).mean()
            temp_summary['feature'] = feature
            temp_summary['feature_value'] = temp_summary.index
            temp_summary.columns = ['mean_target', 'feature', 'feature_value']
            feature_mean_target_summary = pd.concat([feature_mean_target_summary, temp_summary])
            print(f"""Feature: {feature} mean summary generated successfully""")
        feature_mean_target_summary = feature_mean_target_summary.reset_index()[
            ['feature', 'feature_value', 'mean_target']]
        fig3 = px.scatter(feature_mean_target_summary, x='feature_value', y='mean_target', color='feature',
                          trendline='ols', title=plot_title)
        # Store trendline coefficients in the class for future reference
        self.r_squared_per_feature_dict = {px.get_trendline_results(fig3)['feature'][index]: round(
            px.get_trendline_results(fig3).px_fit_results.iloc[index].rsquared, 2) for index in
                                           range(len(px.get_trendline_results(fig3).px_fit_results))}
        self.fig3 = fig3
        fig3.show()
        fig3.write_html(make_html_filename(plot_title))
        print(f'Plot {plot_title} generated successfully')

    @step_time_calculation('feature_linearity')
    def feature_linearity(self):
        plot_title = f'[EDA_4] Feature linearity'
        r_squared_per_feature_df = pd.DataFrame.from_dict([self.r_squared_per_feature_dict])
        r_squared_per_feature_df = r_squared_per_feature_df.melt(
            value_vars=[col for col in r_squared_per_feature_df.columns])
        r_squared_per_feature_df.columns = ['feature', 'r2_score']
        fig4 = px.bar(r_squared_per_feature_df.sort_values(by='r2_score', ascending=False), x='feature', y='r2_score',
                      color='feature', title=plot_title)
        self.fig4 = fig4
        fig4.write_html(make_html_filename(plot_title))
        fig4.show()
        print(f'Plot {plot_title} generated successfully')

    @step_time_calculation('target_balance')
    def target_balance(self):
        plot_title = f'[EDA_5] Target balance check'
        target_df = self.dataframe[self.target_col].value_counts().reset_index()
        target_df.columns = ['target_value', 'target_count']
        fig5 = px.bar(target_df, x='target_value', y='target_count', color='target_value')
        self.fig5 = fig5
        fig5.write_html(make_html_filename(plot_title))
        fig5.show()
        return fig5

    @step_time_calculation('detect_outliers')
    def detect_outliers(self, column, ratio_start=1, ratio_stop=1.6, precision=2):

        ratio_dict = {}

        Q3 = self.dataframe[column].quantile(0.75)
        Q1 = self.dataframe[column].quantile(0.25)

        for ratio in np.linspace(ratio_start, ratio_stop, 10):
            ratio = round(ratio, precision)
            IQR = Q3 - Q1
            lower_bound = round(Q1 - ratio * IQR, precision)
            upper_bound = round(Q3 + ratio * IQR, precision)

            print(f'Column: {column} expected range for ratio IQR +/- ({ratio} * STDDEV): <{lower_bound}, {upper_bound}>')

            data = self.dataframe[column]
            outside_range_mask = (data < lower_bound) | (data > upper_bound)

            values_outside_range = data[outside_range_mask]
            outliers_perc = round(len(values_outside_range) / len(data) * 100, precision)

            ratio_dict[ratio] = outliers_perc

        ratio_df = pd.DataFrame([ratio_dict]).T.reset_index()
        ratio_df.columns = ['ratio', 'perc_filtered']

        fig = px.bar(ratio_df, x='ratio', y='perc_filtered', color='ratio',
                     title=f'Filtered records in {column} with respect to ratio - multiplier of IQR ')
        fig.show()
        return ratio_df