
# 3D Surface Plot
df[['views', 'reads', 'read_ratio']].iplot(
    kind='surface', title='Surface Plot')

# 3D Scatter Plot
df.iplot(x='word_count', y='views', z='fans', kind='scatter3d', xTitle='Word Count', yTitle='Views',
         zTitle='Fans', theme='pearl',
         categories='type', title='3D Scatter Plot by Type')

# 3D Bubble Chart
df.iplot(x='word_count', y='views', z='fans', kind='bubble3d', xTitle='Word Count', yTitle='Views',
         zTitle='Fans', theme='pearl', size='read_ratio',
         categories='type', title='3D Bubble Plot Sized by Read Ratio and Colored by Type')


# IPython Widgets
    import ipywidgets as widgets
    def show_date(date):
        return df[df['published_date'] >= pd.to_datetime(date)]

    date_selection = widgets.DatePicker(value=pd.to_datetime('2018-12-25'))
    widgets.interact(show_date, date=date_selection)

/Users/powersky/Documents/11FlatIronSchoolFolder/01DATASETUP
