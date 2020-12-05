import numpy as np
import pandas as pd

data = pd.read_csv("points.csv")
data.head()

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

init_notebook_mode(connected=True)
cf.go_offline()

data.iplot(kind='scatter',x='ga_p90',y='now_cost',
           mode='markers',text='web_name',size=10,
          xTitle='Points per 90',yTitle='Cost',title='Cost vs Points p90')