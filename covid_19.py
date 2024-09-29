'''
Download Covid_19_India Dataset named“covid_19_india.csv”from
 https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_19_india.csvand perform the following,
 with proper annotations of the legend and axes labels:
 Usecovid_19_india.csvto do the following:
 ● ForeachIndianstate, findmaximumcasesreportedforconfirmed,deathsandrecoveredindividuallyalong
 withdateonwhichthesecaseswerereportedforanythreemonthsofyear2020.Displaytheresult inthe
 self-explanatory format.
 ● Useappropriateyear-monthstringdateconversionsforexample: Identifytheno.ofcasesonthe6thdayof
 the month by converting year-month string to dates.
 ● Createsubplots(linegraph) forshowingtotalnumberofcuredcasesmonth-wisefromApril2020toMarch
 2021 in four states namely Karnataka, Gujarat, Haryana,and Uttar Pradesh.
 ● Compare thedeaths due toCovid-19 in themonthsofMay2020andMay2021 for thestatesnamely
 Karnataka, Delhi, and Madhya Pradesh using stackedbars.
 ● Makeagraph toshowthemonthwise relation (Positive/Negative/Neutral) betweennumberofconfirmed
 Covid-19 cases and Deaths in Uttar Pradesh. Displaycorrelation value too in the graph
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('covid_19_india.csv')

# Convert the 'Date' column to datetime for easier manipulation
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Task 1: Find Maximum Cases for Each State (Confirmed, Deaths, Recovered)

# Filter data for three months of 2020 (April, May, June)
df_2020 = df[(df['Date'] >= '2020-04-01') & (df['Date'] <= '2020-06-30')]

# Group by state and find max for confirmed, deaths, and recovered along with date
max_cases = df_2020.groupby('State/UnionTerritory').agg({
    'Confirmed': ['max', lambda x: df_2020.loc[x.idxmax(), 'Date']],
    'Deaths': ['max', lambda x: df_2020.loc[x.idxmax(), 'Date']],
    'Cured': ['max', lambda x: df_2020.loc[x.idxmax(), 'Date']],
}).reset_index()

print("Max cases for each state:")
print(max_cases)


# Task 2: Identify Cases on the 6th Day of Each Month

# Create a Year-Month column for easy filtering
df['Year-Month'] = df['Date'].dt.to_period('M').astype(str)

# Filter data for the 6th day of each month
df_6th_day = df[df['Date'].dt.day == 6]

print("Cases on the 6th day of the month:")
print(df_6th_day[['Date', 'State/UnionTerritory', 'Confirmed', 'Deaths', 'Cured']])


# Task 3: Subplots for Cured Cases (April 2020 to March 2021)

# Filter data for the required states and time range
states = ['Karnataka', 'Gujarat', 'Haryana', 'Uttar Pradesh']
df_apr20_mar21 = df[(df['Date'] >= '2020-04-01') & (df['Date'] <= '2021-03-31') & df['State/UnionTerritory'].isin(states)]

# Create a column with Year-Month for grouping
df_apr20_mar21['Year-Month'] = df_apr20_mar21['Date'].dt.to_period('M')

# Create subplots for each state
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, state in enumerate(states):
    state_data = df_apr20_mar21[df_apr20_mar21['State/UnionTerritory'] == state]
    monthwise_cured = state_data.groupby('Year-Month')['Cured'].sum()
    
    axes[i].plot(monthwise_cured.index.astype(str), monthwise_cured.values)
    axes[i].set_title(f'{state} - Cured Cases')
    axes[i].set_xlabel('Month')
    axes[i].set_ylabel('Cured Cases')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Task 4: Compare Deaths in May 2020 and May 2021 (Karnataka, Delhi, Madhya Pradesh)

# Filter data for May 2020 and May 2021
states_may = ['Karnataka', 'Delhi', 'Madhya Pradesh']
df_may = df[((df['Date'] >= '2020-05-01') & (df['Date'] <= '2020-05-31')) | ((df['Date'] >= '2021-05-01') & (df['Date'] <= '2021-05-31'))]
df_may = df_may[df_may['State/UnionTerritory'].isin(states_may)]

# Create separate datasets for May 2020 and May 2021
df_may_2020 = df_may[df_may['Date'].dt.year == 2020].groupby('State/UnionTerritory')['Deaths'].sum().reset_index()
df_may_2021 = df_may[df_may['Date'].dt.year == 2021].groupby('State/UnionTerritory')['Deaths'].sum().reset_index()

# Merge the data
df_may_compare = df_may_2020.merge(df_may_2021, on='State/UnionTerritory', suffixes=('_2020', '_2021'))

# Plot stacked bar chart
df_may_compare.set_index('State/UnionTerritory')[['Deaths_2020', 'Deaths_2021']].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Deaths Comparison: May 2020 vs May 2021')
plt.xlabel('State')
plt.ylabel('Number of Deaths')
plt.show()


# Task 5: Correlation Between Confirmed Cases and Deaths in Uttar Pradesh

# Filter data for Uttar Pradesh
up_data = df[df['State/UnionTerritory'] == 'Uttar Pradesh']

# Group by Year-Month
up_data['Year-Month'] = up_data['Date'].dt.to_period('M')
up_monthly = up_data.groupby('Year-Month').agg({'Confirmed': 'sum', 'Deaths': 'sum'}).reset_index()

# Calculate correlation
correlation = up_monthly['Confirmed'].corr(up_monthly['Deaths'])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(up_monthly['Year-Month'].astype(str), up_monthly['Confirmed'], label='Confirmed Cases', color='blue')
plt.plot(up_monthly['Year-Month'].astype(str), up_monthly['Deaths'], label='Deaths', color='red')

# Add correlation text
plt.text(0.5, 0.5, f'Correlation: {correlation:.2f}', fontsize=12, transform=plt.gca().transAxes)

# Annotate the plot
plt.title('COVID-19 Confirmed Cases vs Deaths in Uttar Pradesh (Month-wise)')
plt.xlabel('Month')
plt.ylabel('Number of Cases/Deaths')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
