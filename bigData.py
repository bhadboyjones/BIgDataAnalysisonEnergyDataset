import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Extract the data
pd.set_option('display.max_columns', None, 'display.max_rows', None)
df = pd.read_excel(r'C:\Users\jones\PycharmProjects\Energy_data\bigDataDemand_step1.xlsx', skiprows=22)
# print(df)

# Finding the seasonal: average/max/min load factors with locations as different columns
locations = ['Location #1', 'Location #2', 'Location #3', 'Location #4', 'Location #5']

# Create a new data frame
lf = pd.DataFrame(columns=locations)

#  illustrations for finding mean.max,min values for all weathers
check_if_winter = df['Season'] == 'winter'
df_winter = df[check_if_winter]
df_winter[locations]
average = df_winter[locations].mean()
lf.loc['average winter', :] = average

# Average load factor for each season
lf.loc['average winter', :] = df[df['Season'] == 'winter'][locations].mean()
lf.loc['average spring', :] = df[df['Season'] == 'spring'][locations].mean()
lf.loc['average summer', :] = df[df['Season'] == 'summer'][locations].mean()
lf.loc['average autumn', :] = df[df['Season'] == 'autumn'][locations].mean()

# Max load factor
lf.loc['max winter', :] = df[df['Season'] == 'winter'][locations].max()
lf.loc['max spring', :] = df[df['Season'] == 'spring'][locations].max()
lf.loc['max summer', :] = df[df['Season'] == 'summer'][locations].max()
lf.loc['max autumn', :] = df[df['Season'] == 'autumn'][locations].max()

# Min load factor
lf.loc['min winter', :] = df[df['Season'] == 'winter'][locations].min()
lf.loc['min spring', :] = df[df['Season'] == 'spring'][locations].min()
lf.loc['min summer', :] = df[df['Season'] == 'summer'][locations].min()
lf.loc['min autumn', :] = df[df['Season'] == 'autumn'][locations].min()
# print(lf)

# Computing the daily average/mean load factor
# Create an array for days in a year
daily_lf = pd.DataFrame(columns=locations)
days_in_year = np.arange(1, 366)
df[df['number of day'] == 1]

for k in days_in_year:
    daily_lf.loc[k, :] = df[df['number of day'] == k][locations].mean()
daily_lf.reset_index(inplace=True)
daily_lf.rename({'index': 'number of day'}, axis=1, inplace=True)
# print(daily_lf)

# Classification of days in the year
'''
1-59 and 335-365 : winter 
60-151 : Spring 
152 -243 : summer 
244-334 : autumn
'''

# Create array to comprise specific days of the year for each season
winter_days1 = np.arange(1, 60)
winter_days2 = np.arange(335, 366)
winter_days = np.append(winter_days1, winter_days2)

spring_days = np.arange(60, 152)
summer_days = np.arange(152, 244)
autumn_days = np.arange(244, 335)


# A function to specify season name based on the number of day that comes in the year
def apply_season(x):  # x will be a row of a dataframe. from this row we pick the element
    # at column 'number of day'
    if x['number of day'] in winter_days:
        return 'winter'
    elif x['number of day'] in spring_days:
        return 'spring'
    elif x['number of day'] in summer_days:
        return 'summer'
    else:
        return 'autumn'


# print(daily_lf.apply(lambda x: apply_season(x), axis=1))

daily_lf['seasons'] = daily_lf.apply(lambda x: apply_season(x), axis=1)  # axis=1 means row.
# rearrange columns
daily_lf = daily_lf[['seasons', 'number of day', 'Location #1',
                     'Location #2', 'Location #3', 'Location #4', 'Location #5']]
# print(daily_lf)

# Absolute difference between daily average and seasonal average load factors
daily_lf_diff = pd.DataFrame(columns=locations)  # (we create an empty dataframe with columns locations
# (Location#1 -Location#5, as previously defined)

# Create a column for 365 days
daily_lf_diff['number of day'] = np.arange(1, 366, 1)  # new column 'no of day' populated with numerics
# 1-365
daily_lf_diff['seasons'] = ''  # new empty 'season' column

# Organise the order of columns
daily_lf_diff = daily_lf_diff[
    ['seasons', 'number of day', 'Location #1',
     'Location #2', 'Location #3', 'Location #4', 'Location #5']]

daily_lf_diff['seasons'] = daily_lf_diff.apply(lambda x: apply_season(x), axis=1)  # apply function


# Function to get absolute difference btw daily_lf and seasonal average load factor
# x here represents another dataset to be applied in this case the ds is daily_lf.
# check where the function is called for better remembrance
def absolute_difference(x):
    if x["seasons"] == "winter":
        x[locations] = x[locations] - lf.loc['average winter', locations]
    if x["seasons"] == "spring":
        x[locations] = x[locations] - lf.loc['average spring', locations]
    if x["seasons"] == "summer":
        x[locations] = x[locations] - lf.loc['average summer', locations]
    if x["seasons"] == "autumn":
        x[locations] = x[locations] - lf.loc['average autumn', locations]
    x[locations] = abs(x[locations])
    return x


daily_lf_backup = daily_lf  # saves daily_lf

# Apply the function apply_season(x) to daily_lf_diff
daily_lf_diff = daily_lf.apply(lambda x: absolute_difference(x), axis=1)
# print(daily_lf_diff)

# Minimum error per season
daily_lf_diff['seasons'] == 'winter'  # returns bool value of rows where column seasons = winter
daily_lf_diff[daily_lf_diff['seasons'] == 'winter']  # returns the values
# print(daily_lf_diff[daily_lf_diff['seasons'] == 'winter'][locations].min())

# new dataframe
min_error = pd.DataFrame(columns=locations)
min_error.loc['min error for winter'] = ''  # creates a new row
min_error.loc['min error for winter'] = daily_lf_diff[daily_lf_diff['seasons'] == 'winter'].min()

# do same for spring, summer & autumn
min_error.loc['min error for spring'] = daily_lf_diff[daily_lf_diff['seasons'] == 'spring'].min()
min_error.loc['min error for summer'] = daily_lf_diff[daily_lf_diff['seasons'] == 'summer'].min()
min_error.loc['min error for autumn'] = daily_lf_diff[daily_lf_diff['seasons'] == 'autumn'].min()
# print(min_error)

# Find particular day with the minimum error

# output line 152: series of True & False values for where exact match occurs.
# sum(x) = 1, cos only 1 value = True
daily_lf_diff['Location #1'] == min_error.loc['min error for winter', 'Location #1']

# Output line 156: all rows values for dataset daily_lf_diff, focus on column Location#1 that equals row
# 'min error for winter' and column location#1 in dataset min_err.
daily_lf_diff.loc[daily_lf_diff['Location #1'] == min_error.loc['min error for winter',
                                                                'Location #1']]

# returns exact day & its index column
daily_lf_diff.loc[daily_lf_diff['Location #1'] == min_error.loc['min error for winter',
                                                                'Location #1']]['number of day']
# output line 163: returns exact number of day in an array
daily_lf_diff.loc[daily_lf_diff['Location #1'] == min_error.loc['min error for winter',
                                                                'Location #1']]['number of day'].values
# Output line 166: Returns just number of day
daily_lf_diff.loc[daily_lf_diff['Location #1'] == min_error.loc['min error for winter',
                                                                'Location #1']]['number of day'].values[0]

daily_lf_diff.loc[daily_lf_diff['Location #1'] == min_error.loc['min error for spring',
                                                                'Location #1']]['number of day'].values[0]


# Making a Function to handle the code
def find_day(dataf, season, location):
    return dataf.loc[dataf[location] == min_error.loc['min error for ' + season, location]]['number of day'].values[0]


# print(find_day(daily_lf_diff, 'spring', 'Location #1'))

# Automating processes to reduce code line
day_of_min_err_winter = []
day_of_min_err_spring = []
day_of_min_err_summer = []
day_of_min_err_autumn = []

for Location in locations:
    day_of_min_err_winter.append(find_day(daily_lf_diff, 'winter', Location))
    day_of_min_err_spring.append(find_day(daily_lf_diff, 'spring', Location))
    day_of_min_err_summer.append(find_day(daily_lf_diff, 'summer', Location))
    day_of_min_err_autumn.append(find_day(daily_lf_diff, 'autumn', Location))
# print(day_of_min_err_winter)
# print(day_of_min_err_spring)
# print(day_of_min_err_summer)
# print(day_of_min_err_autumn)

# create new row in min_error dataframe
min_error.loc['day of min error, in winter'] = day_of_min_err_winter
min_error.loc['day of min error, in spring'] = day_of_min_err_spring
min_error.loc['day of min error, in summer'] = day_of_min_err_summer
min_error.loc['day of min error, in autumn'] = day_of_min_err_autumn
# print(min_error)
# print(set(min_error.loc['day of min error, in winter'].values))

# Testing if the day is in the correct season
# print(set(min_error.loc['day of min error, in winter'].values).issubset(winter_days))
# print(set(min_error.loc['day of min error, in spring'].values).issubset(spring_days))
# print(set(min_error.loc['day of min error, in summer'].values).issubset(summer_days))
# print(set(min_error.loc['day of min error, in autumn'].values).issubset(autumn_days))

# Retrieving the corresponding 24hr-ly values for the days that is minimum-error in all seasons
abc = pd.DataFrame(columns=locations)
abc['Hours'] = np.arange(1, 25)
abc.set_index('Hours', inplace=True)
# print(min_error.loc['day of min error, in winter']['Location #1'])

# Check day
check_day = df['number of day'] == min_error.loc['day of min error, in winter']['Location #1']
# print(check_day)
# print(df[check_day]['Location #1'])
our_lf = df[check_day]['Location #1'].values  # converting column 'Location #1' from series to a
# numpy array to enable us place the values in the data frame abc

abc['Location #1'] = our_lf  # arrangement matters here cos if inverted our_lf would not populate


# the column abc['Location #1']
# print(abc)


# create function to automate process
def Dataframe_with_min_error(season):
    x = pd.DataFrame(columns=locations)
    x['Hours'] = np.arange(1, 25)
    x.set_index('Hours', inplace=True)
    x = x[['Location #1', 'Location #2', 'Location #3', 'Location #4', 'Location #5']]
    x['Location #1'] = df[df['number of day'] == min_error.loc['day of min error, in ' + season][0]][
        'Location #1'].values
    x['Location #2'] = df[df['number of day'] == min_error.loc['day of min error, in ' + season][1]][
        'Location #2'].values
    x['Location #3'] = df[df['number of day'] == min_error.loc['day of min error, in ' + season][2]][
        'Location #3'].values
    x['Location #4'] = df[df['number of day'] == min_error.loc['day of min error, in ' + season][3]][
        'Location #4'].values
    x['Location #5'] = df[df['number of day'] == min_error.loc['day of min error, in ' + season][4]][
        'Location #5'].values
    return x


df_winter_min_error = Dataframe_with_min_error('winter')
df_spring_min_error = Dataframe_with_min_error('spring')
df_summer_min_error = Dataframe_with_min_error('summer')
df_autumn_min_error = Dataframe_with_min_error('autumn')

# print(df_winter_min_error)
# print(df_spring_min_error)
# print(df_summer_min_error)
# print(df_autumn_min_error)

# ILLUSTRATION: REMEMBER abc DEFINED ABOVE, USED TO SEARCH FOR THE PARTICULAR HOUR
# print(abc['Location #1'].min())
# print(abc['Location #1'] == abc['Location #1'].min())  # Output 261: returns set of True or False value
# for when statement is satisfied.
abc[abc['Location #1'] == abc['Location #1'].min()]  # returns the row of abc with values.
abc[abc['Location #1'] == abc['Location #1'].max()].index  # returns type of index column
abc[abc['Location #1'] == abc['Location #1'].max()].index[0]  # returns specified location


# APPLICATION
def find_min_hours(q, location):
    position_of_min_value = q[q[location] == q[location].min()].index[0]
    return position_of_min_value


def find_max_hours(q, location):
    position_of_max_value = q[q[location] == q[location].max()].index[0]
    return position_of_max_value


def append_position(q):
    position_of_min_value = []
    position_of_max_value = []

    for k in locations:
        findmin = find_min_hours(q, k)
        findmax = find_max_hours(q, k)
        position_of_min_value.append(findmin)
        position_of_max_value.append(findmax)

    q.loc['Min'] = q.min()
    q.loc['Max'] = q.max()
    q.loc['position of min value'] = position_of_min_value
    q.loc['position of max value'] = position_of_max_value

    return q


# calling the function append_position(q) q represents a dataframe
df_winter_min_error = append_position(df_winter_min_error)
# output 290: append_positions(q) to the dataframe 'df_winter_min_error' reassigning
# the dataframe df_winter_min_error.
df_spring_min_error = append_position(df_spring_min_error)
df_summer_min_error = append_position(df_summer_min_error)
df_autumn_min_error = append_position(df_autumn_min_error)
# print(df_winter_min_error)
# print(df_spring_min_error)
# print(df_summer_min_error)
# print(df_autumn_min_error)

# TYPICAL DAY
# defining A
# note: A = winter_average - (winter_min + winter_max)/24
A = lf.loc['average winter'] - (lf.loc['min winter'] + lf.loc['max winter'])/24
A
# illustrations
df_winter_min_error.iloc[:24, :]  # we use iloc since index is integer
df_winter_min_error.iloc[:24, :].sum(axis=0)  # sums all columns of dataframe df_winter_min_error,
# from start till index 24 and all rows of the data frame
df_winter_min_error.loc['Min']  # returns the row 'Min' of called df

# Defining B
B = df_winter_min_error.iloc[:24, :].sum(axis=0) - df_winter_min_error.loc['Min']\
    - df_winter_min_error.loc['Max']
B
# Defining a constant k
k = 24*A/B
k
# create a dataframe called factors comprised of A, b and k for each location
factors = pd.DataFrame([A, B, k])
factors.index = ['A', 'B', 'k']
# print(factors)
# Create a new dataframe called winter1
winter1 = pd.DataFrame(columns=locations)
winter1['Hours'] = np.arange(1, 25)
winter1.set_index('Hours', inplace=True)
# print(factors.loc['k', locations])
factors.loc['k', locations].values  # changes the row from series to a numpy array

# Fill the values with k for the corresponding location in winter1
winter1[locations] = factors.loc['k', locations].values
# print(winter1)

# create a dataframe minmaxhours comprised of position of hours in the day with minimum error
minmaxhours = df_winter_min_error.iloc[-2:, :]
# print(minmaxhours)

for col in winter1.columns:
    mask = winter1.index.isin(minmaxhours[col].astype(int))  # because minmaxhours is a series
    # and we cannot convert a series into an integer we use the method as type to convert values to int
    winter1[col][mask] = 1

# for loop explained
m = winter1.index.isin(minmaxhours['Location #1'].astype(int))  # returns an array of True or False
# value depending on if condition is satisfied
# print(mask)
# print(winter1[locations])  # returns all columns defined under locations.
# print(winter1[locations][m])  # returns the values that satisfy the condition as True

# Alternatively:
winter1.loc[2, 'Location #1'] = 1
winter1.loc[19, 'Location #1'] = 1

winter1.loc[4, 'Location #2'] = 1
winter1.loc[19, 'Location #2'] = 1

winter1.loc[5, 'Location #3'] = 1
winter1.loc[15, 'Location #3'] = 1

winter1.loc[5, 'Location #4'] = 1
winter1.loc[20, 'Location #4'] = 1

winter1.loc[7, 'Location #5'] = 1
winter1.loc[20, 'Location #5'] = 1
# print(winter1)

# Create a new dataframe called winter2
winter2 = pd.DataFrame(columns=locations)
winter2['Hours'] = np.arange(1, 25)
winter2.set_index('Hours', inplace=True)
winter2[locations] = df_winter_min_error.iloc[:24, :]

winter2.loc[2, 'Location #1'] = lf.loc['min winter', 'Location #1']
winter2.loc[19, 'Location #1'] = lf.loc['max winter', 'Location #1']

winter2.loc[4, 'Location #2'] = lf.loc['min winter', 'Location #2']
winter2.loc[19, 'Location #2'] = lf.loc['max winter', 'Location #2']

winter2.loc[5, 'Location #3'] = lf.loc['min winter', 'Location #3']
winter2.loc[15, 'Location #3'] = lf.loc['max winter', 'Location #3']

winter2.loc[5, 'Location #4'] = lf.loc['min winter', 'Location #4']
winter2.loc[20, 'Location #4'] = lf.loc['max winter', 'Location #4']

winter2.loc[7, 'Location #5'] = lf.loc['min winter', 'Location #5']
winter2.loc[20, 'Location #5'] = lf.loc['max winter', 'Location #5']
# print(winter2)

df_winter_typical_day = winter1 * winter2
# print(df_winter_typical_day)

# Plotting
df_winter_typical_day.plot()
plt.title('Load factor for typical day in winter')
# plt.show()

# Checking to see we retained key information from original df
print(df_winter_typical_day.mean())
print(lf.loc['average winter'])
print(df_winter_typical_day.min())
print(df_winter_typical_day.max() == lf.loc['max winter'])