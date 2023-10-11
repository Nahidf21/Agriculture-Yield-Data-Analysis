import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns  # only for data visualization

precipitation_data= pd.read_csv("average precipitation.csv")
pesticides_data= pd.read_csv("pesticides.csv")
temperature_data= pd.read_csv("Temperature.csv")
yield_data= pd.read_csv("Yield1.csv")

# Check the data states 
precipitation_data.head()
precipitation_data.drop_duplicates(inplace=True)
precipitation_data.drop(columns="Code", inplace=True)

precipitation_data = precipitation_data[(precipitation_data["Year"] >= 1990)]
precipitation_data.head(20)
precipitation_data.rename(columns={"Entity": "Country", "Year":"Year", "Average precipitation in depth (mm per year)": "Average precipitation(mm per year)"}, inplace= True)
precipitation_data.head()

# Check wehich countries names are different from the from our main 3 data sets, we collect our datasets from two 
#different websites. 
set(precipitation_data["Country"])-set(temperature_data["Area"])

replace_country_name= {'Bolivia' :"Bolivia (Plurinational State of)" ,
 'Brunei' : "Brunei Darussalam",
 'Cape Verde' : "Cabo Verde",
 "Cote d'Ivoire" : "Côte d'Ivoire",
 'Democratic Republic of Congo' : "Democratic Republic of the Congo",
 'East Timor': "Timor-Leste",
 'Iran':"Iran (Islamic Republic of)",
 'Laos': "Lao People's Democratic Republic",
 'Moldova' : "Republic of Moldova",
 'Netherlands': "Netherlands (Kingdom of the)",
 'Russia': "Russian Federation",
 'South Korea' : "Republic of Korea",
 'Syria':"Syrian Arab Republic",
 'Tanzania': "United Republic of Tanzania",
 'Turkey': "Türkiye",
 'United Kingdom': "United Kingdom of Great Britain and Northern Ireland",
 'United States': "United States of America",
 'Venezuela':"Venezuela (Bolivarian Republic of)",
 'Vietnam': "Viet Nam"}

precipitation_data.replace(replace_country_name,inplace=True)
precipitation_data.head()


# pesticides_data 
pesticides_data.head()
pesticides_data = pesticides_data[["Area", "Year", "Value"]]
pesticides_data.head()
pesticides_data.rename(columns={"Area": "Country", "Year": "Year", "Value": "pesticides"}, inplace= True)
pesticides_data.head()

# temperature_data
temperature_data.head()
temperature_data= temperature_data[["Area", "Year", "Value"]]
temperature_data.info()
filtered_data = temperature_data[temperature_data["Value"].isna()].count()
filtered_data


mask = temperature_data["Value"].isna()
temperature_data_f = temperature_data[mask]
temperature_data_f

temperature_data["Value"] = temperature_data.groupby("Area")["Value"].transform(lambda x: x.fillna(x.mean()))
temperature_data
temperature_data = temperature_data[(temperature_data["Year"] >= 1990)]
temperature_data
temperature_data.rename(columns={"Area": "Country", "Value": "Temperature"}, inplace= True)
temperature_data


#yield_data
yield_data.head()
yield_data.info()
yield_data = yield_data[["Area","Year","Item","Value"]]
yield_data.dropna()

# We  work with this types of crops 
yield_data["Item"].unique()

yield_data.rename(columns={"Area": "Country", "Item" : "Crop", "Value": "Yield"}, inplace= True)
yield_data=yield_data[(yield_data["Year"] >= 1990)]

#Cleaned datasets
pesticides_data.head(2)
temperature_data.head(2)
precipitation_data.head(2)
yield_data.head(2)


# Merge data 
merged_data = yield_data.merge(pesticides_data, on=['Country', 'Year'], how='left')
merged_data = merged_data.merge(temperature_data, on=['Country', 'Year'], how='left')
Yield = merged_data.merge(precipitation_data, on=['Country', 'Year'], how='left')

Yield.dropna(inplace=True)

#New catrgorical variables
temperature_quantiles = {
    "Low": Yield["Temperature"].quantile(0.25),
    "Moderate": Yield["Temperature"].quantile(0.75),
}

precipitation_quantiles = {
    "Low": Yield["Average precipitation(mm per year)"].quantile(0.25),
    "Moderate": Yield["Average precipitation(mm per year)"].quantile(0.75),
}

Yield["Temperature Category"] = pd.cut(
    Yield["Temperature"],
    bins=[float("-inf"), temperature_quantiles["Low"], temperature_quantiles["Moderate"], float("inf")],
    labels=["Low", "Moderate", "High"]
)

Yield["Precipitation Category"] = pd.cut(
    Yield["Average precipitation(mm per year)"],
    bins=[float("-inf"), precipitation_quantiles["Low"], precipitation_quantiles["Moderate"], float("inf")],
    labels=["Low", "Moderate", "High"]
)

Yield.head()

Yield.to_csv("Yield_output.csv", index=False)

## Check Correlation
numeric_data = Yield.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
correlation_matrix

crop = Yield["Crop"].unique()
for c in crop:
    Yield_for_selective_crops = Yield[Yield["Crop"] == c]
    numeric_data = Yield_for_selective_crops.select_dtypes(include=[np.number])
    correlation_matrix_for_selective_crops = numeric_data.corr()
    print(f"Correlation Matrix for Crop: {c}")
    print(correlation_matrix_for_selective_crops)

Yield_for_selective_crops = Yield[Yield["Crop"].isin(["Maize (corn)","Potatoes","Rice","Sweet potatoes"])]
numeric_data = Yield_for_selective_crops.select_dtypes(include=[np.number])
correlation_matrix_for_selective_crops = numeric_data.corr()
print(correlation_matrix_for_selective_crops)
Yield=Yield_for_selective_crops


# The correlation matrices provided represent the relationships between various factors, 
# including Year, Yield, Pesticides, Temperature, and Average Precipitation for all crops and then specifically 
# for a subset of crops which are Maize (corn), Potatoes, Rice, and Sweet potatoes.

#### For All Crops:
# Year has a strong positive correlation with Temperature at 0.523784, which means as the years progress, the temperature seems to rise.
# The correlations between other factors with Year, Yield, and Pesticides are relatively weak, as all the values are below 0.15.
# Temperature and Average Precipitation have a negative correlation of -0.173553. This indicates that as temperature increases, average precipitation tends to decrease.

#### For Specific Crops (Maize, Potatoes, Rice, Sweet potatoes):
# The correlation between Year and Temperature remains strong at 0.518508, similar to the all-crops scenario.
# Yield now shows a noticeable positive correlation with both Pesticides (0.160674) and Temperature (0.154169). This suggests that for these specific crops, yield might increase with the usage of pesticides and with the rise in temperature.
# Yield and Average Precipitation have a stronger negative correlation of -0.181488 compared to the all-crops data. This indicates that for these specific crops, higher precipitation might lead to lower yields.


## Dominant Crops in Top 25 Countries
#### We'll group by country and crop, sum the yield for each combination, 
#### identify the crop with the highest yield in each country, and then sort countries by 
#### the highest yielding crop to select the top 25.


grouped_data = Yield.groupby(['Country', 'Crop']).agg({'Yield': 'sum'}).reset_index()
idx = grouped_data.groupby(['Country'])['Yield'].transform(max) == grouped_data['Yield']
dominant_crops = grouped_data[idx]
top_25_countries = dominant_crops.sort_values(by='Yield', ascending=False).head(25)
top_25_countries

#### Most countries have potatoes as their dominant crop in terms of yield. 
#### Some exceptions include India (with cassava), Egypt (with sweet potatoes), 
#### and Guatemala and Belize (with plantains and cooking bananas).


# Descriptive Analytics
### Which country has the highest total yield for potatoes?
### For clarity, we'll display only the top 5 countries.

potato_yield = Yield[(Yield["Crop"] == 'Potatoes')]
potatoes_total_yield = potato_yield.groupby('Country')['Yield'].sum().sort_values(ascending=False).head(5)

potatoes_total_yield.plot(kind='bar', color='lightblue', figsize=(10, 6))
plt.title('Top 5 Countries with Highest Total Yield for Potatoes')
plt.ylabel('Total Yield')
plt.xlabel('Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.show()
plt.close()

#### The bar chart displays the top 5 countries with the highest total yield for potatoes. 
#### From the chart, we can see that Japan has the highest total yield, followed by countries like the USA and Jamaica.


# Comparative Analysis
### Let's compare the total yields of potatoes for the USA and Canada.
### For simplicity, we'll visualize the total yield for these two countries over the years in a bar chart


usa_canada_data = potato_yield[potato_yield['Country'].isin(['United States of America', 'Canada'])]
yield_comparison = usa_canada_data.groupby(['Year', 'Country'])['Yield'].sum().unstack()

yield_comparison.plot(kind='bar', stacked=True, figsize=(12, 7), color=['blue', 'green'])
plt.title('Total Potato Yields: USA vs Canada (Over Years)')
plt.ylabel('Total Yield')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.legend(title='Country')
plt.show()
plt.close()
#### The stacked bar chart compares the total potato yields between the USA (in blue) and Canada (in green) over the years. 
#### From the chart, it's evident that the USA consistently has a higher yield compared to Canada.


# Correlation & Causation
### visualize the relationship between average temperature and yield.
### We'll use a simple scatter plot for this.

plt.figure(figsize=(10, 6))
plt.scatter(Yield['Temperature'], Yield['Yield'], color='coral', alpha=0.5)
plt.xlabel('Temperature')
plt.ylabel('Yield')
plt.title('Relationship between Temperature and Yield')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
plt.close()

#### The scatter plot depicts the relationship between temperature and yield. 
#### Points are more densely clustered around certain temperature ranges, indicating areas where yields are more consistent.


# Anomaly Detection
### We'll look at global average yields over the years and identify any significant dips or spikes.
### For this, we'll use a simple line plot.

global_avg_yield = Yield.groupby('Year')['Yield'].mean()

plt.figure(figsize=(10, 6))
global_avg_yield.plot(color='purple')
plt.xlabel('Year')
plt.ylabel('Average Yield')
plt.title('Global Average Crop Yields Over Years')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
plt.close()

#### The line plot displays the global average crop yields over the years. 
#### While the trend remains relatively consistent, there are noticeable fluctuations. 
#### These dips or spikes could be explored further to identify potential anomalies or significant events that might 
#### have caused them.


# Optimization Questions
#### We'll visualize the average yields of potatoes based on temperature.
#### For clarity, we'll use a simple bar chart to display average yields in specific temperature ranges.

bins = [-5, 0, 5, 10, 15, 20, 25, 30]
labels = ['-5 to 0', '0 to 5', '5 to 10', '10 to 15', '15 to 20', '20 to 25', '25 to 30']
Yield['Temperature Range'] = pd.cut(Yield['Temperature'], bins=bins, labels=labels, right=False)

avg_yield_per_temp = Yield.groupby('Temperature Range')['Yield'].mean()

avg_yield_per_temp.plot(kind='bar', color='lightgreen', figsize=(10, 6))
plt.title('Average Potato Yield Based on Temperature Ranges')
plt.ylabel('Average Yield')
plt.xlabel('Temperature Range (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.show()
plt.close()

#### The bar chart showcases the average potato yield based on different temperature ranges. 
#### It's clear that potatoes tend to have higher average yields in the temperature range of -5 to 5 Degrees celsius( Avg Year) 

# Here's the bar chart depicting the average pesticide usage for each crop

avg_pesticide_usage = Yield.groupby('Crop')['pesticides'].mean().sort_values(ascending=False)

plt.figure(figsize=(15, 8))
sns.barplot(x=avg_pesticide_usage.index, y=avg_pesticide_usage.values, palette="viridis")

plt.title('Average Pesticide Usage for Each Crop', fontsize=16)
plt.xlabel('Crop', fontsize=14)
plt.ylabel('Average Pesticides', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
plt.close()

#### "Maize (corn)" and "Potatoes" have the highest average pesticide usage.
#### "Plantains and cooking bananas" and "Soya beans" have comparatively lower average pesticide usages.
#### The chart provides a clear visual representation of how pesticide usage varies among different crops.

# Identify the top 10 countries based on pesticide usage.

total_pesticide_by_country = Yield.groupby('Country')['pesticides'].sum().sort_values(ascending=False)
top_10_countries = total_pesticide_by_country.head(10).index
top_10_countries

# Here's the line plot showing the trend of pesticide usage over time for the top 10 countries

top_countries_data = Yield[Yield['Country'].isin(top_10_countries)]

plt.figure(figsize=(12, 8))

sns.lineplot(data=top_countries_data, x='Year', y='pesticides', hue='Country', ci=None, palette='tab10')

plt.title('Pesticide Usage Over Time for Top 10 Countries', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Pesticides', fontsize=14)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
plt.close()

#### The United States of America consistently has the highest pesticide usage among the top 10 countries throughout the years.
#### Brazil, China, and Argentina also show significant pesticide usage, with Brazil displaying a rising trend over the years.
#### Other countries like Italy and Japan exhibit relatively stable trends in pesticide usage.


# Here's the line plot showing the trend of yield over time for the top 10 countries

plt.figure(figsize=(12, 8))

sns.lineplot(data=top_countries_data, x='Year', y='Yield', hue='Country', ci=None, palette='tab10')

plt.title('Yield Over Time for Top 10 Countries', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Yield', fontsize=14)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
plt.close()

#### The United States of America displays a generally increasing trend in yield over the years.
#### China and India also show increasing yields throughout the timeframe.
#### Brazil, while having significant pesticide usage, exhibits a more fluctuating yield trend.
#### Other countries like Argentina and Italy have relatively stable yield trends, with some fluctuations.

# Here's the pie chart depicting the yield production for the top 10 countries

total_yield_top_countries = top_countries_data.groupby('Country')['Yield'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 7))
plt.pie(total_yield_top_countries, labels=total_yield_top_countries.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", n_colors=10))
plt.title('Yield Production for Top 10 Countries', fontsize=16)

plt.show()
plt.close()

# Create a figure for the visualization
plt.figure(figsize=(16, 8))

# Bar Plot for Average Yield by Crop for each Temperature Category
sns.barplot(data=Yield, x="Crop", y="Yield", hue="Temperature Category", ci=None, palette="coolwarm")
plt.title("Average Yield by Crop for each Temperature Category")
plt.ylabel("Average Yield")
plt.xlabel("Crop")
plt.xticks(rotation=45)
plt.legend(title="Temperature Category")

plt.tight_layout()
plt.show()
plt.close()

#### The bar plot visualizes the average yield for each crop under different temperature categories. Here's a breakdown of the insights:
#### The plot provides a comparative view of yield values across different crops and how they vary with temperature conditions.
#### Some crops might exhibit a more pronounced variation in yield across temperature categories, indicating their sensitivity to temperature changes.

plt.figure(figsize=(16, 8))
sns.barplot(data=Yield, x="Crop", y="Yield", hue="Precipitation Category", ci=None, palette="Blues_d")
plt.title("Average Yield by Crop for each Precipitation Category")
plt.ylabel("Average Yield")
plt.xlabel("Crop")
plt.xticks(rotation=45)
plt.legend(title="Precipitation Category")

plt.tight_layout()
plt.show()
plt.close()
#### The bar plot visualizes the average yield for each crop under different precipitation categories. Here's what you can observe:

####  The plot provides insights into how the yield of different crops varies with precipitation conditions.
####  Some crops might have a clear preference for a specific precipitation category, indicating their water requirements for optimal growth.

plt.figure(figsize=(13, 8))
# Select top 30 countries based on the number of data points
top_countries = Yield['Country'].value_counts().head(30).index.tolist()

# Filter the dataset for these top countries
filtered_data = Yield[Yield['Country'].isin(top_countries)]

# Create a FacetGrid to visualize the interaction
g = sns.FacetGrid(filtered_data, col="Country", col_wrap=5, height=4, sharey=False)
g.map_dataframe(sns.countplot, x="Precipitation Category", hue="Temperature Category", palette="coolwarm")
g.set_axis_labels("Precipitation Category", "Count")
g.set_titles("{col_name}")
g.add_legend(title="Temperature Category")

plt.tight_layout()
plt.show()
plt.close()

#### The visualization displays separate facet grids for each of the top 30 countries. Within each grid, 
#### the x-axis represents the "Precipitation Category" while the different colored bars represent the counts 
#### for each "Temperature Category".

#### This allows you to see, for each country:

#### The distribution of data points across different precipitation categories.
#### Within each precipitation category, the breakdown of temperature categories.
#### For instance, if a country mostly has bars representing the "High" temperature category within 
#### the "Low" precipitation category, it indicates that this country frequently experiences high temperatures 
#### with low precipitation.