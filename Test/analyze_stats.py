import pandas as pd

x = "sanctuaries"
data = pd.read_excel("../Documentation/faraway_data.xlsx",  engine='openpyxl')


data.value_counts()


def clean_df(df):
    df = df.points.fillna(0)
# Clean data (convert points to numeric where possible)
data['points'] = pd.to_numeric(data['points'], errors='coerce')

# 1. Frequency distribution of resources
data['resources_split'] = data['resources'].str.split(',')
resources_exploded = data.explode('resources_split')
resource_counts = resources_exploded['resources_split'].value_counts()

# 2. Summary statistics of points
points_summary = data['points'].describe()

# 3. Distribution of colors
color_counts = data['color'].value_counts()

# 4. Relationship between color and points
color_points = data.groupby('color')['points'].mean()

# 5. High-value cards
high_value_cards = data[data['points'] > 10]

# Output key insights
print("Resource Counts:\n", resource_counts)
print("Points Summary:\n", points_summary)
print("Color Counts:\n", color_counts)
print("Color vs Points:\n", color_points)
print("High Value Cards:\n", high_value_cards)