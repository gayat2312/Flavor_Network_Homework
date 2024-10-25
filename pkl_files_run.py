import pandas as pd
file_path = 'data/yum_tfidf.pkl'  # Update this path as needed
yum = pd.read_pickle(file_path)
# Set pandas options to display all rows
pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)  # None means display all columns
# Display the DataFrame
print("Displaying all rows of the dataset:")
print(yum)
# Display basic information about the dataset
print("\nDataset Information:")
print(yum.info())
import pandas as pd
file_path = 'data/yum_flavor.pkl'  # Update this path as needed
yum = pd.read_pickle(file_path)
# Set pandas options to display all rows
pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)  # None means display all columns
# Display the DataFrame
print("Displaying all rows of the dataset:")
print(yum)
# Display basic information about the dataset
print("\nDataset Information:")
print(yum.info())

import pandas as pd
# Load the pickle file
file_path = '/Users/gayatrimalladi/Flavor-Network/data/yummly.pkl'  # Update this path as needed
yum = pd.read_pickle(file_path)
# Set pandas options to display all rows
pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)  # None means display all columns
# Display the DataFrame
print("Displaying all rows of the dataset:")
print(yum)
# Display basic information about the dataset
print("\nDataset Information:")
print(yum.info())

file_path = '/Users/gayatrimalladi/Flavor-Network/data/yummly_ingrX.pkl'
yum = pd.read_pickle(file_path)
# Set pandas options to display all rows
pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)  # None means display all columns
# Display the DataFrame
print("Displaying all rows of the dataset:")
print(yum)
# Display basic information about the dataset
print("\nDataset Information:")
print(yum.info())
