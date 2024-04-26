# %%
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col,first, countDistinct, isnan, when, count, round, substring_index,substring, split, regexp_replace, udf
from tabulate import tabulate

# %% [markdown]
# 
# 
#  ## Since we write local [*] in the master, it will use all cores in our machine. If we said local [4] it will work with 4 cores.
# 
# 
# 
#  ## getOrCreate is used to create a SparkSession if not present.

# %%
print_reports = False

# %%

spark=SparkSession.builder\
    .master("local[*]")\
    .appName("LoanApproval")\
    .getOrCreate()


# %%

sc=spark.sparkContext


# %% [markdown]
#  ## Read Data - SBAnational.csv

# %%

data_path="../../data/SBAnational.csv"


# %%

loan_df =  spark.read.csv(data_path, header=True, inferSchema=True, quote='"', escape='"', multiLine=True)


# %%

loan_df.show(5)
print('=====================')
print("Number of rows in the dataframe:")
print('=====================')
loan_df_count = loan_df.count()
print(loan_df_count)
print('=====================')
print("Number of columns in the dataframe:")
print('=====================')
print(len(loan_df.columns))
print('=====================')
print("Schema of the dataframe:")
print('=====================')
loan_df.printSchema() #prints the dataframe schema
print('=====================')
print("Columns in the dataframe:")
print('=====================')
print(loan_df.columns) 


# %%
loan_df.describe().show()

# %% [markdown]
#  # Preprocessing and cleaning

# %% [markdown]
#  ### Report

# %%
# =========================================================================
# =========================================================================
# ============================= DF REPORT =================================
# =========================================================================
# =========================================================================
def report_df(df, header):
    # Calculate the total number of rows
    rdd_count = df.count()

    # Initialize lists to store column statistics
    col_names = []
    data_types = []
    unique_samples = []
    num_uniques = []
    nan_percentages = []
    report_data = []

    # Iterate over each column
    for col_name in header:
        print(col_name)
        # Append column name
        col_names.append(col_name)
        selected_col = col(col_name)
        selected_col_df = df.select(selected_col)

        # Determine data type
        dtype = selected_col_df.dtypes[0][1]
        data_types.append(dtype)
        distinct_df = selected_col_df.distinct()
        # Collect unique values
        unique_sample = [row[col_name] for row in distinct_df.limit(2).collect()]
        unique_samples.append(unique_sample)

        # Count number of unique values
        n_unique = distinct_df.count()
        num_uniques.append(n_unique)

        # Calculate percentage of NaN values
        none_percentage_val = df.filter(selected_col.isNull()).count() / rdd_count * 100
        nan_percentages.append(none_percentage_val)
        report_data.append([col_name, dtype, unique_sample, n_unique, none_percentage_val])

    return report_data


# %%
if print_reports:
    report_res = report_df(loan_df, loan_df.columns)
    # Display the result
    column_names = ['Column', 'Type', 'Unique Sample', 'N Unique', '%None']
    print(tabulate(report_res, headers=column_names, tablefmt='grid'))




# %%


# %%

def show_percentage_of_each_value_in_column(df, df_count,col_name,show_num=10):
    # Calculate percentage of 0s and 1s
    percentage_df = df.groupBy(col_name).agg((count("*") / df_count).alias("Percentage"))

    # Round percentage values to two decimal places
    percentage_df = percentage_df.withColumn("Percentage", round(col("Percentage") * 100, 2))

    # sort the dataframe by percentage descending
    percentage_df = percentage_df.sort(col("Percentage").desc())

    # Show result
    percentage_df.show(show_num)

def show_df_where_col_isnull(df, col_name, show_num):
    # Filter rows where 'Name' column is null
    filtered_df = df.filter(col(col_name).isNull())

    # Show the resulting DataFrame
    filtered_df.show(show_num)
    null_count = filtered_df.count()
    print(f"Null Count: {null_count}")

def print_unique_val_num_in_col(df, col_name):
    # percentage of unique values in the city
    unique_count = loan_df.select(col_name).distinct().count()
    print(f"Number of unique values in {col_name}: {unique_count}")
    percentage = unique_count / loan_df_count * 100
    print(f"Percentage of unique values in {col_name}: {percentage:.2f}%")

# %% [markdown]
#  ### 1. LoanNr_ChkDgt - ID
# 
#  Drop the column as it is an ID column and does not provide any information for the analysis.

# %%

loan_df = loan_df.drop('LoanNr_ChkDgt')


# %% [markdown]
#  ### 2. Name - Name of Borrower
# 
#  Drop the column as it is a name column and does not provide any information for the analysis.

# %%

col_name = 'Name'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name, show_num=5)

# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
# Fill null values in the 'Name' column with 'Unknown Company'
loan_df = loan_df.fillna({col_name: 'Unknown Name'})
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%

print_unique_val_num_in_col(loan_df, col_name)


# %% [markdown]
#  Drop as most of the names are unique

# %%

# loan_df = loan_df.drop('Name')


# %% [markdown]
#  ### 3. City - City of Borrower
# 
# 

# %%

# Count the occurrences of each value in city column
col_name = 'City'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name, show_num=5)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
# Fill null values in the 'Name' column with 'Unknown Company'
loan_df = loan_df.fillna({col_name: 'Unknown City'})
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %%

unique_city_df = loan_df.select(col_name).groupBy(col_name).agg((count("*")).alias("Count")).sort(col("Count").desc())
unique_city_df.show()


# %% [markdown]
#  ### 4. State - State of Borrower

# %%

col_name = 'State'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
print_unique_val_num_in_col(loan_df, col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
# Filter rows where 'State' column is null
filtered_df = loan_df.filter(col(col_name).isNull())

# Extract unique values from 'Zip' column
unique_zips = filtered_df.select('Zip').distinct().collect()

# Extract unique zip codes as a list
unique_zip_list = [row['Zip'] for row in unique_zips]

# Print unique zip codes
print(unique_zip_list)

# %%
# Sort the DataFrame by 'Zip Code' in ascending order
df_sorted = loan_df.orderBy('Zip')

# Define a window specification for the group
window_spec = Window.partitionBy('Zip').orderBy('Zip')

# Fill the null 'State' values with the corresponding non-null 'State' value within each group
loan_df = df_sorted.withColumn(col_name, first(col_name, ignorenulls=True).over(window_spec))

# Show the resulting DataFrame
loan_df.show()

# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %% [markdown]
# Only 1 state with null values exist, so it is easy to look it up using its zip code.
# Zip code 96205 exists in AP.

# %%
loan_df = loan_df.fillna({col_name: 'AP'})
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %% [markdown]
#  ### 5. Zip - Zip code of Borrower

# %%

col_name = 'Zip'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
print_unique_val_num_in_col(loan_df, col_name)

# %% [markdown]
# Least zip code in the US starts with 502, so any value less than this is unvalid.

# %%
# Filter rows where 'Zip' column is less than 501
filtered_df = loan_df.filter(col('Zip') < 501)

# Extract unique values from the filtered 'Zip' column
unique_zips = filtered_df.select('Zip').distinct().collect()
all_zips = filtered_df.select('Zip').collect()

# Extract unique zip codes as a list
unique_zip_list = [row['Zip'] for row in unique_zips]
all_zip_list = [row['Zip'] for row in all_zips]

# Print unique zip codes
print("Unique Zip codes less than 501:")
print(unique_zip_list)
print(f"Count: {len(all_zip_list)}")

# %%
# Filter rows where 'Zip' column is less than 501
loan_df = loan_df.filter(loan_df[col_name] >= 501)

# %%

unique_zip_df = loan_df.select(col_name).groupBy(col_name).agg((count("*")).alias("Count")).sort(col("Count").desc())
unique_zip_df.show()


# %% [markdown]
# Cast Zip to string to treat is as a categorical feature.

# %%
# Cast the 'Zip' column to string
loan_df = loan_df.withColumn('Zip', col('Zip').cast('string'))

# %%

# loan_df = loan_df.drop(col_name)


# %% [markdown]
#  ### 6. Bank - Name of the bank that gave the loan

# %%

col_name = 'Bank'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %%
# Fill null values in the 'Bank' column with 'Unknown Bank'
loan_df = loan_df.fillna({col_name: 'Unknown Bank'})
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %% [markdown]
#  ### 7. BankState - State of Bank

# %%

col_name = 'BankState'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %% [markdown]
# Drop nulls as we cant populate them.

# %%
loan_df = loan_df.dropna(subset=[col_name])

# %%
show_df_where_col_isnull(loan_df, "BankState", show_num=5)

# %% [markdown]
#  ### 8. NAICS - North American Industry Classification System code for the industry where the business is located

# %%

col_name='NAICS'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %%

print_unique_val_num_in_col(loan_df, col_name)

# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%

# Convert NAICS code into related sector

# Extract first two characters of NAICS code
first_two_chars = substring(loan_df["NAICS"], 1, 2)
# print(first_two_chars)[0]

# Apply mapping using when and otherwise
loan_df = loan_df.withColumn("Sector",
    first_two_chars
)
loan_df = loan_df.drop("NAICS")
col_name='Sector'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%

naics_to_sector = {
    11: 'Agriculture, Forestry, Fishing and Hunting',
    21: 'Mining, Quarrying, and Oil and Gas Extraction',
    22: 'Utilities',
    23: 'Construction',
    31: 'Manufacturing',
    32: 'Manufacturing',
    33: 'Manufacturing',
    42: 'Wholesale Trade',
    44: 'Retail Trade',
    45: 'Retail Trade',
    48: 'Transportation and Warehousing',
    49: 'Transportation and Warehousing',
    51: 'Information',
    52: 'Finance and Insurance',
    53: 'Real Estate and Rental and Leasing',
    54: 'Professional, Scientific, and Technical Services',
    55: 'Management of Companies and Enterprises',
    56: 'Administrative and Support and Waste Management and Remediation Services',
    61: 'Educational Services',
    62: 'Health Care and Social Assistance',
    71: 'Arts, Entertainment, and Recreation',
    72: 'Accommodation and Food Services',
    81: 'Other Services (except Public Administration)',
    92: 'Public Administration'
}

loan_df = loan_df.withColumn(col_name, 
                             when(col(col_name) == 32, 31)
                             .when(col(col_name) == 33, 31)
                             .when(col(col_name) == 45, 44)
                             .when(col(col_name) == 49, 48)
                             .otherwise(col(col_name)))
# Cast the 'Zip' column to string
# loan_df = loan_df.withColumn('Sector', col('Sector').cast('string'))
# # Convert NAICS codes to their corresponding sectors
# loan_df_temp = loan_df.withColumn(col_name, 
#                    when(col(col_name).isin(list(naics_to_sector.keys())), 
#                         naics_to_sector[col(col_name).cast("int")])  # Cast to int before accessing dictionary
#                    .otherwise("Unknown"))

# loan_df_temp.show(5)


# %%
loan_df =loan_df.dropna(subset=["Sector"]) 

# %%
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %% [markdown]
#  ### 9. ApprovalDate - Date SBA commitment issued

# %%

col_name = 'ApprovalDate'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %%

# the full date has too much detail, so we will extract the month only

loan_df = loan_df.withColumn("ApprovalMonth", split(col(col_name), "-")[1])
loan_df = loan_df.drop(col_name)
col_name = 'ApprovalMonth'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,"ApprovalMonth")


# %% [markdown]
#  ### 10. ApprovalFY - Fiscal Year of commitment
# 
#  Drop the column as it is a date column and does not provide any information for the analysis.

# %%

loan_df = loan_df.drop('ApprovalFY')


# %% [markdown]
#  ### 11. Term - Loan term in months

# %%

col_name = 'Term'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %%

# loan_df = loan_df.withColumn("Term_category", 
#                              when((col(col_name) <=90),'Below 3 months')
#                              .when(((col(col_name)>90) & (col(col_name)<=180)), '3-6 months')
#                              .when(((col(col_name)>180) & (col(col_name)<=365)),  '6-12 months')
#                              .otherwise('More Than a Year'))
# loan_df = loan_df.drop(col_name)
# col_name = "Term_category"
# show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %% [markdown]
#  ### 12. NoEmp - Number of Business Employees

# %%

col_name = 'NoEmp'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %% [markdown]
#  ### 13. NewExist - 1 = Existing business, 2 = New business

# %%

col_name = 'NewExist'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %% [markdown]
#  Drop rows with 0 or Null

# %%

col_name = 'NewExist'
loan_df = loan_df.filter(loan_df[col_name] != 0)
loan_df = loan_df.filter(loan_df[col_name].isNotNull())
loan_df_count = loan_df.count()

# %%

show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %% [markdown]
# Convert it to boolean, '2' is true, '1' is false.

# %%

loan_df = loan_df.withColumn(col_name, 
                   when(col(col_name) == "2", 1)
                   .otherwise(0)
                   .cast("int"))


# %%

show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %% [markdown]
#  ### 14. CreateJob - Number of jobs created

# %%

col_name='CreateJob'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %%

# loan_df = loan_df.drop(col_name)


# %% [markdown]
#  ### 15. RetainedJob - Number of jobs retained

# %%

col_name='RetainedJob'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %%

# loan_df = loan_df.drop(col_name)


# %% [markdown]
#  ### 16. FranchiseCode - Franchise code, (00000 or 00001) = No franchise

# %%

col_name='FranchiseCode'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %% [markdown]
#  We don't care about the franchise code, we only care if there is a franchise or not

# %%

# make 0 or 1 = 0, anything else = 1
loan_df = loan_df.withColumn("IsFranchise", when((col(col_name) == 0) | (col(col_name) == 1), 0).otherwise(1))


# %%

col_name = 'IsFranchise'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%

loan_df = loan_df.drop('FranchiseCode')


# %% [markdown]
#  ### 17. UrbanRural - 1 = Urban, 2 = rural, 0 = undefined

# %%

col_name = 'UrbanRural'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %% [markdown]
#  ### 18. RevLineCr - Revolving line of credit: Y = Yes, N = No

# %%

col_name = 'RevLineCr'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %% [markdown]
#  Filter only N and Y

# %%

col_name = 'RevLineCr'
print(f"Number of rows before filtering: {loan_df_count}")
loan_df = loan_df.filter(loan_df[col_name].isin('N', 'Y'))
loan_df_count = loan_df.count()
print(f"Number of rows after filtering: {loan_df_count}")
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %% [markdown]
#  Transform N and Y to 0 and 1

# %%

loan_df = loan_df.withColumn(col_name, 
                   when(col(col_name) == "Y", 1)
                   .otherwise(0)
                   .cast("int"))


# %% [markdown]
#  ### 19. LowDoc - LowDoc Loan Program: Y = Yes, N = No

# %%

col_name = "LowDoc"
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
print_unique_val_num_in_col(loan_df, col_name)

# %% [markdown]
#  Filter only N and Y

# %%

col_name = 'LowDoc'
print(f"Number of rows before filtering: {loan_df_count}")
loan_df = loan_df.filter(loan_df[col_name].isin('N', 'Y'))
loan_df_count = loan_df.count()
print(f"Number of rows after filtering: {loan_df_count}")
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %% [markdown]
#  Transform N and Y to 0 and 1

# %%

loan_df = loan_df.withColumn(col_name, 
                   when(col(col_name) == "Y", 1)
                   .otherwise(0)
                   .cast("int"))


# %% [markdown]
#  ### 20. ChgOffDate - The date when a loan is declared to be in default
# 
#  Drop the column due to the high number of missing values.

# %%

loan_df = loan_df.drop('ChgOffDate')


# %% [markdown]
#  ### 21. DisbursementDate - Date when loan was disbursed

# %%

loan_df = loan_df.drop('DisbursementDate')


# %% [markdown]
#  ### 22. DisbursementGross - Amount disbursed

# %%
col_name = "DisbursementGross"
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
loan_df = loan_df.withColumn("clean_DisbursementGross", regexp_replace("DisbursementGross", "\$", ""))  # Remove $
loan_df = loan_df.withColumn("clean_DisbursementGross", regexp_replace("clean_DisbursementGross", ",", ""))  # Remove comma
loan_df = loan_df.withColumn("clean_DisbursementGross", col("clean_DisbursementGross").cast("float"))
col_name = "clean_DisbursementGross"
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%

loan_df = loan_df.drop('DisbursementGross')


# %% [markdown]
#  ### 23. BalanceGross - Gross amount outstanding

# %%

col_name = 'BalanceGross'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %% [markdown]
#  Drop as most of the values are 0

# %%

loan_df = loan_df.drop('BalanceGross')


# %% [markdown]
#  ### 24. MIS_Status - Target variable

# %% [markdown]
#  Delete rows that have null target value (MIS_Status)

# %%

col_name ="MIS_Status"
show_percentage_of_each_value_in_column(loan_df, loan_df_count, col_name)


# %%

# drop rows with null values in MIS_Status column
loan_df = loan_df.dropna(subset=[col_name])
show_percentage_of_each_value_in_column(loan_df, loan_df_count, col_name)



# %% [markdown]
#  ### Replace target values with 0 and 1
# 
#  Target value column is: MIS_Status
# 
#  "P I F" = 1
# 
#  "CHGOFF" = 0

# %%

loan_df = loan_df.withColumn(col_name, 
                   when(col(col_name) == "P I F", 1)
                   .otherwise(0)
                   .cast("int"))


# %% [markdown]
#  Show the percentage of:
# 
#  - Paid in full loans (approved loans), MIS_Status = 1
# 
#  - Charged off loans (rejected loans), MIS_Status = 0

# %%

show_percentage_of_each_value_in_column(loan_df, loan_df_count, col_name)


# %% [markdown]
# Place target column at the end

# %%
# Assuming df is your DataFrame and column_name is the name of the column you want to move to the end
column_name = "MIS_Status"

# Get the current column names
current_columns = loan_df.columns

# Select columns excluding the column to be moved to the end
new_columns = [col for col in current_columns if col != column_name]

# Add the column to be moved to the end
new_columns.append(column_name)

# Reorder the DataFrame with the new column order
loan_df = loan_df.select(*new_columns)


# %% [markdown]
#  ### 25. ChgOffPrinGr - Charged-off amount

# %%

col_name = 'ChgOffPrinGr'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
loan_df = loan_df.withColumn("clean_ChgOffPrinGr", regexp_replace("ChgOffPrinGr", "\$", ""))  # Remove $
loan_df = loan_df.withColumn("clean_ChgOffPrinGr", regexp_replace("clean_ChgOffPrinGr", ",", ""))  # Remove comma
loan_df = loan_df.withColumn("clean_ChgOffPrinGr", col("clean_ChgOffPrinGr").cast("float"))
col_name = "clean_ChgOffPrinGr"
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %% [markdown]
#  Drop this column as it will leak info to the column, because if the value is 0, this means that the loan is charged off

# %%

loan_df = loan_df.drop('ChgOffPrinGr')


# %% [markdown]
#  ### 26. GrAppv - Gross amount of loan approved by bank

# %%

col_name = "GrAppv"
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %% [markdown]
#  #### Clean this column
# 
#  - Remove $
# 
#  - Remove ,
# 
#  - Convert to float

# %%

loan_df = loan_df.withColumn("clean_GrAppv", regexp_replace("GrAppv", "\$", ""))  # Remove $
loan_df = loan_df.withColumn("clean_GrAppv", regexp_replace("clean_GrAppv", ",", ""))  # Remove comma
loan_df = loan_df.withColumn("clean_GrAppv", col("clean_GrAppv").cast("float"))
col_name = "clean_GrAppv"
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)


# %%
show_df_where_col_isnull(loan_df, "clean_GrAppv", show_num=5)

# %%

loan_df = loan_df.drop('GrAppv')


# %% [markdown]
#  ### 27. SBA_Appv - SBA's guaranteed amount of approved loan
# 
#  Drop as we don't know this amount in the future

# %%
col_name = 'SBA_Appv'
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %%
loan_df = loan_df.withColumn("clean_SBA_Appv", regexp_replace("SBA_Appv", "\$", ""))  # Remove $
loan_df = loan_df.withColumn("clean_SBA_Appv", regexp_replace("clean_SBA_Appv", ",", ""))  # Remove comma
loan_df = loan_df.withColumn("clean_SBA_Appv", col("clean_SBA_Appv").cast("float"))
col_name = "clean_SBA_Appv"
show_percentage_of_each_value_in_column(loan_df, loan_df_count,col_name)

# %%
show_df_where_col_isnull(loan_df, col_name, show_num=5)

# %%
loan_df = loan_df.drop('SBA_Appv')


# %% [markdown]
#  ### Final schema

# %%

loan_df.printSchema()


# %% [markdown]
#  ### Check duplicated rows based on all columns
# 
# 

# %%

print("Number of duplicate rows in the dataframe:")
loan_df_duplicates = loan_df_count - loan_df.dropDuplicates().count()
print(loan_df_duplicates)
loan_df = loan_df.dropDuplicates()

# %% [markdown]
#  ### Final DF Count

# %%

loan_df_count = loan_df.count()
print(f"Final DF count: {loan_df_count}")


# %%
if print_reports:
    report_res = report_df(loan_df, loan_df.columns)
    # Display the result
    column_names = ['Column', 'Type', 'Unique Sample', 'N Unique', '%None']
    print(tabulate(report_res, headers=column_names, tablefmt='grid'))

# %%
# output_path = "../data/preprocessed.csv"

# # Save the DataFrame to a CSV file
# loan_df.write.csv(output_path, header=True, mode="overwrite")

# %%
# Convert PySpark DataFrame to Pandas DataFrame
pandas_df = loan_df.toPandas()

# Specify the path where you want to save the CSV file
output_path = "../data/preprocessed.csv"

# Save the Pandas DataFrame to a CSV file
pandas_df.to_csv(output_path, index=False)

# %%
sample_size = 50000
# Save a sample
output_path = f"../sample_data/{sample_size}.csv"

# Save the first 50000 rows of the Pandas DataFrame to a CSV file
pandas_df.head(sample_size).to_csv(output_path, index=False)

# %%
sample_size = 1000
# Save a sample
output_path = f"../sample_data/{sample_size}.csv"

# Save the first 50000 rows of the Pandas DataFrame to a CSV file
pandas_df.head(sample_size).to_csv(output_path, index=False)


