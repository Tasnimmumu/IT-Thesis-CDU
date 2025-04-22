import pandas as pd
from sqlalchemy import create_engine # type: ignore

# Replace with your MySQL credentials
user = 'root'
password = 'root'
host = 'localhost'
database = 'diabetics prediction'

# File path to the CSV
csv_file = 'diabetes.csv'

# Create engine
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')

# Read CSV
df = pd.read_csv(csv_file)

# Load data into MySQL
df.to_sql('patients_data', con=engine, index=False, if_exists='append')

print("Data successfully loaded into MySQL!")
