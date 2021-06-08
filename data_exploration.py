import pandas as pd

data_url = "https://raw.githubusercontent.com/casualcomputer/accounting_fraud/master/uscecchini28.csv"
data = pd.read_csv(data_url, error_bad_lines=False)

start_year = data['fyear'].min()
end_year = data['fyear'].max()

print(data[data['gvkey']==1009])