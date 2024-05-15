import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

table = pd.read_excel("Ask A Manager Salary Survey 2021 (Responses).xlsx")
df = pd.DataFrame(table)


'''# Delete columns that we don't use
df = df.drop([0, 4, 9, 11, 12], axis = 1)

print(df.columns)
df.loc[df['If "Other," please indicate the currency here: '] == 'Other', 'A'] = df.loc[df['A'] == 'Other', 'B']

names_columns = ['Age', 'Industry', 'Job Title', 'Salary', 'Additional', 'Currency', 'Country']
df_columns = names_columns
#list_columns = df.columns.values.tolist()
#print(list_columns)


'''
# Delete elements where the salary is equal to 0 
df = df[df.iloc[:, 4] != 0]

df['What country do you work in?'] = df['What country do you work in?'].str.strip()

# Filtered by USA
us = ['U.S', 'U.S>', 'United States', 'usa', 'United States of America', 'United States Of America', 'UXZ',
        'U.S.A.', 'US', 'Usa', 'The United State', 'Usa ', 'UsA', 'United  States', 'U.S.A', 'USaa', 'Unted States',
        'United statew', 'United Sttes', 'Unitied States', 'USAB', 'Uniited States', 'Untied States', 'United Statues',
        'United Statesp', 'Uniteed States', 'USS', 'U.s.a.', 'U.SA', 'united stated', 'United Stattes', 'United Statees',
        'UNited States', 'Uniyed states', 'Uniyed states', 'Uniyed States', 'United States of Americas', 'US of A', 'UA',
        'United Statss', 'uS', 'USD', 'United states of america', 'United States is America', 'america', 'United Statws',
        'United Stateds', 'U. S', 'Uniter Statez', 'united states of aamerica', 'uSA', 'USaa', 'United STates', 'Unitef Stated',
        'Usat', 'San Francisco', 'United States of American', 'United Status', 'United Sates of America', 'u.s.', 'United y',
        'Unite States', 'The US', 'USA tomorrow', 'IS', 'is', 'I.S.', 'United Statea', 'ISA','U.s.', 'United Stares',
        'Unites states', 'United State of America', '🇺🇸', 'united States', 'UnitedStates', 'United states of America',
        'United Sates', 'The United States', 'UNITED STATES', 'United States of america', 'Unites States', 'United State',
        'America', 'Us', 'united states', 'United states', 'USA', 'US', 'U.S.', 'California', 'america', 'Hartford', 'U.A.', 
        'USA-- Virgin Islands', 'united states of america', 'Uniyes States', 'U. S.', 'us', 'United Stated', 'Virginia',
        'Worldwide (based in US but short term trips aroudn the world', 'USA (company is based in a US territory, I work remote)',
        'United States (I work from home and my clients are all over the US/Canada/PR', 'For the United States government, but posted overseas',
        'Currently finance', 'bonus based on meeting yearly goals set w/ my supervisor', 'Y', 'US govt employee overseas, country withheld',
        "I earn commission on sales. If I meet quota, I'm guaranteed another 16k min. Last year i earned an additional 27k. It's not uncommon for people in my space to earn 100k+ after commission.",
        'I work for a UAE-based organization, though I am personally in the US.', "USA, but for foreign gov't", 'Remote',
        'Cayman Islands', 'The Bahamas', 'Bermuda']
df['What country do you work in?'] = df['What country do you work in?'].replace(us, 'USA')

# Filtered by Europe
europe = ['europe','FRANCE', 'Jersey, Channel islands', 'the netherlands']

europe = {'the Netherlands': 'The Netherlands', 'NL': 'The Netherlands', 'Danmark': 'Denmark', 'Czechia': 'Czech Republic',
                'Czech republic': 'Czech Republic', 'Company in Germany. I work from Pakistan.': 'Germany', 'croatia': 'Croatia',
                'finland':'Finland', 'france': 'France', 'czech republic': 'Czech Republic', 'denmark': 'Denmark', 'spain': 'Spain',
                'From Romania, but for an US based company': 'Romania', 'Austria, but I work remotely for a Dutch/British company': 'Austria',
                'SWITZERLAND': 'Switzerland', 'switzerland': 'Switzerland', 'Luxemburg': 'Luxembourg', 'The netherlands': 'The Netherlands',
                'ireland': 'Ireland', 'netherlands': 'The Netherlands', 'germany': 'Germany', 'Netherlands': 'The Netherlands',
                'Nederland': 'Netherlands', 'Nederland': 'The Netherlands', 'Italy (South)': 'Italy', 'Catalonia': 'Spain'}
# Filtered by Canada
canada = ['Canada', 'canada', 'CANADA', 'Canda', 'Canadw', 'Can', 'Canad', 'Canadá', 'Csnada', 'Canada, Ottawa, ontario',
          'I am located in Canada but I work for a company in the US', '$2,175.84/year is deducted for benefits', 'Policy']
df['What country do you work in?'] = df['What country do you work in?'].replace(canada, 'Canada')

# Filtered by UK
uk = ['United Kingdom', 'UK', 'England', 'Uk', 'Scotland', 'United kingdom', 'U.K.', 'united kingdom', 'uk',
                'Great Britain', 'England, UK', 'Wales', 'England, United Kingdom', 'Northern Ireland', 'Scotland, UK',
                'england', 'UK (England)', 'United Kingdom (England)', 'Wales (UK)', 'Wales, UK', 'Northern Ireland, United Kingdom',
                'London', 'ENGLAND', 'Unites kingdom', 'England/UK', 'United Kingdom.', 'UK (Northern Ireland)', 'United Kindom',
                'United Kingdomk', 'UK, remote', 'England, UK.', 'Britain', 'Englang', 'Wales (United Kingdom)', 'U.K',
                'England, Gb', 'U.K. (northern England)', 'Isle of Man', 'UK for U.S. company']
df['What country do you work in?'] = df['What country do you work in?'].replace(uk, 'United Kingdom')

errage = {'INDIA': 'India', 'Sri lanka': 'Sri Lanka', 'pakistan':'Pakistan', 'ARGENTINA BUT MY ORG IS IN THAILAND': 'Argentina',
                'United States- Puerto Rico': 'Puerto Rico', 'México':'Mexico', 'Brasil': 'Brazil', 'NZ': 'New Zealand',
                'New zealand': 'New Zealand', 'australia': 'Australia', 'Australian': 'Australia', 'New Zealand Aotearoa': 'New Zealand',
                'Australi': 'Australia', 'Aotearoa New Zealand': 'New zealand', 'new zealand': 'New Zealand',  'japan': 'Japan',
                'From New Zealand but on projects across APAC': 'New Zealand', 'Remote (philippines)': 'Philippines',
                'singapore': 'Singapore', 'UAE': 'United Arab Emirates', 'Japan, US Gov position': 'Japan','Mainland China': 'China',
                'hong konh': 'Hong Kong', 'NIGERIA': 'Nigeria', 'South africa': 'South Africa'}

df['What country do you work in?'] = df['What country do you work in?'].replace(errage)

total_values = df['What country do you work in?'].value_counts()
print(tabulate(total_values.reset_index(), headers=['Country', 'Count'], tablefmt='pretty'))
