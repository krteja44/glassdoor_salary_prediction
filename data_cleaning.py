# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:54:02 2020

@author: ravit
"""


import pandas as pd
import os 


df = pd.read_csv("glassdoor_jobs.csv")

# Salary parsing

df["hourly"] = df["Salary Estimate"].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df["employer_provided"]= df["Salary Estimate"].apply(lambda x: 1 if 'employer provided' in x.lower() else 0)

df =  df[df["Salary Estimate"] != '-1']

salary = df["Salary Estimate"].apply(lambda x: x.split('(')[0])
min_hr_eps = salary.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:',''))
min_kd = min_hr_eps.apply(lambda x: x.lower().replace('$','').replace('k',''))

df["min_salary"] = min_kd.apply(lambda x: int(x.split('-')[0]))
df["max_salary"] = min_kd.apply(lambda x: int(x.split('-')[1]))
df["avg_salary"] = (df.min_salary + df.max_salary)/2

# Company name text only

df["company_name_txt"] = df.apply(lambda x: x["Company Name"] if x["Rating"] < 0 else x["Company Name"][:-3], axis=1)

# State field
df["job_state"] = df["Location"].apply(lambda x: x.split(', ')[1])
df.job_state.value_counts()
#Modifying Los Angeles value to CA
df["job_state"] = df["job_state"].apply(lambda x: x if x != 'Los Angeles' else 'CA')

df["hq_same_state"] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)

# age of company
df["age_of_company"] = df["Founded"].apply(lambda x: x if x<0 else 2020-x)

# parsing of job description(python, etc..,)
df["python_yn"] = df["Job Description"].apply(lambda x: 1 if 'python' in x.lower() else 0)
df.python_yn.value_counts()

df["r_yn"] = df["Job Description"].apply(lambda x: 1 if 'r studio' in x.lower()  or 'r-studio' in x.lower() else 0)
df.r_yn.value_counts()

df["spark_yn"] = df["Job Description"].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark_yn.value_counts()

df["aws_yn"] = df["Job Description"].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws_yn.value_counts()

df["excel_yn"] = df["Job Description"].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel_yn.value_counts()

#drop Unnamed(first) column
df.columns
df.drop("Unnamed: 0", axis=1, inplace = True)


#change job title to the higher hierarchy
def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'data analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'machine learning engineer'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
df["job_simp"] = df['Job Title'].apply(title_simplifier)
df.job_simp.value_counts()

#seniority data transformation
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr.' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'junior'
    else:
        return 'na'
    
df["seniority"] = df['Job Title'].apply(seniority)
df.seniority.value_counts()

#length of job description
df["job_desc_len"] = df["Job Description"].apply(lambda x: len(x))
    
#count of competitors
df["comp_count"] = df["Competitors"].apply(lambda x: 0 if x == '-1' else len(x.split(',')))

#converting hourly wage to annual
df["min_salary"]= df.apply(lambda x: (x.min_salary*1816)//1000 if x.hourly==1 else x.min_salary, axis =1) 
df["max_salary"]= df.apply(lambda x: (x.max_salary*1816)//1000 if x.hourly==1 else x.max_salary, axis =1) 
#df[df["min_salary"]>df["max_salary"]][["hourly", "min_salary", "max_salary"]]
#df[df["hourly"]==1][["hourly", "min_salary", "max_salary"]]
#df.drop("max_salary_temp", axis=1, inplace =True)

#removing \n from the company_txt column
#df[df["company_name_txt"].head()
df["company_name_txt"] = df["company_name_txt"].apply(lambda x: x.replace('\n',''))

filename = 'glassdoor_jobs_data_cleaned.csv'

if os.path.exists(filename):
    os.remove(filename)
df.to_csv(filename, index= False)
    