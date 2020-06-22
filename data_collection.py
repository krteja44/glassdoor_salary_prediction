import glassdoor_scraper as gs
import pandas as pd

path = "C:/Users/ravit/OneDrive/Desktop/Practice/DS_Project/chromedriver"

df = pd.DataFrame(gs.get_jobs('data scientist', 1000, False, path, 10))
df.to_csv('glassdoor_data.csv')
