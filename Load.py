import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
from supabase import create_client, Client
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import csv
import codecs
import urllib.request
import urllib.error
import sys
import os


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
ApiKey=os.environ.get("WEATHER_API1")

class Load():
    def __init__(self):
       pass
    def getLastDate(self):
        try:
            last_record = supabase.table('Chembarambakkam_supply').select('*').order('Date', desc=True).limit(1).execute()
            return datetime.strptime(last_record.data[0]['Date'], "%Y-%m-%d")
        except Exception as err:
            # Print the error if something goes wrong
            print(err)
    def generate_DateList(self, data, date_col):
        start_date = data.loc[0,date_col]
        end_date = data.loc[data.index[-1],date_col]
        date_range = pd.date_range(start=start_date, end=end_date)
        date_list = date_range.to_list()
        return date_list
    def  obtain_date_col_name(self,data):
        date_columns = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        return date_columns[0]
    def mergeData(self, data, date_col):
        date_list = self.generate_DateList(data,date_col)
        merged=pd.DataFrame({date_col:date_list})
        merged_data=pd.merge(merged, data, on=date_col, how='left')
        
        return merged_data
    def imputation(self,df,cols):
        for col in cols:
            df[col]=df[col].interpolate(method='time')
            
    def fill_missing_Data(self,data):
        date_col = self.obtain_date_col_name(data)
        data = self.mergeData(data, date_col)
        data.set_index(date_col, inplace=True)
        self.imputation(data, data.columns)
        return data
    def fetch_data(self,date_str):
        url = "https://cmwssb.tn.gov.in/lake-level"
        params = {'date': date_str}
        response = requests.get(url, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    def addReservoirData(self,start_date,end_date):
        delta = timedelta(days=1)
        reservoirs = ['POONDI', 'CHOLAVARAM', 'PUZHAL',  'CHEMBARAMBAKKAM', ]
        df_dict = {reservoir: [] for reservoir in reservoirs}
        while start_date <= end_date:
            date_str = start_date.strftime('%d-%m-%Y')
            print(f"Fetching data for {date_str}")

            soup = self.fetch_data(date_str)

            table = soup.find('table', class_='lack-view')
            if table:
                rows = table.find_all('tr')[1:-1]  # Exclude header and total rows

                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 9:  # Ensure we have enough cells
                        reservoir = cells[0].text.strip()
                        if reservoir in reservoirs:
                            data = {
                                'Date': date_str,
                                'Storage (mcft)':  float(cells[4].text.strip()) if cells[4].text.strip()!='-' else 0.0,
                                'Inflow (cusecs)':  float(cells[6].text.strip()) if cells[6].text.strip()!='-' else 0.0,
                                'Outflow (cusecs)':  float(cells[7].text.strip()) if cells[7].text.strip()!='-' else 0.0,
                                'Rainfall (mm)':  float(cells[8].text.strip()) if cells[8].text.strip()!='-' else 0.0
                            }
                            df_dict[reservoir].append(data)
                        
            else:
                for reservoir in reservoirs:
                    data = {
                        'Date': date_str,
                        'Storage (mcft)': 0.0,
                        'Inflow (cusecs)': 0.0,
                        'Outflow (cusecs)':  0.0,
                        'Rainfall (mm)':  0.0
                    }
                    df_dict[reservoir].append(data)
                print(f"No data found for {date_str}")

            start_date += delta
            time.sleep(1)
        for reservoir in reservoirs:
            df_dict[reservoir] = pd.DataFrame(df_dict[reservoir])
        return df_dict 
    def getWeatherData(self,StartDate, EndDate ):
        BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
        if isinstance(StartDate, datetime):
            StartDate = StartDate.strftime('%Y-%m-%d')
        if isinstance(EndDate, datetime):
            EndDate = EndDate.strftime('%Y-%m-%d')
        UnitGroup='us'
        Location='chennai'
        ContentType="csv"
        Include="days"
        ApiQuery=BaseURL + Location
        if (len(StartDate)):
            ApiQuery+="/"+StartDate
            if (len(EndDate)):
                ApiQuery+="/"+EndDate
        ApiQuery+="?"
        if (len(UnitGroup)):
            ApiQuery+="&unitGroup="+UnitGroup

        if (len(ContentType)):
            ApiQuery+="&contentType="+ContentType

        if (len(Include)):
            ApiQuery+="&include="+Include

        ApiQuery+="&key="+ApiKey
        try: 
            CSVBytes = urllib.request.urlopen(ApiQuery)
        except urllib.error.HTTPError  as e:
            ErrorInfo= e.read().decode() 
            print('Error code: ', e.code, ErrorInfo)
            sys.exit()
        except  urllib.error.URLError as e:
            ErrorInfo= e.read().decode() 
            print('Error code: ', e.code,ErrorInfo)
            sys.exit()
        CSVText = csv.reader(codecs.iterdecode(CSVBytes, 'utf-8'))
        all_rows = []
        headers = None
        for row_index, row in enumerate(CSVText):
            if row_index == 0:
                headers = row
            else:
                # Create a dictionary for each row with proper column names
                row_dict = {}
                for col_index, value in enumerate(row):
                    row_dict[headers[col_index]] = value
                all_rows.append(row_dict)

        # Create pandas DataFrame
        if all_rows:
            df = pd.DataFrame(all_rows)
            print('Weather data extracted successfully')
        
        else:
            df = pd.DataFrame({})
            print('No data was retrieved from the weather service.')
            return df
        return df[['datetime','tempmax','cloudcover','windspeed','humidity']]
        
    def combineData(self, data1, data2):
        # data1['Date'] =data1.index
        # data2['Date'] = data2.index
        data1 = self.fill_missing_Data(data1)
        data2 = self.fill_missing_Data(data2)
        merged = pd.merge(data1, data2, left_index=True, right_index=True, how = 'left')
        return merged
    def getData(self):
        delta = timedelta(days=1)
        start_date = self.getLastDate()+delta
        end_date = datetime.now()
        if start_date.date() >= end_date.date():
            print("Data is up to date")
            return {}
        data2 = self.getWeatherData(start_date, end_date)
        data2['datetime']=pd.to_datetime(data2['datetime'], dayfirst = True)
        data1 = self.addReservoirData(start_date,end_date)
        for reservoir, df in data1.items():
            df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)
            data1[reservoir] = self.combineData(df,data2)
        return data1
    def updateData(self):
        data1 = self.getData()
        if(len(data1) == 0):
            return 
        for reservoir, df in data1.items():
            name = 'Redhills' if reservoir == 'PUZHAL' else reservoir.title()
            
            try:
                # Convert DataFrame to list of dictionaries
                df['Date'] = df.index.strftime('%Y-%m-%d')
                df = df.drop_duplicates(subset=['Date'], keep='first')
                records = df.to_dict('records')
                
                # Insert all records into the table
                supabase.table(f'{name}_supply').insert(records).execute()
                
            except Exception as e:
                print(f"Error inserting data for {name}: {e}")
       
  # Returns the count of NaN values


        

