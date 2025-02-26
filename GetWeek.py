from supabase import create_client, Client
import os
from datetime import datetime, timedelta
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import pickle
import torch
import numpy as np
import pickletools
from models.storage_90days_to_30days_forecast.model import EncoderDecoder

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
class GetWeek:
    def __init__(self):
        self.chem = pd.DataFrame(supabase.table('Chembarambakkam_supply').select('*').order('Date', desc=True).limit(7).execute().data)
        self.chol = pd.DataFrame(supabase.table('Cholavaram_supply').select('*').order('Date', desc=True).limit(7).execute().data)
        self.poondi = pd.DataFrame(supabase.table('Poondi_supply').select('*').order('Date', desc=True).limit(7).execute().data)
        self.red = pd.DataFrame(supabase.table('Redhills_supply').select('*').order('Date', desc=True).limit(7).execute().data)
        self.reservoirs = [ "Redhills",'Chembarambakkam', 'Cholavaram', 'Poondi']
        self.reservoir_data = {
            'Chembarambakkam': self.chem,
            'Cholavaram': self.chol,
            'Poondi': self.poondi,
            'Redhills': self.red
        }
        
    def getReservoirs(self):
        return self.reservoirs
    def getDateList(self):
        start_date = self.chem.loc[:,'Date'].values[-1]
        end_date = self.chem.loc[:,"Date"].values[0]
        date_range = pd.date_range(start=start_date, end=end_date)
        date_list = date_range.to_list()
        return date_list
    def getStorage(self):
        
        df = pd.DataFrame({'Date':self.getDateList()})
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
       
        for reservoir in self.reservoirs:
            df[reservoir] = self.reservoir_data[reservoir]['Storage (mcft)'].values
        return df
    def getInflow(self):
        
        df = pd.DataFrame({'Date':self.getDateList()})
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
       
        for reservoir in self.reservoirs:
            df[reservoir] = self.reservoir_data[reservoir]['Inflow (cusecs)'].values
        return df
    def getOutflow(self):
        
        df = pd.DataFrame({'Date':self.getDateList()})
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
       
        for reservoir in self.reservoirs:
            df[reservoir] = self.reservoir_data[reservoir]['Outflow (cusecs)'].values
        return df


class GetDays:
    def __init__(self,numDays,date = None):
        self.numDays = numDays
        self.time = 30
        if date is None:
            self.fetch_Data(numDays)
        else:
            self.fetch_Data_Date(numDays,date)
    def fetch_Data(self,numDays):
        
        
        self.chem = pd.DataFrame(supabase.table('Chembarambakkam_supply').select('*').order('Date', desc=True).limit(numDays).execute().data)
        self.chol = pd.DataFrame(supabase.table('Cholavaram_supply').select('*').order('Date', desc=True).limit(numDays).execute().data)
        self.poondi = pd.DataFrame(supabase.table('Poondi_supply').select('*').order('Date', desc=True).limit(numDays).execute().data)
        self.red = pd.DataFrame(supabase.table('Redhills_supply').select('*').order('Date', desc=True).limit(numDays).execute().data)
        self.reservoirs = [ "Redhills",'Chembarambakkam', 'Cholavaram', 'Poondi']
        self.reservoir_data = {
            'Chembarambakkam': self.chem,
            'Cholavaram': self.chol,
            'Poondi': self.poondi,
            'Redhills': self.red
        }
    def fetch_Data_Date(self, numDays, date):
        date = datetime.fromisoformat(date)
        end_date = date
        start_date = end_date - timedelta(days=numDays)
        
        # Fetch data for each reservoir within the specified date range
        self.chem = pd.DataFrame(
            supabase.table('Chembarambakkam_supply')
            .select('*')
            .filter('Date', 'gte', start_date.isoformat())  # Greater than or equal to start_date
            .filter('Date', 'lte', end_date.isoformat())    # Less than or equal to end_date
            .order('Date', desc=True)
            .execute().data
        )
        
        self.chol = pd.DataFrame(
            supabase.table('Cholavaram_supply')
            .select('*')
            .filter('Date', 'gte', start_date.isoformat())
            .filter('Date', 'lte', end_date.isoformat())
            .order('Date', desc=True)
            .execute().data
        )
        
        self.poondi = pd.DataFrame(
            supabase.table('Poondi_supply')
            .select('*')
            .filter('Date', 'gte', start_date.isoformat())
            .filter('Date', 'lte', end_date.isoformat())
            .order('Date', desc=True)
            .execute().data
        )
        
        self.red = pd.DataFrame(
            supabase.table('Redhills_supply')
            .select('*')
            .filter('Date', 'gte', start_date.isoformat())
            .filter('Date', 'lte', end_date.isoformat())
            .order('Date', desc=True)
            .execute().data
        )
        
        # List of reservoirs and their corresponding data
        self.reservoirs = ["Redhills", "Chembarambakkam", "Cholavaram", "Poondi"]
        self.reservoir_data = {
            "Chembarambakkam": self.chem,
            "Cholavaram": self.chol,
            "Poondi": self.poondi,
            "Redhills": self.red
        }
    def generateDate(self):
    # Ensure the 'Date' column is converted to datetime
        self.chem['Date'] = pd.to_datetime(self.chem['Date'])

        # Get the last date from the chemset
        last_date = self.chem['Date'].iloc[0]  # Use iloc to get the last element
        # print(last_date)
        
        # Convert last_date to a Python datetime object if it's a NumPy datetime64
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
        
        start_date = last_date + timedelta(days=1)     
        end_date = last_date + timedelta(days=self.time)

        # Generate a range of dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Convert the date range to a list
        date_list = date_range.to_list()

        return date_list
    def getData(self,reservoir):
        # print(type(self.reservoir_data[reservoir.title()].iloc[::-1]))
        return self.reservoir_data[reservoir.title()].iloc[::-1]
    def getPredictions(self,reservoir,target):
        df = self.getData(reservoir)
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        df=df.drop('Date',axis=1)
        # print(df.head())
        scalers_path =os.path.join('models',  'storage_90days_to_30days_forecast',f'{reservoir}' ,'scalers.pkl')
        with open(scalers_path, 'rb') as f:
                scalers_dict = pickle.load(f)
        scaler = scalers_dict['scaler']
        scaled_df = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        X = torch.tensor(scaled_df.values,dtype=torch.float32).unsqueeze(0)
        # print(X.shape)
        try:
            # Define the model architecture
            input_size = X.shape[2]  # Number of features
            hidden_size = 50         # Example hidden size (adjust as needed)
            output_size = 30          # Example output size (adjust as needed)
            model = EncoderDecoder(input_size, hidden_size, output_size)

            # Load the state dictionary
            temp = 'seq2seq_model(ewma)90to30.pth' if  target == 'Storage (mcft)' else 'seq2seq_model_Outflow (cusecs)(ewma)90to30.pth'
            model_path = os.path.join('models',  'storage_90days_to_30days_forecast',f'{reservoir}' ,temp)
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
            model.eval()  # Set the model to evaluation mode
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found. Please check the path.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Generate predictions
        with torch.no_grad():
            predictions = model(X).cpu().numpy()

        # Create dummy data for inverse transformation
        dummy_data = np.zeros((predictions.shape[0] * predictions.shape[1], len(df.columns)))
        target_col_idx = list(df.columns).index(target)
        dummy_data[:, target_col_idx] = predictions.reshape(-1)

        # Inverse transform the predictions
        final = scaler.inverse_transform(dummy_data)[:, target_col_idx]
        final = final.reshape(predictions.shape[0], predictions.shape[1])

        # Generate future dates
        date_range = self.generateDate()

        # Create a DataFrame with predictions
        result_df = pd.DataFrame({
            'Date': date_range[:final.shape[1]],  # Ensure dates match prediction length
            target: final[0]  # Assuming single sequence prediction
        })

        return result_df
    
class GetDemand:
    def __init__(self,numDays):
        self.numDays = numDays
        self.time = 30
        self.data = pd.DataFrame(supabase.table('Chennai_Demand').select('*').order('Date', desc=True).limit(numDays).execute().data)
    def generateDate(self):
    # Ensure the 'Date' column is converted to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'],dayfirst=True)

        # Get the last date from the dataset
        last_date = self.data['Date'].iloc[0]  # Use iloc to get the last element
        # print(last_date)
        
        # Convert last_date to a Python datetime object if it's a NumPy datetime64
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
        
        start_date = last_date + timedelta(days=1)     
        end_date = last_date + timedelta(days=self.time)

        # Generate a range of dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Convert the date range to a list
        date_list = date_range.to_list()

        return date_list
    def getPredictions(self,target='Outflow'):
        df = self.data
        #print('columns',df.columns)
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = self.generateDate()
        df['month'] = df['Date'].dt.month
        df=df.drop('Date',axis=1)
        # print(df.head())
        scalers_path = os.path.join('models', 'Demand_estimation_of_metroplotian','scalers.pkl')
        with open(scalers_path, 'rb') as f:
                scalers_dict = pickle.load(f)
        scaler = scalers_dict['scaler']
        scaled_df = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        X = torch.tensor(scaled_df.values,dtype=torch.float32).unsqueeze(0)
        # print(X.shape)
        try:
            # Define the model architecture
            input_size = X.shape[2]  # Number of features
            hidden_size = 50         # Example hidden size (adjust as needed)
            output_size = 30          # Example output size (adjust as needed)
            model = EncoderDecoder(input_size, hidden_size, output_size)

            # Load the state dictionary
            model_path = os.path.join('models','Demand_estimation_of_metroplotian','Demand_seq2seq_model_Outflow30to30.pth')
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
            model.eval()  # Set the model to evaluation mode
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found. Please check the path.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Generate predictions
        with torch.no_grad():
            predictions = model(X).cpu().numpy()

        # Create dummy data for inverse transformation
        dummy_data = np.zeros((predictions.shape[0] * predictions.shape[1], len(df.columns)))
        target_col_idx = list(df.columns).index(target)
        dummy_data[:, target_col_idx] = predictions.reshape(-1)

        # Inverse transform the predictions
        final = scaler.inverse_transform(dummy_data)[:, target_col_idx]
        final = final.reshape(predictions.shape[0], predictions.shape[1])

        # Generate future dates
        

        # Create a DataFrame with predictions
        result_df = pd.DataFrame({
            'Date': date_range[:final.shape[1]],  # Ensure dates match prediction length
            target: final[0]  # Assuming single sequence prediction
        })

        return result_df

class GetSupply:
    def __init__(self):
        self.gd = GetDays(90,"2024-04-26")
        self.time = 30
    def generateDate(self,data):
   
        data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)


        last_date = data['Date'].iloc[0] 
        
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
        
        start_date = last_date    
        end_date = last_date + timedelta(days=self.time-1)

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        date_list = date_range.to_list()

        return date_list
    def getInput(self,reservoir):
        reservoir = reservoir.title()
        outflow = self.gd.getPredictions(reservoir, 'Outflow (cusecs)')
        storage = self.gd.getPredictions(reservoir, 'Storage (mcft)')
        outflow['Date'] = pd.to_datetime(outflow['Date'])
        storage['Date'] = pd.to_datetime(storage['Date'])
        df = storage.merge(outflow, how = 'left',on = 'Date')
       
        return df

    def getOutput(self,reservoir):
        df = self.getInput(reservoir)
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = self.generateDate(df)
        df['month'] = df['Date'].dt.month
        df=df.drop('Date',axis=1)
        # print(df.head())
        scalar_path =os.path.join('models', 'Supply', f'Supply_estimate_for_{reservoir.lower()}_reservoir', 'scalers.pkl')
        with open(scalar_path, 'rb') as f:
                scalers_dict = pickle.load(f)
        scalerX = scalers_dict['scalerX']
        scalerY = scalers_dict['scalerY']
        scaled_df = pd.DataFrame(
            scalerX.transform(df),
            columns=df.columns,
            index=df.index
        )
        model_path =os.path.join('models', 'Supply', f'Supply_estimate_for_{reservoir.lower()}_reservoir', 'Supply_Outflow_model.pt')
        try:
            model = torch.jit.load(model_path,map_location=torch.device('cpu'))
            model.eval()  # Set the model to evaluation mode
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        X = torch.tensor(scaled_df.values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predictions = model(X).cpu().numpy()
        final = scalerY.inverse_transform(predictions.reshape(-1,1))
        #print(final.reshape(-1).shape,len(date_range))
        result_df = pd.DataFrame({
            'Date': date_range,  
            f'{reservoir}_outflow': final.reshape(-1) 
        })
        
        return result_df
    def getTable(self):
        chem = self.getOutput('Chembarambakkam')
        red = self.getOutput('Redhills')
        chem['Date'] = pd.to_datetime(chem['Date'])
        red['Date'] = pd.to_datetime(red['Date'])
        df = chem.merge(red, how='left', on= 'Date')
        df_ = self.getFinal()
        df_['Date'] = pd.to_datetime(df_['Date'])
        df = df.merge(df_, how = 'left', on='Date')
        df.rename(columns={'Supply_Outflow':'Supply (cusecs)'}, inplace=True)
        
        return df
    def getFinal(self):
        li = ['Chembarambakkam', 'Redhills']
        d = {}
        for i in li:
            d.update({i:self.getOutput(i)})
        date_range = self.generateDate(d[li[0]])
        df = pd.DataFrame({'Date':date_range})

        df['Supply_Outflow'] = np.zeros(len(df), dtype=float)
        for i in range(len(df)):
            row_sum =0
            for j in li:
                row_sum += round(d[j].iloc[i,1],2)
            df.iloc[i,1] = row_sum
        
        return df