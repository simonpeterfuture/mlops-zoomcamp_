import pickle
import pandas as pd
import sys 
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline




with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')



dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)



year=int(sys.argv[1])
month=int(sys.argv[2])
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


df_resuts=pd.DataFrame()
df_resuts['ride_id']=df['ride_id']
df_resuts['predicted_duration']=y_pred
df_resuts.head()
print(df_resuts['predicted_duration'].mean())



output_file=f'{year}_{month}results.parquet'
df_resuts.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

