from models.lstmwithtrends import TrendAwareLSTM
from models.fcnn import FCNN


from utils import create_dataset
from utils import EarlyStopping

from utils import train,process_data
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--label",type=str,default='1')
args = parser.parse_args()

label = args.label
label_dict = {"1":[56634,56875],"2":[56875,57117]}


ALL_MODEL = {"TrendAwareLSTM":TrendAwareLSTM,"FCNN":FCNN}  

LOOKBACK = 100
BATCHSIZE = 64


df = process_data()

timeseries = df[["spread"]].values.astype('float32')
# model = TrendAwareLSTM() 56634,56875,57117
for name,model in ALL_MODEL.items():
    first_index,end_index = label_dict[label]
    train(name,ALL_MODEL,timeseries,first_index,end_index,1,LOOKBACK,BATCHSIZE) 














