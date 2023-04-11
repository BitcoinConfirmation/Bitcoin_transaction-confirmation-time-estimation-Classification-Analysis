import pandas as pd

data=pd.read_csv("txinblock.csv",sep=",",names=['tx_index','vin_sz','vout_sz','ver','size', 'weight', 'time','relayed_by', 'lock_time','fee',
                                                 'block_height', 'block_index','confirmedtime','watingtime','feerate','enterBlock','watiingblock'])
data['feerate']=4*data['fee']/data['weight']
data['watiingblock']+=1
data.to_csv("txinblock2.csv",index=False,header=False)

