from torch.utils.data import IterableDataset
import pandas as pd


class VariableMisuseDataset(IterableDataset):
    def __init__(self, dataframe):
        self.df = dataframe
        
    def getRows(self,dataframe):
        # iterate over rows in dataframe
        for _, row in dataframe.iterrows():
            # yield the row as a tuple
            yield (row['function'], row['label'])

    def __iter__(self):
        return self.getRows(self.df)
        

