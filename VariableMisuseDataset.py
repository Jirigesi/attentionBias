from torch.utils.data import IterableDataset
import pandas as pd


class VariableMisuseDataset(IterableDataset):
    def __init__(self, dataframe):
        self.df = dataframe
        
    def getRows(self,dataframe):
        # build a n_label column that label "Correct" as float 0 and others as 1
        self.df["n_label"] = self.df["label"].apply(lambda x: 0.0 if x == "Correct" else 1.0)
        # iterate over rows in dataframe
        for _, row in dataframe.iterrows():
            # yield the row as a tuple
            yield (row['function'], row['n_label'])

    def __iter__(self):
        return self.getRows(self.df)
    