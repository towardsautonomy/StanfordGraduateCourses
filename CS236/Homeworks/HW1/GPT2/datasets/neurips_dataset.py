import os
import csv

class NIPS2015Dataset(object):
    def __init__(self, data_folder='datasets/'):
        super().__init__()
        self.papers = []
        if os.path.exists(data_folder):
            with open(os.path.join(data_folder, 'papers.csv'), newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.papers.append({'title': row['Title'], 'abstract': row['Abstract'], 'text': row['PaperText']})
        else:
            raise FileNotFoundError("Please download papers.csv")

        self.p = 0

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        if self.p >= len(self.papers):
            raise StopIteration

        value = self.papers[self.p]
        self.p += 1
        return value

    def _reset(self):
        self.p = 0
