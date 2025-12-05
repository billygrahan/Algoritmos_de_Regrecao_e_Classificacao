#sudo apt install python3-pandas
#sudo apt install python3-sklearn

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)