import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open('LaptopPriceModel.pkl', 'rb'))
test_input = {
    'Company': ['HP'],
    'TypeName': ['Notebook'],
    'Inches': [14],
    'Ram': [16],
    'Weight': [1.48],
    'Touchscreen': [0],
    'IPS': [1],
    'PPI': [141.22],
    'HDD': [64],
    'SSD': [512],
    'Hybrid': [0],
    'Flash_Storage': [0],
    'Gpu_Brand': ['AMD'],
    'os': ['Windows']
}
test_df = pd.DataFrame(test_input)
predicted_log_price = pipe.predict(test_df)
predicted_price = np.exp(predicted_log_price)[0]
print(f"Predicted Price for test input: ₹{int(predicted_price)}")
