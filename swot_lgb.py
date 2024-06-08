#%%
import os
from sklearn.model_selection import train_test_split
from loss_function import *
from spray_lgb import *
import h5py
import scipy.io as sio
import lleaves
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gyh(data_name, index):
    x = data_name[:, index]
    x = x[:, np.newaxis]
    x_norm = x
    return x_norm

data = h5py.File('/home/zhoushuyi/swot/r_all_nonan_dac_cal_25.mat')
dataset = data['data'][:]
dataset = np.swapaxes(dataset, 1, 0)

lon = gyh(dataset, 2)  # longitude
lat = gyh(dataset, 3)  # latitude
u_swot = gyh(dataset, 6)  # latitude
v_swot = gyh(dataset, 7)  # latitude
ds = gyh(dataset, 9)  # latitude
ua_swot = gyh(dataset, 10)  # latitude
va_swot = gyh(dataset, 11)  # latitude
mdt = gyh(dataset, 12)  # latitude
mss = gyh(dataset, 13)  # latitude
tide = gyh(dataset, 14)  # latitude
ssha = gyh(dataset, 15)  # latitude
sshan = gyh(dataset, 16)  # latitude
dac = gyh(dataset, 17)  # latitude

u_drifter = gyh(dataset, 4)  # latitude
v_drifter = gyh(dataset, 5)  # latitude

A = np.concatenate((lat, u_swot, v_swot, ua_swot, va_swot, mdt, mss, tide, ssha, sshan, dac), axis=1)
train_x, test_val_x, train_y, test_val_y = train_test_split(A, v_drifter, test_size=0.2, random_state=10)
test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, test_size=0.5, random_state=10)

model_name = '/home/zhoushuyi/swot/model_lgb_v_nolon_dac_25.txt'
# pred = spray_lgb(train_x, train_y, test_x, test_y, val_x, val_y, model_name)
my_model = lgb.Booster(model_file=model_name)
pred = my_model.predict(val_x, num_iteration=my_model.best_iteration)
pred = pred[:, np.newaxis]
#
loss_functions(pred, val_y)
loss_functions(val_x[:, 2], val_y)
#%%
sio.savemat('/home/zhoushuyi/swot/val_dac_25.mat', {'swot_v': val_x[:, 2], 'drifter_v': val_y, 'lgb_v': pred})

#%%
sio.savemat('/home/zhoushuyi/swot/dataset_lgb_u_nolon_dac_25.mat', {'train_x': train_x, 'train_y': train_y,
            'test_x': test_x, 'test_y': test_y,
            'val_x': val_x, 'val_y': val_y})

