import torch

# models
model_names = ['CNN_baseline','SANet','TANet','STANet']
model_name = model_names[3]# # you could change the code to other models by only changing the number
process_data_dir = '../'
dataset_name = 'EAR_4_direction_1D.mat'

device_ids = 0
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 100
batch_size = 128
sample_rate = 128
categorie_num = 4
sbnum = 16
kfold_num = 5

lr=1e-3
weight_decay=0.01

# the length of decision window
decision_window = 128 #1s



