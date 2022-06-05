from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import glob
import torch
import os
from PIL import Image

class CustomDataset(Dataset):
	def __init__(self):
		self.dataset_path = os.getcwd() + "/data/"				                   	
		directory_list = glob.glob(self.dataset_path + "*") 	
		#print(directory_list)	
		self.data = []
		self.transform = transforms.Compose([transforms.PILToTensor])
		for each_class in directory_list:
			#print("EACH CLASS", each_class)
			class_name = each_class.split("/")[-1]
			persons_list = glob.glob(each_class + "/*")
			for each_person in persons_list:
				img_x = glob.glob(each_person + "/x.png")
				img_y = glob.glob(each_person + "/y.png")
				img_z = glob.glob(each_person + "/z.png")
				self.data.append([img_x, img_y, img_z, class_name])
		
		self.class_map = {"freeze" : 0, "nofreeze": 1} 

	def __len__(self):
		return len(self.data)    
	
	def __getitem__(self, idx):
	
		img_x_path, img_y_path, img_z_path, class_name = self.data[idx][0][0], self.data[idx][1][0], self.data[idx][2][0], self.data[idx][3]

		# img_x = Image.open(img_x_path)
		# img_y = Image.open(img_y_path)
		# img_z = Image.open(img_z_path)
		img_x = Image.open(img_x_path)
		img_y = Image.open(img_y_path)
		img_z = Image.open(img_z_path)

		tensor_x = transforms.PILToTensor()(img_x)
		tensor_y = transforms.PILToTensor()(img_y)
		tensor_z = transforms.PILToTensor()(img_z)

		tensor_x = tensor_x[:3]
		tensor_y = tensor_y[:3]
		tensor_z = tensor_z[:3]
		
		# tensor_x = read_image(img_x_path)
		# tensor_y = read_image(img_y_path)
		# tensor_z = read_image(img_z_path)

		class_id = self.class_map[class_name]
		class_id = torch.tensor([class_id])
		#print("GET ITEM !!!!")
		return (tensor_x, tensor_y, tensor_z), class_id
