from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import torch

class CustomDataset(Dataset):
	def __init__(self):
		self.dataset_path = "" 						# dataset path
		directory_list = glob.glob(self.dataset_path + "*") 		# This list should contain paths for both freeze and nofreeze folders(containing images)
		self.data = []						        # List containing list of tensor_img and class_name [[img1, class_name1],[img2, class_name2]....]
		for single_class in directory_list:	
			class_name = single_class.split("/")[-1]
			for tensor_img in glob.glob(single_class + "/*.png"):	# Change the format of .png based on the image extensions 
				self.data.append([tensor_img, class_name])
		
		self.class_map = {"freeze" : 0, "nofreeze": 1} 

	def __len__(self):
		return len(self.data)    
	
	def __getitem__(self, idx):
		tensor_image, class_name = self.data[idx]
		class_id = self.class_map[class_name]
		#img_tensor = torch.from_numpy(img)		### Exclude this if they are already a torch tensor
		class_id = torch.tensor([class_id])
		return tensor_image, class_id


def getDataLoader(mode, batch_size, dataset, num_workers=4):
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=(mode=='train'),
                            num_workers=num_workers)
    return dataloader
