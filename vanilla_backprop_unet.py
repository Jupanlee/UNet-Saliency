"""
Created on Mon Oct 08 10:53 2018
@author: Jupanlee
"""

import torch
import argparse
from misc_functions import get_params, convert_to_grayscale, save_gradient_images
from pathlib import Path

class VanillaBackprop():
	"""
	Produces gradients generated with vanilla back propagation from the image
	"""
	def __init__(self, model):
		self.model = model
		
		self.gradients = None
		
		# Put model in evaluation mode
		self.model.eval()
		
		# Hook the first layer to get the gradient
		self.hook_layers()


	def hook_layers(self):
		def hook_function(module, grad_in, grad_out):
			self.gradients = grad_in[0]
		
		# Register hook to the first layer
		# first_layer = list(self.model.features._modules.items())[0][1]
		# first_layer = self.model.conv1
		first_layer = self.model.dec5
		first_layer.register_backward_hook(hook_function)

	def generate_gradients(self, input_image, input_mask):
		# Forward
		model_output = self.model(input_image)
		# Zero grads
		self.model.zero_grad()
		# Target for backprop
		one_hot_output = input_mask
		# Backward pass
		model_output.backward(gradient=one_hot_output)
		# Convert Pytorch variable to numpy array
		gradients_as_arr = self.gradients.data.numpy()
		return gradients_as_arr


		
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Vanilla Backprop Visualization')
	parser.add_argument('--image_path', type = str, default = "../dataset/msra/valid/image/im100.jpg")
	parser.add_argument('--mask_path', type = str, default = "../dataset/msra/valid/label/im100.png")
	parser.add_argument('--model_path', type = str, default = "./runs/debug/")
	parser.add_argument('--fold', type = str, default = "unet16")
	parser.add_argument('--model_type', type = str, default = "UNet16")

	args = parser.parse_args()
	image_path = args.image_path	
	mask_path = args.mask_path
	model_path = str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=args.fold)))
	model_type = args.model_type

	(original_image, prep_img, prep_mask, file_name_to_export, pretrained_model, img_width, img_height) = get_params(image_path, mask_path, model_path, model_type)
	
	# Vanilla backprop
	VBP = VanillaBackprop(pretrained_model)
	
	# Generate gradients
	vanilla_grads = VBP.generate_gradients(prep_img, prep_mask)
	print(vanilla_grads.shape)
	
	# Save colored gradients
	# save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color', img_width, img_height)

	# Convert to grayscale
	grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
	
	# Save grayscale gradients
	save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray', img_width, img_height)

	print('Vanilla backprop completed')
