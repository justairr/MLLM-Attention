from vlmeval.config import supported_VLM
model = supported_VLM['Qwen2-VL-7B-Instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
# ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
# print(ret)  # There are two apples in the provided images.