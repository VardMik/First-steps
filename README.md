# First-steps
first steps in AI
# This bot draws pictures, he is shocked)
#@title Необходимые функции
!pip install jax==0.4.23 jaxlib==0.4.23
!pip -q install diffusers
!pip -q install transformers scipy ftfy accelerate
!pip -q install "ipywidgets>=7,<8"
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from google.colab import output
output.enable_custom_widget_manager()

stableDiffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
stableDiffusion = stableDiffusion.to("cuda")

def createImagesStableDiffusion(prompt='', rows=2, cols=2, iteration=20):
#Let's start generation
images =  stableDiffusion([prompt] * (rows*cols), num_inference_steps=iteration).images
w, h = images[0].size
grid = Image.new('RGB', size=(cols*w, rows*h))
grid_w, grid_h = grid.size

for i, img in enumerate(images):
    grid.paste(img, box=(i%cols*w, i//cols*h))
display(grid)

  # You need to run the code and then enter a command that includes text with the condition that you want the bot to draw, try it and be surprised
  # I will leave examples below

  # Изменяя текст в кавычках получайте различные изображени. (By changing the text in quotes you can get different images.)
  # №1
  createImagesStableDiffusion('Salvador Dali walks down the street with a cockroach on a leash, city, surrealism, crowded, people turn around, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)
  # №2
  createImagesStableDiffusion('4 cats are fighting with pillows in the ring, the ring is on the bed of the owner, who has turned away over a cup of tea, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)
  # №3
  createImagesStableDiffusion('A parrot flies to the moon and waves to the astronauts, the astronauts are in a rocket and look out the window, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)
  # №4
  createImagesStableDiffusion('Draw a picture in the style of Picasso, a girl drinking coffee by a stream, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)
  # №5
  createImagesStableDiffusion('Draw in Davinci style how a million Metairites fly around the earth, the view of the picture should be from the side, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)
  # №6
  createImagesStableDiffusion('draw a futuristic beautiful violin lying on the piano, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)

  # Run these commands in the bot and you will see pictures and we will enjoy the art of AI together
