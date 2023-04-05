from diffusers import StableDiffusionPipeline

# Load the model
model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")

# Prompt
prompt = input("Enter a prompt: ")
prompt = prompt + 'mdjrny-v4 style'

# Generate image
image = pipe(prompt).images[0]
image.save("image.png")