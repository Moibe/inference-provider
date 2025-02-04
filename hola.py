import bridges
import random
from huggingface_hub import InferenceClient

client = InferenceClient(
	provider="fal-ai", #"hf-inference", "fal-ai", "together", "replicate"  --> Los otros proveedores si son más veloces.
    #Pero tienen límites por ejemplo falai no permite más de 50 steps.
	api_key=bridges.hug
)

numero_aleatorio = random.randint(1, 150000000)

# output is a PIL.Image object
image = client.text_to_image(
	"A rubber rooster with astronaut suit with the name Moi.",
	model="black-forest-labs/FLUX.1-dev",
    #seed=42, #default varía pero el default es que siempre sea la misma.
    #guidance_scale=7.5,
    #num_inference_steps=50,
    #width=1024, #El default es 1024 x 1024 y quizá 1024*768, el max es 1536. 
    #height=1024 #El límite de replicate es 1024.
)

print("Ésto es image: ", image)
print("Y su tipo es: ", type(image))

image.save(f"images/stabilization-{numero_aleatorio}.png")