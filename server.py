import base64
import pickle

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, DEISMultistepScheduler
from fastapi import FastAPI
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


# generic text 2 image
def txt2img(model_id, prompt, negative_prompt, num_inference_steps=25, height=720, width=1088):
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True,
                                                        timestep_spacing="trailing")
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_sequential_cpu_offload()
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, height=height,
                 width=width).images[0]
    image.save('right_image.jpg')


    return image


# generic image 2 image
def img2img(model_id, image, prompt, negative_prompt, num_inference_steps=25):
    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_sequential_cpu_offload()
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=num_inference_steps,
                 strength=.2).images[0]
    image.save("left_image.jpg")


    return image


def normal_merger(l_image, r_image):
    # Open both images using Pillow
    image1 = l_image
    image2 = r_image

    # Get the dimensions (width and height) of both images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate the width and height of the combined image
    combined_width = width1 + width2
    combined_height = max(height1, height2)  # Choose the maximum height

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new("RGB", (combined_width, combined_height))

    # Paste the first image on the left side of the combined image
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right side of the combined image
    combined_image.paste(image2, (width1, 0))

    # Save the combined image
    combined_image.save("./generated_combined.jpg")


def similar_merger(l_image, r_image):
    # Open both images using Pillow
    image1 = l_image
    image2 = r_image

    # Get the dimensions (width and height) of both images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate the width and height of the combined image
    combined_width = width1 + width2
    combined_height = max(height1, height2)  # Choose the maximum height

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new("RGB", (combined_width, combined_height))

    # Paste the first image on the left side of the combined image
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right side of the combined image
    combined_image.paste(image2, (width1, 0))

    # Save the combined image
    combined_image.save("./copy_combined.jpg")
    return combined_image


def banner_adder(base_image):
    image_to_add = Image.open("Banner.jpg")
    base_width, base_height = base_image.size
    print(base_width, base_height)
    image_to_add = image_to_add.resize((base_width, image_to_add.height))
    padding_height = int(2 * image_to_add.height)
    new_height = base_height + padding_height
    new_image = Image.new("RGB", (base_width, new_height))
    new_image.paste(image_to_add, (0, 0))
    new_image.paste(base_image, (0, image_to_add.height))
    new_image.paste(image_to_add, (0, image_to_add.height + base_height))
    return new_image


def mask_maker(image):
    width, height = image.size
    black_mask_image = Image.new("RGB", (width, height), (0, 0, 0))
    padding_height = int(0.25 * height)
    new_height = height + 2 * padding_height
    new_image = Image.new("RGB", (width, new_height), (0, 0, 0))  # Use (255, 255, 255) for white background
    white_mask_image = Image.new("RGB", (width, new_height), (255, 255, 255))
    x_offset = 0
    white_mask_image.paste(black_mask_image, (x_offset, padding_height))
    new_image.paste(image, (x_offset, padding_height))
    white_mask_image.save("mask_image.png")
    new_image.save("padded_image.png")
    return white_mask_image, new_image


# for text to image

# photo anime, masterpiece, high quality, absurdres use imagination than real places ->dreamlike-anime-1.0
# redshift style -> nitrosocke/redshift-diffusion-768
# cinematic, colorful background, concept art, dramatic lighting, high detail, highly detailed, hyper realistic, intricate, intricate sharp details, octane render, smooth, studio lighting, trending on artstation -> lykon/dreamshaper-8


provider = "sd"
with open("provider.pkl", "wb") as p:
    pickle.dump(provider, p)

model_options = {"reality": "lykon/dreamshaper-8", "anime": "dreamlike-art/dreamlike-anime-1.0",
                 "mystical": "nitrosocke/redshift-diffusion"}
NEGATIVE_PROMPT = "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime"

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class Item(BaseModel):
    model: str
    prompt: str


class Provider(BaseModel):
    provider: str


@app.post('/change_provider')
async def change_provider(prompt_model: Provider) -> Response:
    with open("provider.pkl", "wb") as p:
        pickle.dump(prompt_model.provider, p)


@app.post('/')
async def root(prompt_model: Item) -> Response:
    image_base64 = " "
    prompt = prompt_model.prompt
    model = prompt_model.model
    with open("provider.pkl", "rb") as p:
        provider = pickle.load(p)
    if provider == "sd":
        if model in model_options:
            print(f"{model} Model found... generating the image")
            right_image = txt2img(model_options[model], prompt, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=25,
                                  height=720 * 2, width=1088 * 2)
            # white_mask_image, new_image = mask_maker(right_image)
            print("Image is now ready, creating the mirror of it...\n")
            # inpainted_image = inpainter(new_image,white_mask_image,prompt,negative_prompt=NEGATIVE_PROMPT)
            flipped_right = right_image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_right.save('.flipped_right.jpg')
            print("Modifying the mirror image\n")
            left_image = img2img(model_options[model], flipped_right, prompt, negative_prompt=NEGATIVE_PROMPT,
                                 num_inference_steps=30)
            print("Merging both images")
            normal_merger(left_image, right_image)
            img = similar_merger(flipped_right, right_image)
            img = banner_adder(img)
            img.save("bannered_image.jpg")
            with open("bannered_image.jpg", "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            # return FileResponse('bannered_image.jpg')
            return image_base64
        else:
            return Response("model not found!!")
    elif provider == "bd":
        print("Blockade labs not yet integrated....")
        return Response("Blockade labs not yet integrated....")
    else:
        return Response("Invalid provider")
