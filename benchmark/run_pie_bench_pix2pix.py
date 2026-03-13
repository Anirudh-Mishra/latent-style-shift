import os
import torch
import json
import argparse
import math
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True, help="Path to PIE-Bench dataset root")
    parser.add_argument('--target_path', type=str, required=True, help="Path to save edited images")
    parser.add_argument('--model_id', type=str, default="timbrooks/instruct-pix2pix", help="InstructPix2Pix model ID")
    parser.add_argument('--num_inference_steps', type=int, default=20, help="Number of inference steps")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Text guidance scale")
    parser.add_argument('--image_guidance_scale', type=float, default=1.5, help="Image guidance scale")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading pipeline from {args.model_id}...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id, 
        torch_dtype=torch_dtype, 
        safety_checker=None
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    root = args.source_path
    target = args.target_path
    
    annotation_file_name = os.path.join(root, "mapping_file.json")
    if not os.path.exists(annotation_file_name):
        print(f"Error: {annotation_file_name} not found.")
        return

    with open(annotation_file_name, "r") as f:
        annotation_file = json.load(f)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for annotation_idx, annotation in tqdm(annotation_file.items(), desc="Running PIE-Bench"):
        img_rel_path = annotation["image_path"]
        img_path = os.path.join(root, "annotation_images", img_rel_path)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue

        image_in = Image.open(img_path).convert("RGB")
        
        # InstructPix2Pix usually works well with 512x512
        # We can resize if needed, but let's follow the original script's logic if applicable
        # Or just use the pipe's internal handling
        
        instruction = annotation["editing_instruction"]
        
        with torch.no_grad():
            output = pipe(
                prompt=instruction,
                image=image_in,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                generator=generator
            )
            image_out = output.images[0]

        # Save the image
        out_full_path = os.path.join(target, "annotation_images", img_rel_path)
        os.makedirs(os.path.dirname(out_full_path), exist_ok=True)
        image_out.save(out_full_path)

    print(f"Done! Results saved to {target}")

if __name__ == "__main__":
    main()
