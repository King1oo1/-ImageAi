from flask import Flask, request, send_file, jsonify
from diffusers import DiffusionPipeline
import torch
import io

app = Flask(__name__)

# Load SDXL base + refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
refiner.to("cuda")

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        n_steps = 40
        high_noise_frac = 0.8

        # Run base
        base_output = base(prompt=prompt, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type="latent")
        # Run refiner
        image = refiner(prompt=prompt, num_inference_steps=n_steps, denoising_start=high_noise_frac, image=base_output.images).images[0]

        # Convert to bytes
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
