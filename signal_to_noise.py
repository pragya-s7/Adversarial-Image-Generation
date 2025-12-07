import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import requests
from PIL import Image
from qwen_vl_utils import process_vision_info

# --- 1. CONFIGURATION ---
OUTPUT_DIR = "./demo_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NOISE_STEPS = 10
MAX_NOISE = 1.0  # 1.0 = Pure static
IMG_SIZE = (512, 512) 

print("--- SETTING UP SIGNAL-TO-NOISE DEMO ---")

# Ensure model is in eval mode (deterministic)
model.eval()

# --- 2. THE HOOK (Captures the internal Gate Score) ---
global_gate_values = []

def gate_hook(module, input, output):
    # output is the raw logit from the Linear layer
    # Apply sigmoid to get 0-1 range, detach to save memory
    score = torch.sigmoid(output).detach().cpu().numpy().flatten()
    global_gate_values.extend(score)

# FIX: Find ALL gate modules, but only register the hook on the LAST one.
# The last layer is usually the most semantically meaningful.
gate_modules = [m for n, m in model.named_modules() if "grounding_gate" in n]

if len(gate_modules) > 0:
    # Hook only the very last gate layer
    hook_handle = gate_modules[-1].register_forward_hook(gate_hook)
    print(f"Hook registered on the last grounding gate (Total gates found: {len(gate_modules)})")
else:
    print("WARNING: No 'grounding_gate' found! Is the wrapper applied?")
    hook_handle = None

# --- 3. HELPER: ADD NOISE ---
def add_noise(image, noise_level):
    # Resize to save compute
    img_resized = image.resize(IMG_SIZE)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    
    # Generate Noise
    noise = np.random.normal(loc=0.0, scale=noise_level, size=img_arr.shape)
    
    # Add and Clip
    noisy_img = img_arr + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    
    # Convert back to PIL
    return Image.fromarray((noisy_img * 255).astype(np.uint8))

# --- 4. LOAD IMAGE ---
# Try to get the "Bear" image (Image 12), fallback to download if needed
try:
    # Assuming val_data_static exists from previous cells
    base_image = val_data_static[12]['images'][0] 
    print("Loaded Image 12 from local dataset.")
except:
    print("Dataset not in memory, downloading sample image...")
    url = "https://images.unsplash.com/photo-1575485670541-824ff288aaf8?q=80&w=1000&auto=format&fit=crop"
    base_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

print(f"Starting Stress Test ({NOISE_STEPS} steps)...")

results = []

# --- 5. THE LOOP ---
for i in range(NOISE_STEPS + 1):
    # Calculate noise level
    noise_level = (i / NOISE_STEPS) * MAX_NOISE
    
    # Create Noisy Image
    current_image = add_noise(base_image, noise_level)
    
    # Prepare Inputs
    conversation = [
        {"role": "user", "content": [
            {"type": "image", "image": current_image},
            {"type": "text", "text": "What is this object?"}
        ]}
    ]
    
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Reset hook container for this pass
    global_gate_values = [] 
    
    # Run Inference
    with torch.no_grad():
        # Generate just a few tokens to trigger the hooks
        generated_ids = model.generate(**inputs, max_new_tokens=5, use_cache=False)
    
    # Calculate Confidence
    if len(global_gate_values) > 0:
        avg_confidence = float(np.mean(global_gate_values))
    else:
        avg_confidence = 0.0 # Fallback if hook failed
    
    # Decode text output (just for display)
    out_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if "assistant\n" in out_text:
        out_text = out_text.split("assistant\n")[-1]
    
    results.append({
        "noise": noise_level,
        "confidence": avg_confidence,
        "image": current_image,
        "output": out_text
    })
    
    print(f"Step {i}/{NOISE_STEPS} | Noise: {noise_level:.2f} | Gate Confidence: {avg_confidence:.4f}")

    # Aggressive Cleanup for RTX 3050
    del inputs, generated_ids, image_inputs, video_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# --- 6. VISUALIZATION ---
print("Generating Frames...")
plt.style.use('dark_background')

for i, res in enumerate(results):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    # Left: Image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(res['image'])
    ax1.axis('off')
    ax1.set_title(f"Input Signal (Noise: {int(res['noise']*100)}%)", color='white', fontsize=14)
    
    # Right: Bar Chart
    ax2 = fig.add_subplot(gs[1])
    
    conf = res['confidence']
    # Color logic
    if conf > 0.7: bar_color = '#00ff00' # Green
    elif conf > 0.4: bar_color = '#ffff00' # Yellow
    else: bar_color = '#ff0000' # Red
    
    bars = ax2.bar(["Visual Confidence"], [conf], color=bar_color, width=0.4)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Last Layer Gate Score", fontsize=12)
    
    ax2.text(0, conf + 0.02, f"{conf:.2%}", ha='center', color='white', fontsize=16, fontweight='bold')
    
    ax2.text(0, 0.5, f"Output:\n{res['output']}...", 
             ha='center', va='center', transform=ax2.transAxes, 
             fontsize=12, color='#aaaaaa', style='italic')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/frame_{i:03d}.png", dpi=100)
    plt.close(fig) # Important to prevent memory leak in loop

# Safe hook removal
if hook_handle is not None:
    hook_handle.remove()
    print("Hook removed.")

print(f"\nDONE! Frames saved to {OUTPUT_DIR}")