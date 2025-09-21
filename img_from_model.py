import torch
import numpy as np
from PIL import Image
from atmosphere_net import AtmosphereNet
import os

def generate_ray_directions(width, height):
    """Generate ray directions that match JavaScript quad interpolation."""
    directions = []
    
    # Generate directions to match WebGL texture coordinate system
    for y in range(height):
        for x in range(width):
            # Convert to normalized coordinates (-1 to 1)
            nx = (2 * x / (width - 1)) - 1
            # Flip Y coordinate to match WebGL texture coordinates
            # WebGL texture (0,0) is bottom-left, but we generate top-left first
            ny = -((2 * y / (height - 1)) - 1)  # Flipped Y
            nz = -1  # Forward direction
            
            # Normalize the ray direction
            length = np.sqrt(nx * nx + ny * ny + nz * nz)
            directions.append([nx / length, ny / length, nz / length])
    
    return np.array(directions, dtype=np.float32)

def generate_atmosphere_texture():
    """Generate 128x128 atmosphere texture using trained PyTorch model."""
    
    # Fixed size - always 128x128
    width, height = 128, 128
    output_path = 'atmosphere_texture.png'
    model_path = 'best_atmosphere_model.pth'
    
    print("Atmosphere Texture Generator")
    print("=" * 40)
    print(f"Generating {width}x{height} atmosphere texture...")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found!")
        print("Please make sure you have trained and saved the model.")
        return None
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    print("Loading trained model...")
    model = AtmosphereNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Generate ray directions
    print(f"Generating ray directions...")
    ray_directions = generate_ray_directions(width, height)
    print(f"Generated {len(ray_directions)} ray directions")
    
    # Convert to tensor
    ray_tensor = torch.FloatTensor(ray_directions).to(device)
    
    print(f"Running neural network inference...")
    
    # Predict colors in batches
    batch_size = 1024
    all_colors = []
    
    with torch.no_grad():
        for i in range(0, len(ray_directions), batch_size):
            end_idx = min(i + batch_size, len(ray_directions))
            batch = ray_tensor[i:end_idx]
            
            # Get predictions from neural network
            colors = model(batch).cpu().numpy()
            all_colors.extend(colors)
            
            # Progress update
            progress = (end_idx / len(ray_directions)) * 100
            if (i // batch_size + 1) % 5 == 0 or end_idx == len(ray_directions):
                print(f"  Progress: {progress:.1f}% ({end_idx}/{len(ray_directions)} rays)")
    
    # Convert to numpy array and reshape to image format
    colors = np.array(all_colors).reshape(height, width, 3)
    
    # Apply exposure mapping (like in the original GLSL shader)
    print("Applying exposure and gamma correction...")
    colors = 1.0 - np.exp(-1.0 * colors)
    
    # Apply gamma correction for display
    # colors = np.power(np.clip(colors, 0, 1), 1.0 / 2.2)
    
    # Convert to 8-bit image
    image_8bit = (colors * 255).astype(np.uint8)
    
    # Save the image
    img = Image.fromarray(image_8bit, 'RGB')
    img.save(output_path)
    
    print(f"✅ Atmosphere texture saved to: {output_path}")
    print(f"📐 Image size: {width}x{height}")
    
    # Print color statistics
    print(f"\n📊 Color Statistics:")
    print(f"  Red   - min: {colors[:,:,0].min():.4f}, max: {colors[:,:,0].max():.4f}, mean: {colors[:,:,0].mean():.4f}")
    print(f"  Green - min: {colors[:,:,1].min():.4f}, max: {colors[:,:,1].max():.4f}, mean: {colors[:,:,1].mean():.4f}")
    print(f"  Blue  - min: {colors[:,:,2].min():.4f}, max: {colors[:,:,2].max():.4f}, mean: {colors[:,:,2].mean():.4f}")
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"💾 File size: {file_size:,} bytes")
    
    return img

def preview_image():
    """Show a preview of the generated image."""
    image_path = 'atmosphere_texture.png'
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        img = Image.open(image_path)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title('Generated Atmosphere Texture (128x128)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print("🖼️  Image preview displayed!")
        
    except ImportError:
        print("ℹ️  matplotlib not available for preview")
        print(f"✅ You can view the image manually: {image_path}")
    except Exception as e:
        print(f"❌ Could not preview image: {e}")

if __name__ == "__main__":
    try:
        # Generate the texture
        img = generate_atmosphere_texture()
        
        if img is not None:
            print("\n" + "="*40)
            print("🎉 Texture generation completed successfully!")
            print("="*40)
            
            # Try to show preview
            preview_image()
            
            print("\n📝 Next steps:")
            print("1. Make sure 'atmosphere_texture.png' is in your JavaScript project directory")
            print("2. Run your JavaScript application")
            print("3. The texture should load automatically!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()