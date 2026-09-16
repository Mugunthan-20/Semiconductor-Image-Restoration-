"""
Evaluation Script for Semiconductor Image Restoration
This script loads the trained model and restores all test images.
USAGE: python evaluate.py --input_dir <path_to_test_images> --output_dir <path_to_output>
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import time
from model import SuperResolutionNet

def evaluate(input_dir, output_dir):
    """
    Load trained model and restore all test images
    
    Args:
        input_dir: Path to directory containing .npy test images
        output_dir: Path to directory where restored images will be saved
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n[1/3] Loading trained model...")
    model = SuperResolutionNet().to(device)
    
    # Try to load model weights
    model_path = Path('models/best_model.pth')
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_path}")
    
    # Process all test images
    print("\n[2/3] Processing test images...")
    input_path = Path(input_dir)
    test_files = sorted(input_path.glob('*.npy'))
    
    if not test_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")
    
    print(f"Found {len(test_files)} test images\n")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, test_file in enumerate(test_files):
            try:
                # Load noisy image
                noisy_np = np.load(test_file).astype(np.float32)
                
                # Convert to tensor
                noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).to(device)
                
                # Restore
                restored_tensor = model(noisy_tensor)
                
                # Convert back to numpy
                restored_np = restored_tensor.squeeze().cpu().numpy().astype(np.float32)
                
                # Handle numerical issues but don't clip
                # NoisyLR may have values > 1.0, model learned this pattern
                restored_np = np.nan_to_num(restored_np, nan=0.5, posinf=1.0, neginf=0.0)
                restored_np = np.clip(restored_np, 0.0, 1.0)
                
                # Save
                output_file = output_path / test_file.name
                np.save(output_file, restored_np)
                
                # Progress
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{i+1}/{len(test_files)}] Processed in {elapsed:.1f}s")
            
            except Exception as e:
                print(f"ERROR processing {test_file.name}: {e}")
                continue
    
    elapsed_total = time.time() - start_time
    
    # Verify outputs
    print("\n[3/3] Verification...")
    output_files = list(output_path.glob('*.npy'))
    print(f"✓ Successfully saved {len(output_files)} restored images")
    print(f"✓ Total time: {elapsed_total:.1f}s")
    print(f"✓ Average time per image: {elapsed_total/len(test_files):.2f}s")
    
    if output_files:
        sample = np.load(output_files[0])
        print(f"✓ Sample output shape: {sample.shape}")
        print(f"✓ Sample output range: [{sample.min():.4f}, {sample.max():.4f}]")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Semiconductor Image Restoration - Evaluation Script'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing test images (.npy files)'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to directory where restored images will be saved'
    )
    
    args = parser.parse_args()
    
    evaluate(args.input_dir, args.output_dir)