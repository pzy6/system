"""
Test CUDA initialization for ONNX Runtime
Run this to verify if CUDA is properly configured
"""
import onnxruntime as ort
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("CUDA Initialization Test")
print("=" * 60)

print(f"\nONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Test with actual ONNX model
model_path = "models/emotion/efficient_emotion_int8.onnx"
if not os.path.exists(model_path):
    model_path = "models/fall_detection/stgcn_fall_int8.onnx"

if os.path.exists(model_path):
    print(f"\nTesting with model: {model_path}")
    
    # Test CUDA session creation
    print("\n" + "=" * 60)
    print("Testing CUDA session creation...")
    print("=" * 60)

    options = ort.SessionOptions()
    options.log_severity_level = 3

    try:
        # Try to create a session with CUDA provider
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(f"✓ Session created successfully")
        print(f"  Active providers: {session.get_providers()}")
        
        if "CUDAExecutionProvider" in session.get_providers():
            print("\n✓ CUDA is working! GPU acceleration enabled.")
        else:
            print("\n✗ CUDA not available. Using CPU fallback.")
            print("  This means cuDNN is not properly configured.")
    except Exception as e:
        print(f"✗ Session creation failed: {e}")
        print("\nThis is expected if cuDNN is not installed.")
        print("The system will fall back to CPU automatically.")
else:
    print(f"\nModel not found: {model_path}")
    print("Skipping CUDA test with actual model.")

print("\n" + "=" * 60)
print("cuDNN Installation Instructions")
print("=" * 60)
print("""
If CUDA initialization failed above, you need to install cuDNN 9:

1. Download cuDNN 9.x for CUDA 13 from:
   https://developer.nvidia.com/cudnn
   (Requires NVIDIA Developer account)

2. Extract the downloaded zip file to a local directory, e.g.:
   C:\\tools\\cudnn

3. Add the cuDNN bin directory to your system PATH:
   C:\\tools\\cudnn\\bin

4. Restart your terminal and run this test again.

5. If successful, you should see:
   - No CUDA initialization errors
   - Active providers include CUDAExecutionProvider
""")
