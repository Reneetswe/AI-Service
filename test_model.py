from ultralytics import YOLO
import os

# Test the model directly
try:
    print("Loading model...")
    model = YOLO("models/best.pt")
    print("✅ Model loaded successfully!")
    
    # Print model info
    print(f"Model names: {model.names}")
    
    # Test with a dummy image (should show if model can process)
    import numpy as np
    from PIL import Image
    
    # Create a simple test image
    test_image = Image.new('RGB', (640, 640), color='blue')
    print("Running inference on test image...")
    
    results = model.predict(test_image, conf=0.25, verbose=True)
    print(f"Results: {len(results)} result(s)")
    
    if results and results[0].boxes:
        print(f"Found {len(results[0].boxes)} detections")
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            print(f"  Detection {i}: class={cls}, confidence={conf}")
    else:
        print("No detections found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
