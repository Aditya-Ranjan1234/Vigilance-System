import cv2
import numpy as np
import os

def create_placeholder(width=640, height=480, text="Select an algorithm and click Start"):
    """Create a placeholder image with text."""
    # Create a black image
    img = np.zeros((height, width, 3), np.uint8)
    
    # Add a dark gray background
    img[:] = (30, 30, 30)
    
    # Add a border
    cv2.rectangle(img, (10, 10), (width-10, height-10), (50, 50, 50), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, (200, 200, 200), 2, cv2.LINE_AA)
    
    # Add a camera icon
    icon_size = 50
    icon_x = (width - icon_size) // 2
    icon_y = text_y - 80
    
    # Draw camera body
    cv2.rectangle(img, (icon_x, icon_y), (icon_x + icon_size, icon_y + icon_size//2), (100, 100, 100), -1)
    
    # Draw camera lens
    center_x = icon_x + icon_size // 2
    center_y = icon_y + icon_size // 4
    cv2.circle(img, (center_x, center_y), icon_size // 6, (150, 150, 150), -1)
    cv2.circle(img, (center_x, center_y), icon_size // 10, (80, 80, 80), -1)
    
    return img

if __name__ == "__main__":
    # Create placeholder image
    placeholder = create_placeholder()
    
    # Save the image
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "placeholder.jpg")
    cv2.imwrite(output_path, placeholder)
    
    print(f"Placeholder image created at {output_path}")
