"""
Template matching system for Pokemon recognition
Much more reliable than OCR for known Pokemon/moves
"""

import cv2
import numpy as np
import os
import json
from typing import Tuple, List, Optional, Dict

class TemplateLibrary:
    """
    Manages template images for matching Pokemon, moves, etc.
    Templates are saved images of known text that we match against
    """
    
    def __init__(self, template_dir='templates'):
        self.template_dir = template_dir
        self.templates = {}
        self.metadata = {}
        
        # Create directory if it doesn't exist
        os.makedirs(template_dir, exist_ok=True)
        
        # Load existing templates
        self.load_all_templates()
    
    def save_template(self, image: np.ndarray, name: str, category: str = 'pokemon'):
        """
        Save a new template
        
        Args:
            image: The cropped region containing the text/image
            name: Name of the template (e.g., "TEPIG", "TACKLE")
            category: Category (pokemon, move, item, etc.)
        """
        filename = f"{category}_{name}.png"
        filepath = os.path.join(self.template_dir, filename)
        
        # Convert to grayscale and threshold for consistent matching
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(filepath, binary)
        
        # Save metadata
        self.metadata[name] = {
            'category': category,
            'filename': filename,
            'size': image.shape[:2]
        }
        
        self._save_metadata()
        self.templates[name] = binary
        
        print(f"✓ Saved template: {name} ({category})")
    
    def load_all_templates(self):
        """Load all templates from disk"""
        metadata_path = os.path.join(self.template_dir, 'metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load template images
        for name, meta in self.metadata.items():
            filepath = os.path.join(self.template_dir, meta['filename'])
            if os.path.exists(filepath):
                self.templates[name] = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if self.templates:
            print(f"Loaded {len(self.templates)} templates")
    
    def _save_metadata(self):
        """Save metadata to JSON"""
        metadata_path = os.path.join(self.template_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def match_template(self, image: np.ndarray, threshold: float = 0.8) -> Optional[str]:
        """
        Try to match the image against all templates
        
        Returns:
            Name of matched template, or None if no match
        """
        # Preprocess input image same way as templates
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        best_match = None
        best_score = threshold
        
        for name, template in self.templates.items():
            # Try multiple scales
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                # Resize template
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                
                if new_h > binary.shape[0] or new_w > binary.shape[1]:
                    continue
                
                resized_template = cv2.resize(template, (new_w, new_h))
                
                # Try matching
                try:
                    result = cv2.matchTemplate(binary, resized_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    if max_val > best_score:
                        best_score = max_val
                        best_match = name
                except:
                    continue
        
        return best_match
    
    def match_in_region(self, frame: np.ndarray, region: Tuple[int, int, int, int],
                       threshold: float = 0.8) -> Optional[str]:
        """Match template in a specific region of the frame"""
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        return self.match_template(roi, threshold)

class TemplateCollector:
    """
    Interactive tool to build template library from game footage
    """
    
    def __init__(self, library: TemplateLibrary):
        self.library = library
        self.reference_frame = None
        self.current_box = {'x': 100, 'y': 100, 'width': 100, 'height': 30}
    
    def collect_from_frame(self, frame: np.ndarray):
        """
        Interactive collection from a frame
        """
        self.reference_frame = frame
        print("\n" + "="*60)
        print("TEMPLATE COLLECTION MODE")
        print("="*60)
        print("\nControls:")
        print("  Arrow Keys - Move/resize box")
        print("  TAB - Switch position/resize mode")
        print("  S - Save template")
        print("  Q - Quit")
        
        cv2.namedWindow('Template Collection')
        mode = 'position'
        step = 5
        
        while True:
            display = frame.copy()
            
            # Draw box
            box = self.current_box
            color = (0, 255, 255) if mode == 'position' else (255, 0, 255)
            cv2.rectangle(display, (box['x'], box['y']),
                         (box['x'] + box['width'], box['y'] + box['height']),
                         color, 2)
            
            # Show mode
            cv2.putText(display, f"Mode: {mode.upper()} (TAB to switch)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, "Press S to save template",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Template Collection', display)
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 9:  # TAB
                mode = 'resize' if mode == 'position' else 'position'
            elif key == ord('s'):
                self._save_template()
            elif key == 82:  # Up
                if mode == 'position':
                    box['y'] = max(0, box['y'] - step)
                else:
                    box['height'] = max(10, box['height'] - step)
            elif key == 84:  # Down
                if mode == 'position':
                    box['y'] += step
                else:
                    box['height'] += step
            elif key == 81:  # Left
                if mode == 'position':
                    box['x'] = max(0, box['x'] - step)
                else:
                    box['width'] = max(10, box['width'] - step)
            elif key == 83:  # Right
                if mode == 'position':
                    box['x'] += step
                else:
                    box['width'] += step
        
        cv2.destroyAllWindows()
    
    def _save_template(self):
        """Save current box as template"""
        box = self.current_box
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        
        roi = self.reference_frame[y:y+h, x:x+w]
        
        # Show preview
        cv2.imshow('Template Preview', roi)
        cv2.waitKey(500)
        
        print("\nSave this as template?")
        name = input("Enter name (e.g., TEPIG, TACKLE): ").strip().upper()
        
        if not name:
            print("Cancelled")
            return
        
        category = input("Category (pokemon/move/item) [pokemon]: ").strip().lower()
        if not category:
            category = 'pokemon'
        
        self.library.save_template(roi, name, category)

def build_starter_library():
    """
    Helper to build a basic template library for BW2 starters
    """
    print("\n" + "="*60)
    print("STARTER TEMPLATE LIBRARY BUILDER")
    print("="*60)
    print("\nThis will help you build templates for:")
    print("  - Starter Pokemon (Tepig, Oshawott, Snivy)")
    print("  - Common moves")
    print("  - Early wild Pokemon")
    print("\n1. Start Pokemon BW2")
    print("2. Get into battles with different Pokemon")
    print("3. Capture frames and save templates")
    
    library = TemplateLibrary()
    collector = TemplateCollector(library)
    
    from perception_module import PerceptionModule
    perception = PerceptionModule()
    
    input("\nPress Enter to start capturing...")
    
    cv2.namedWindow('Live Capture')
    
    while True:
        output = perception.perceive()
        frame = output.raw_frame.copy()
        
        cv2.putText(frame, "Press SPACE to capture for template collection",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to finish",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Live Capture', frame)
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord(' '):
            cv2.destroyWindow('Live Capture')
            collector.collect_from_frame(output.raw_frame)
            cv2.namedWindow('Live Capture')
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"\n✓ Template library created with {len(library.templates)} templates")

if __name__ == "__main__":
    print("Template Matching System")
    print("1. Build template library (first time setup)")
    print("2. Test template matching")
    
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == '1':
        build_starter_library()
    elif choice == '2':
        library = TemplateLibrary()
        print(f"Loaded {len(library.templates)} templates")
        print("Templates:", list(library.templates.keys()))