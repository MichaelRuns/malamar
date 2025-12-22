"""
Improved calibration tool with better UX
Uses keyboard controls instead of clunky mouse dragging
"""

import cv2
import numpy as np
from perception_module import PerceptionModule
import json
import time

class ImprovedCalibrator:
    """
    Much better calibration experience:
    - Capture a single frame first
    - Use arrow keys to position/resize boxes
    - Press Enter to save region
    - Visual feedback and guides
    """
    
    def __init__(self):
        self.perception = PerceptionModule()
        self.reference_frame = None
        self.regions = {}
        
        # Current box being edited
        self.current_box = {
            'x': 100,
            'y': 100, 
            'width': 100,
            'height': 30
        }
        
        self.region_names = [
            'player_pokemon_name',
            'player_level',
            'player_hp',
            'opponent_pokemon_name',
            'opponent_level',
            'opponent_hp_bar',
            'move_1',
            'move_2',
            'move_3',
            'move_4'
        ]
        
        self.current_region_idx = 0
        self.mode = 'position'  # 'position' or 'resize'
        self.step_size = 5
        
    def capture_reference_frame(self):
        """Capture and save a reference frame"""
        print("\n" + "="*60)
        print("STEP 1: Capture Reference Frame")
        print("="*60)
        print("\nMake sure you're IN A BATTLE in Pokemon")
        print("Press any key when ready...")
        
        cv2.namedWindow('Reference Frame Capture')
        
        while True:
            output = self.perception.perceive()
            frame = output.raw_frame.copy()
            
            # Add text overlay
            cv2.putText(frame, "Press SPACE to capture, Q to quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Reference Frame Capture', frame)
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord(' '):  # Space to capture
                self.reference_frame = output.raw_frame.copy()
                cv2.imwrite('reference_battle_screen.png', self.reference_frame)
                print("✓ Captured reference frame")
                print("  Saved as: reference_battle_screen.png")
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def calibrate_regions(self):
        """Interactive region calibration with keyboard controls"""
        if self.reference_frame is None:
            print("Error: No reference frame captured")
            return
        
        print("\n" + "="*60)
        print("STEP 2: Calibrate Regions")
        print("="*60)
        print("\nKEYBOARD CONTROLS:")
        print("  Arrow Keys    - Move box (in position mode)")
        print("  Arrow Keys    - Resize box (in resize mode)")
        print("  TAB           - Switch between position/resize mode")
        print("  +/-           - Increase/decrease step size")
        print("  ENTER         - Save region and move to next")
        print("  T             - Test OCR on current box")
        print("  S             - Skip current region")
        print("  Q             - Quit calibration")
        print("  R             - Reset current box")
        print("\n")
        
        cv2.namedWindow('Region Calibration')
        
        while self.current_region_idx < len(self.region_names):
            region_name = self.region_names[self.current_region_idx]
            
            # Create display frame
            display = self.reference_frame.copy()
            self._draw_interface(display, region_name)
            
            cv2.imshow('Region Calibration', display)
            key = cv2.waitKey(50) & 0xFF
            
            # Handle keypresses
            if key == ord('q'):
                break
            elif key == 9:  # TAB
                self.mode = 'resize' if self.mode == 'position' else 'position'
            elif key == ord('+') or key == ord('='):
                self.step_size = min(20, self.step_size + 1)
            elif key == ord('-'):
                self.step_size = max(1, self.step_size - 1)
            elif key == 13:  # ENTER
                self._save_current_region(region_name)
            elif key == ord('s'):
                print(f"Skipped: {region_name}")
                self.current_region_idx += 1
            elif key == ord('t'):
                self._test_ocr()
            elif key == ord('r'):
                self._reset_box()
            elif key == 82:  # Up arrow
                self._handle_arrow('up')
            elif key == 84:  # Down arrow
                self._handle_arrow('down')
            elif key == 81:  # Left arrow
                self._handle_arrow('left')
            elif key == 83:  # Right arrow
                self._handle_arrow('right')
        
        cv2.destroyAllWindows()
        self._save_calibration()
    
    def _draw_interface(self, frame, region_name):
        """Draw the calibration interface"""
        box = self.current_box
        
        # Draw current box
        color = (0, 255, 255) if self.mode == 'position' else (255, 0, 255)
        cv2.rectangle(frame, 
                     (box['x'], box['y']),
                     (box['x'] + box['width'], box['y'] + box['height']),
                     color, 2)
        
        # Draw crosshair in center
        center_x = box['x'] + box['width'] // 2
        center_y = box['y'] + box['height'] // 2
        cv2.drawMarker(frame, (center_x, center_y), color, 
                      cv2.MARKER_CROSS, 20, 2)
        
        # Draw grid lines for alignment
        h, w = frame.shape[:2]
        for i in range(0, w, 50):
            cv2.line(frame, (i, 0), (i, h), (50, 50, 50), 1)
        for i in range(0, h, 50):
            cv2.line(frame, (0, i), (w, i), (50, 50, 50), 1)
        
        # Info panel
        info_bg = np.zeros((150, w, 3), dtype=np.uint8)
        info_bg[:] = (40, 40, 40)
        
        y_offset = 25
        cv2.putText(info_bg, f"Region: {region_name} ({self.current_region_idx + 1}/{len(self.region_names)})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 25
        mode_text = "POSITION MODE" if self.mode == 'position' else "RESIZE MODE"
        mode_color = (0, 255, 255) if self.mode == 'position' else (255, 0, 255)
        cv2.putText(info_bg, f"Mode: {mode_text} (TAB to switch)",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        y_offset += 25
        cv2.putText(info_bg, f"Box: x={box['x']}, y={box['y']}, w={box['width']}, h={box['height']}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(info_bg, f"Step size: {self.step_size} (use +/- to adjust)",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(info_bg, "Press T to test OCR | ENTER to save | S to skip",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Combine
        combined = np.vstack([frame, info_bg])
        frame[:] = combined[:frame.shape[0], :]
    
    def _handle_arrow(self, direction):
        """Handle arrow key presses"""
        step = self.step_size
        box = self.current_box
        
        if self.mode == 'position':
            if direction == 'up':
                box['y'] = max(0, box['y'] - step)
            elif direction == 'down':
                box['y'] = min(self.reference_frame.shape[0] - box['height'], box['y'] + step)
            elif direction == 'left':
                box['x'] = max(0, box['x'] - step)
            elif direction == 'right':
                box['x'] = min(self.reference_frame.shape[1] - box['width'], box['x'] + step)
        else:  # resize mode
            if direction == 'up':
                box['height'] = max(10, box['height'] - step)
            elif direction == 'down':
                box['height'] = min(200, box['height'] + step)
            elif direction == 'left':
                box['width'] = max(10, box['width'] - step)
            elif direction == 'right':
                box['width'] = min(300, box['width'] + step)
    
    def _test_ocr(self):
        """Test OCR on current box with debug output"""
        box = self.current_box
        region = (box['x'], box['y'], box['width'], box['height'])
        
        print("\n" + "="*50)
        print("OCR DEBUGGING")
        print("="*50)
        
        # Extract region
        x, y, w, h = region
        roi = self.reference_frame[y:y+h, x:x+w]
        
        # Show original
        cv2.imshow('Original ROI', cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        
        # Test with enhanced OCR
        text = self.perception._ocr_region(self.reference_frame, region, debug=True)
        
        print(f"\nFinal Result: '{text}'")
        
        if not text.strip():
            print("\n⚠ OCR Failed! Suggestions:")
            print("  1. Make box tighter around text")
            print("  2. Make box slightly larger to include more context")
            print("  3. Check if text is clear in the reference frame")
            print("  4. Consider using template matching instead")
            print("  5. Increase melonDS window size for clearer text")
        else:
            print("\n✓ OCR Success!")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _save_current_region(self, region_name):
        """Save current region and move to next"""
        box = self.current_box
        region = (box['x'], box['y'], box['width'], box['height'])
        
        # Test OCR
        text = self.perception._ocr_region(self.reference_frame, region)
        
        self.regions[region_name] = region
        print(f"\n✓ Saved {region_name}: {region}")
        print(f"  OCR Result: '{text}'")
        
        self.current_region_idx += 1
        
        # Suggest next position based on typical layout
        self._smart_reposition()
    
    def _reset_box(self):
        """Reset box to default position"""
        self.current_box = {
            'x': 100,
            'y': 100,
            'width': 100,
            'height': 30
        }
        print("Box reset to default")
    
    def _smart_reposition(self):
        """Intelligently reposition box for next region"""
        region_name = self.region_names[self.current_region_idx - 1] if self.current_region_idx > 0 else None
        
        # Smart positioning based on what we just calibrated
        if region_name == 'player_pokemon_name':
            # Level is usually right next to name
            self.current_box['x'] += self.current_box['width'] + 5
            self.current_box['width'] = 40
        elif region_name == 'player_level':
            # HP is usually below
            self.current_box['y'] += 20
            self.current_box['width'] = 60
        elif region_name == 'player_hp':
            # Jump to opponent side (top-left)
            self.current_box['x'] = 10
            self.current_box['y'] = 30
            self.current_box['width'] = 80
        elif region_name == 'opponent_level':
            # Move to moves (bottom screen area)
            self.current_box['x'] = 10
            self.current_box['y'] = self.reference_frame.shape[0] // 2 + 20
            self.current_box['width'] = 100
            self.current_box['height'] = 25
    
    def _save_calibration(self):
        """Save calibration to JSON file"""
        if not self.regions:
            print("\nNo regions were saved")
            return
        
        # Save as JSON
        with open('calibration.json', 'w') as f:
            json.dump(self.regions, f, indent=2)
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        print(f"\n✓ Saved {len(self.regions)} regions to calibration.json")
        print("\nTo use these in your code:")
        print("\n# Add to perception_module.py __init__:")
        print("self.regions = {")
        for name, region in self.regions.items():
            print(f"    '{name}': {region},")
        print("}")
    
    def run(self):
        """Run the full calibration process"""
        print("\n" + "="*60)
        print("POKEMON PERCEPTION CALIBRATION TOOL")
        print("="*60)
        
        # Step 1: Capture reference frame
        if not self.capture_reference_frame():
            return
        
        time.sleep(0.5)
        
        # Step 2: Calibrate regions
        self.calibrate_regions()

if __name__ == "__main__":
    calibrator = ImprovedCalibrator()
    calibrator.run()