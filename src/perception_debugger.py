"""
Debugging tool for calibrating the perception layer
Helps you:
1. Find correct screen regions
2. Test OCR accuracy
3. Visualize detection regions
4. Save training templates
"""

import cv2
import numpy as np
from eyes import PerceptionModule, GameState
import time

class PerceptionDebugger:
    """Interactive debugging tool for perception calibration"""
    
    def __init__(self):
        self.perception = PerceptionModule()
        self.current_frame = None
        self.regions = []
        self.drawing = False
        self.start_point = None
        
    def run_interactive_capture(self):
        """
        Capture frames and display with annotations
        Press keys:
        - 's': Save current frame
        - 'q': Quit
        - 'r': Draw region (click and drag)
        - 't': Test OCR on drawn region
        """
        print("Starting interactive capture...")
        print("Controls:")
        print("  's' - Save frame")
        print("  'q' - Quit")
        print("  'r' - Draw region for OCR test")
        print("  'c' - Clear regions")
        
        cv2.namedWindow('Pokemon Perception Debug')
        cv2.setMouseCallback('Pokemon Perception Debug', self._mouse_callback)
        
        while True:
            output = self.perception.perceive()
            self.current_frame = output.raw_frame.copy()
            
            # Draw overlays
            self._draw_debug_info(self.current_frame, output)
            
            cv2.imshow('Pokemon Perception Debug', self.current_frame)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_frame(output.raw_frame)
            elif key == ord('c'):
                self.regions = []
                print("Cleared regions")
        
        cv2.destroyAllWindows()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                # Draw temporary rectangle
                pass
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                region = (
                    min(self.start_point[0], x),
                    min(self.start_point[1], y),
                    abs(x - self.start_point[0]),
                    abs(y - self.start_point[1])
                )
                self.regions.append(region)
                print(f"Added region: {region}")
                
                # Test OCR on this region
                if self.current_frame is not None:
                    text = self.perception._ocr_region(self.current_frame, region)
                    print(f"OCR Result: '{text}'")
    
    def _draw_debug_info(self, frame: np.ndarray, output):
        """Draw debug overlays on frame"""
        h, w = frame.shape[:2]
        
        # Draw game state
        state_color = {
            GameState.BATTLE: (0, 255, 0),
            GameState.OVERWORLD: (255, 0, 0),
            GameState.DIALOGUE: (0, 255, 255),
            GameState.MENU: (255, 255, 0),
            GameState.UNKNOWN: (128, 128, 128)
        }
        
        color = state_color.get(output.game_state, (255, 255, 255))
        cv2.putText(frame, f"State: {output.game_state.value}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw battle info if available
        if output.battle_state:
            bs = output.battle_state
            info_lines = [
                f"Player: {bs.player_pokemon} Lv.{bs.player_level}",
                f"HP: {bs.player_hp}/{bs.player_hp_max}",
                f"Opponent: {bs.opponent_pokemon} Lv.{bs.opponent_level}",
                f"Opp HP: {bs.opponent_hp_percent*100:.1f}%"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 70 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw user-defined regions
        for i, region in enumerate(self.regions):
            x, y, rw, rh = region
            cv2.rectangle(frame, (x, y), (x+rw, y+rh), (0, 255, 255), 2)
            cv2.putText(frame, f"R{i}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _save_frame(self, frame: np.ndarray):
        """Save frame with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pokemon_frame_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
    
    def test_ocr_accuracy(self):
        """
        Capture multiple frames and test OCR consistency
        """
        print("Testing OCR accuracy over 10 frames...")
        results = []
        
        for i in range(10):
            output = self.perception.perceive()
            if output.battle_state:
                results.append({
                    'player': output.battle_state.player_pokemon,
                    'opponent': output.battle_state.opponent_pokemon,
                    'player_hp': output.battle_state.player_hp,
                    'moves': output.battle_state.available_moves
                })
            time.sleep(0.5)
        
        print("\nOCR Consistency Report:")
        if results:
            print(f"Player Pokemon: {[r['player'] for r in results]}")
            print(f"Opponent Pokemon: {[r['opponent'] for r in results]}")
            print(f"Player HP: {[r['player_hp'] for r in results]}")
        else:
            print("No battle states detected")
    
    def calibrate_regions(self):
        """
        Helper to find exact pixel coordinates for different UI elements
        """
        print("\nRegion Calibration Mode")
        print("This will help you find exact coordinates for UI elements")
        print("\n1. Battle - Player HP bar")
        print("2. Battle - Opponent HP bar")
        print("3. Battle - Move menu")
        print("4. Battle - Pokemon names")
        print("5. Overworld - Party menu")
        
        print("\nSwitch to melonDS and position it for a battle screen...")
        input("Press Enter when ready...")
        
        output = self.perception.perceive()
        frame = output.raw_frame
        
        # Save reference frame
        cv2.imwrite("reference_frame.png", frame)
        print(f"Saved reference frame: {frame.shape}")
        print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Let user click to define regions
        print("\nUse the interactive mode to draw regions")

    def benchmark_performance(self):
        """Test perception speed"""
        print("Benchmarking perception performance...")
        
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            output = self.perception.perceive()
        
        elapsed = time.time() - start_time
        fps = iterations / elapsed
        
        print(f"\nPerformance:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.2f}")
        print(f"  Per-frame: {elapsed/iterations*1000:.2f}ms")

def main():
    debugger = PerceptionDebugger()
    
    print("Pokemon Perception Debugger")
    print("=" * 50)
    print("Choose mode:")
    print("1. Interactive capture (recommended)")
    print("2. Test OCR accuracy")
    print("3. Calibrate regions")
    print("4. Benchmark performance")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        debugger.run_interactive_capture()
    elif choice == '2':
        debugger.test_ocr_accuracy()
    elif choice == '3':
        debugger.calibrate_regions()
    elif choice == '4':
        debugger.benchmark_performance()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()