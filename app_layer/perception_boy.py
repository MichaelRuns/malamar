from PIL import Image
import cv2
import numpy as np
class PerceptionModule:
    """
    Multi-layer perception system for Pokemon Blue
    Layer 1: Fast pixel-based heuristics (no ML)
    Layer 2: OCR for text/numbers
    Layer 3: Vision LLM for complex scenes
    """
    
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.previous_frames = []
        self.frame_similarity_threshold = 0.95

    # Cells are 32x32px
    # There are 9 rows and 10 cols
    #
    def perceive(self, frame: Image):
        arr = np.array(frame.convert("L"))
        cv2.imshow('Perception view',arr)
        self.check_all_cells(frame)

    
    def check_all_cells(self, frame: Image):
        arr = np.array(frame.convert("L"))
        tile_width = 8
        tile_height = 8
        print(np.shape(arr))
        for row in range(18):
            for col in range(20):
                tile = arr[row*tile_width:(row+1)*tile_height,col*16:(col+1)*tile_width]
                cv2.imshow(f"tile({row},{col})",tile)
                cv2.waitKey(0)




