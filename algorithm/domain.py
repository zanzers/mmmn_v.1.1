from algorithm.core.spatial_domain import *
from algorithm.tools.functions import *
import yaml


# Entry Point
def mmmn_domain(user_input: np.ndarray):

    CONFIG = g_loadConfig()

    file_type = g_checkInput(user_input)
    match file_type:
        case True:
            return process_img(user_input, CONFIG)
        case False:
            return process_video(user_input, CONFIG)
        case _:
            raise ValueError(f"[ERROR] Unsupported file format!")




def process_img(img: np.ndarray, CONFIG) -> dict:

    original = cv2.imread(img)
    g_saveImage([
        ('original.jpg', original)
    ], remove_after=True)

    image_analysis = Spatial_Domain(original)
    image_config = CONFIG.get("image", {})

    if ela_cfg := image_config.get("ela"):
        if ela_cfg.get("enabled"):
            image_analysis.ela(
                ela_cfg.get("compression", 85),
                ela_cfg.get("multiplier", 40)
            )
    

def process_video(img: np.ndarray) -> dict:
    print("video")
    
