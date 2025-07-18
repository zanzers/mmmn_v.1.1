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
                ela_cfg.get("compression"),
                ela_cfg.get("multiplier")
            )
    
    if noise_cfg := image_config.get("noise"):
        if noise_cfg.get("enabled"):
            image_analysis.noise(
                noise_cfg.get("local_mean"),
                noise_cfg.get("variance")
            )

    if copyMove_cfg := image_config.get("copyMove"):
        if copyMove_cfg.get("enabled"):
            image_analysis.copyMove(
                copyMove_cfg.get("block_size"),
                copyMove_cfg.get("step"),
                copyMove_cfg.get("threshold")
            )
    image_analysis.resampling()
    

def process_video(img: np.ndarray) -> dict:
    print("video")
    
