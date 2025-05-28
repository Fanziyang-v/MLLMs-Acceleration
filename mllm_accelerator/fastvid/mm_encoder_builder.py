from .siglip_encoder import SigLipVisionTowerAbstract


def build_vision_abstract(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    return SigLipVisionTowerAbstract(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
