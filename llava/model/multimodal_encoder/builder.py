import os
from .clip_encoder import CLIPVisionTower
from .patentmme_enocder import PatentMMETower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    return PatentMMETower(vision_tower, args=vision_tower_cfg, **kwargs)
