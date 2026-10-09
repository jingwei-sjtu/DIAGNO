import os, sys
import types
from functools import partial

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.DiagNO import DiagNO

def get_baseline_models(img_size=(128, 256), in_chans=3, out_chans=3, res_pred=True):
    
    # prepare dicts containing models and corresponding metrics
    model_registry = dict(
        diagno_e128 = partial(
            DiagNO,
            img_size=img_size, 
            scale_factor=2, 
            in_chans=in_chans, out_chans=out_chans, embed_dim=128, num_heads=8, 
            mlp_ratio=2.0, res_pred=res_pred, dpr=0.1, bilinear=True
        ),

        diagno_e64 = partial(
            DiagNO,
            img_size=img_size, 
            scale_factor=2, 
            in_chans=in_chans, out_chans=out_chans, embed_dim=64, num_heads=8, 
            mlp_ratio=4.0, res_pred=res_pred, dpr=0.1, bilinear=True
        ),

        diagno_e256 = partial(
            DiagNO,
            img_size=img_size, 
            scale_factor=2, 
            in_chans=in_chans, out_chans=out_chans, embed_dim=256, num_heads=4, 
            mlp_ratio=2.0, res_pred=res_pred, dpr=0.1, bilinear=True
        ),
    )

    return model_registry
