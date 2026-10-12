"""Connect your installed, frozen encoders here. No weights are downloaded."""
import numpy as np

class LocalFrozenEncoders:
    def __init__(self,countse_checkpoint,scene_encoder_checkpoint,**kwargs):
        # Supply local CountSE and visual-text wrappers using the installed
        # implementations and your audited, versioned checkpoint files.
        # Set every module to eval() and every parameter requires_grad_(False).
        self.provenance={"counting_model":"CountSE","counting_checkpoint":countse_checkpoint,
                         "scene_checkpoint":scene_encoder_checkpoint,
                         "feature_dimensions":"set from actual encoder outputs",
                         "preprocessing":"record the actual frozen preprocessing"}

    def counting(self,image,category):
        raise NotImplementedError("Return CountSE's text-guided evidence vector from local frozen weights")

    def scene(self,target_removed_image):
        raise NotImplementedError("Return a frozen visual scene embedding")

    def category(self,category):
        raise NotImplementedError("Return the frozen text embedding of the target category")

    def joint(self,image,category):
        raise NotImplementedError("Return frozen image-text features for the supervised MLP control")

class LocalInpainter:
    def __init__(self,checkpoint,**kwargs):
        self.checkpoint=checkpoint
        # Load the actual local inpainting model in evaluation mode here.

    def __call__(self,image,mask,seed=0):
        raise NotImplementedError("Return a PIL RGB image jointly inpainted under the supplied binary mask")
