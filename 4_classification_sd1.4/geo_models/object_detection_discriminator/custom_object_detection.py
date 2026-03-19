from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Detect

import torch

class CustomDetectionModel(DetectionModel):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        last_model_layer = self.model[-1]

        if isinstance(last_model_layer, Detect):
            self.model = self.model[:-1]


# class CustomDetectionPredictor(BasePredictor):
# class CustomDetectionPredictor(BasePredictor):
#     """
#     A class extending the BasePredictor class for prediction based on a detection model.

#     Example:
#         ```python
#         from ultralytics.utils import ASSETS
#         from ultralytics.models.yolo.detect import DetectionPredictor

#         args = dict(model="yolo11n.pt", source=ASSETS)
#         predictor = DetectionPredictor(overrides=args)
#         predictor.predict_cli()
#         ```
#     """

#     def postprocess(self, preds, img, orig_imgs):
#         """Post-processes predictions and returns a list of Results objects."""
#         # breakpoint()
#         preds = preds[1]
#         assert len(preds) == 3

#         pred_1 = preds[0]
#         pred_2 = torch.tile(preds[1], (2, 2))
#         pred_3 = torch.tile(preds[2], (4, 4))
#         pred = torch.cat([pred_1, pred_2, pred_3], dim=1)
#         return pred



class CustomYOLO(YOLO):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": CustomDetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                # "predictor": CustomDetectionPredictor,
            },
        }

    def __call__(self, x):
        result = self.model(x)
        preds = result[1]
        assert len(preds) == 3

        pred_1 = preds[0]
        pred_2 = torch.tile(preds[1], (2, 2))
        pred_3 = torch.tile(preds[2], (4, 4))
        pred = torch.cat([pred_1, pred_2, pred_3], dim=1)
        return pred


