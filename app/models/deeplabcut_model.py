from dlclive.dlclive import DLCLive
from dlclive.processor import Processor

model_path = "./models/DLC_runaly_resnet_50_iteration-0_shuffle-1"
dlc_live = DLCLive(
    model_path,
    model_type="base",
    processor=Processor(),
)
