from dlclive.dlclive import DLCLive
from dlclive.processor import Processor


model_path = "./app/models/DLC_run-analysis_resnet_50_iteration-0_shuffle-1"
dlc_live = DLCLive(
    model_path,
    model_type="base",
    # cropping=[1000, 1300, 400, 750],
    # dynamic=(True, 0.8, 50),
    resize=0.5,
    processor=Processor(),
)
