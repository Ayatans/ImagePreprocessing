import logging

from .nwpuv2_eval import do_nwpuv2_evaluation

def nwpuv2_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("nwpuv2 evaluation doesn't support box_only, ignored.")
    logger.info("performing nwpuv2 evaluation, ignored iou_types.")
    return do_nwpuv2_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )


