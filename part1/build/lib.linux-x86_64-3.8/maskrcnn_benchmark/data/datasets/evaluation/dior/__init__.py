import logging

from .dior_eval import do_dior_evaluation


def dior_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("dior evaluation doesn't support box_only, ignored.")
    logger.info("performing dior evaluation, ignored iou_types.")
    return do_dior_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
