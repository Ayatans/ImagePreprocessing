import logging

from .nwpu_eval import do_nwpu_evaluation


def nwpu_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("dior evaluation doesn't support box_only, ignored.")
    logger.info("performing dior evaluation, ignored iou_types.")
    return do_nwpu_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
