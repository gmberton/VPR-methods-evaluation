import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale
from PIL import Image, ImageDraw, ImageFont


# Height and width of a single image
H = 512
W = 512
TEXT_H = 175
FONTSIZE = 50
SPACE = 50  # Space between two images


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with text"""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
    img = Image.new("RGB", ((W * len(labels)) + 50 * (len(labels) - 1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        _, _, w, h = d.textbbox((0, 0), text, font=font)
        d.text(((W + SPACE) * i + W // 2 - w // 2, 1), text, fill=(0, 0, 0), font=font)
    return np.array(img)[:100]  # Remove some empty space


def draw(img, c=(0, 255, 0), thickness=20):
    """Draw a colored (usually red or green) box around an image."""
    p = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    for i in range(3):
        cv2.line(img, (p[i, 0], p[i, 1]), (p[i + 1, 0], p[i + 1, 1]), c, thickness=thickness * 2)
    return cv2.line(img, (p[3, 0], p[3, 1]), (p[0, 0], p[0, 1]), c, thickness=thickness * 2)


def build_prediction_image(images_paths, preds_correct):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    """
    assert len(images_paths) == len(preds_correct)
    labels = ["Query"]
    for i, is_correct in enumerate(preds_correct[1:]):
        if is_correct is None:
            labels.append(f"Pred{i}")
        else:
            labels.append(f"Pred{i} - {is_correct}")

    num_images = len(images_paths)
    images = [np.array(Image.open(path).convert("RGB")) for path in images_paths]
    for img, correct in zip(images, preds_correct):
        if correct is None:
            continue
        color = (0, 255, 0) if correct else (255, 0, 0)
        draw(img, color)
    concat_image = np.ones([H, (num_images * W) + ((num_images - 1) * SPACE), 3])
    rescaleds = [
        rescale(i, [min(H / i.shape[0], W / i.shape[1]), min(H / i.shape[0], W / i.shape[1]), 1]) for i in images
    ]
    for i, image in enumerate(rescaleds):
        pad_width = (W - image.shape[1] + 1) // 2
        pad_height = (H - image.shape[0] + 1) // 2
        image = np.pad(image, [[pad_height, pad_height], [pad_width, pad_width], [0, 0]], constant_values=1)[:H, :W]
        concat_image[:, i * (W + SPACE) : i * (W + SPACE) + W] = image
    try:
        labels_image = write_labels_to_image(labels)
        final_image = np.concatenate([labels_image, concat_image])
    except OSError:  # Handle error in case of missing PIL ImageFont
        final_image = concat_image
    final_image = Image.fromarray((final_image * 255).astype(np.uint8))
    return final_image


def save_file_with_paths(query_path, preds_paths, positives_paths, output_path, use_labels=True):
    file_content = []
    file_content.append("Query path:")
    file_content.append(query_path + "\n")
    file_content.append("Predictions paths:")
    file_content.append("\n".join(preds_paths) + "\n")
    if use_labels:
        file_content.append("Positives paths:")
        file_content.append("\n".join(positives_paths) + "\n")
    with open(output_path, "w") as file:
        _ = file.write("\n".join(file_content))


def save_preds(predictions, eval_ds, log_dir, save_only_wrong_preds=None, use_labels=True):
    """For each query, save an image containing the query and its predictions,
    and a file with the paths of the query, its predictions and its positives.

    Parameters
    ----------
    predictions : np.array of shape [num_queries x num_preds_to_viz], with the preds
        for each query
    eval_ds : TestDataset
    log_dir : Path with the path to save the predictions
    save_only_wrong_preds : bool, if True save only the wrongly predicted queries,
        i.e. the ones where the first pred is uncorrect (further than 25 m)
    """
    if use_labels:
        positives_per_query = eval_ds.get_positives()

    viz_dir = log_dir / "preds"
    viz_dir.mkdir()
    for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving preds in {viz_dir}")):
        query_path = eval_ds.queries_paths[query_index]
        list_of_images_paths = [query_path]
        # List of None (query), True (correct preds) or False (wrong preds)
        preds_correct = [None]
        for pred_index, pred in enumerate(preds):
            pred_path = eval_ds.database_paths[pred]
            list_of_images_paths.append(pred_path)
            if use_labels:
                is_correct = pred in positives_per_query[query_index]
            else:
                is_correct = None
            preds_correct.append(is_correct)

        if save_only_wrong_preds and preds_correct[1]:
            continue

        prediction_image = build_prediction_image(list_of_images_paths, preds_correct)
        pred_image_path = viz_dir / f"{query_index:03d}.jpg"
        prediction_image.save(pred_image_path)

        if use_labels:
            positives_paths = [eval_ds.database_paths[idx] for idx in positives_per_query[query_index]]
        else:
            positives_paths = None
        save_file_with_paths(
            query_path=list_of_images_paths[0],
            preds_paths=list_of_images_paths[1:],
            positives_paths=positives_paths,
            output_path=viz_dir / f"{query_index:03d}.txt",
            use_labels=use_labels,
        )
