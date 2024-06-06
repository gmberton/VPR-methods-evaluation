
import numpy
import pickle

from pathlib import Path

def save_descriptors(args, test_ds, descriptors):
    if args.save_descriptors == "kapture":
        images_rel_paths = test_ds.get_images_rel_paths()
        with open(args.output_dir / "global_features.txt", "w") as file:
            file.write("# kapture format: 1.1\n")
            file.write("# Salad - See: https://github.com/serizba/salad\n") # Should point the the github of the used method
            file.write("# name, dtype, dsize, metric_type\n")
            file.write(f"{args.method}, {str(descriptors.dtype)}, {descriptors[0].shape[0]}, L2")
        
        for image_rel_path, image_descriptor in zip(images_rel_paths, descriptors):
            dir_name = args.output_dir / image_rel_path
            dir_name.parent.mkdir(parents=True, exist_ok=True)
            dir_name = dir_name.with_suffix(".jpg.gfeat")
            with open(dir_name, "wb") as file:
                image_descriptor.squeeze().tofile(file, sep="")
    else:
        with open(args.output_dir / "global_features.pickle", "wb") as file:
            pickle.dump(descriptors, file)

def save_pairsfile(args, test_ds, predictions, distances):
    pred_paths_dir = args.output_dir / "predictions_paths.txt"
    if args.rel_paths_in_pairsfile:
        rel_paths = test_ds.get_images_rel_paths()
        db_paths = rel_paths[:test_ds.num_database]
        q_paths = rel_paths[test_ds.num_database:]
    else:
        db_paths = test_ds.database_paths
        q_paths = test_ds.queries_paths
    rows = []
    for query_index, preds in enumerate(predictions):
        for pred_index, pred in enumerate(preds[:args.num_candidates_in_pairsfile]):
            row = [q_paths[query_index]]
            row.append(db_paths[pred])
            row.append(f"{distances[query_index, pred_index]:.4f}")
            rows.append(" ".join(row)+"\n")
    with open(pred_paths_dir, "w") as file:
        file.writelines(rows)