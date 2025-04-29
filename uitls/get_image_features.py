import os
import numpy as np
from scene_graph_benchmark.wrappers import VinVLVisualBackbone

def save_image_features(image_folder, output_folder, detector, top_k = 30):
    jpg_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
    total_files = len(jpg_files)

    for idx, filename in enumerate(jpg_files, 1):
        img_path = os.path.join(image_folder, filename)
        if (idx % 1000 == 0): 
            print(f"[{idx}/{total_files} processed.")
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_folder, f"{base_name}.npz")
        if os.path.exists(out_path):
            continue
        try:
            dets = detector(img_path)
            v_feats = np.concatenate((dets['features'], dets['boxes']), axis=1)
            scores = np.array(dets["scores"])
            if len(scores) > top_k:
                top_indices = np.argsort(scores)[-top_k:][::-1]
            else:
                top_indices = np.argsort(scores)[::-1]

            v_feats = v_feats[top_indices]
            v_feats = v_feats.astype(np.float16)
            np.savez_compressed(os.path.join(output_folder, f"{base_name}.npz"), v_feats)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def __main():
    detector = VinVLVisualBackbone()

    for name in ["train2014", "val2014"]:
        image_folder = f"VQA_data/{name}"
        output_folder = f"VQA_data/{name}_image_feats"
        os.makedirs(output_folder, exist_ok=True)
        save_image_features(image_folder, output_folder, detector)

if __name__ == "__main__":
    __main()
    