import json
import shutil
from pathlib import Path
from tqdm import tqdm

def merge_coco_jsons_train_test(input_dir, output_dir, only_train=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    train_input_dir = input_dir / "Treningowy"
    test_input_dir = input_dir / "Testowy"

    # Tworzymy foldery tylko dla treningowego, jeśli only_train
    train_images_dir = output_dir / "train"
    train_images_dir.mkdir(parents=True, exist_ok=True)

    # Tworzymy folder testowy tylko jeśli nie ma flagi only_train
    if not only_train:
        test_images_dir = output_dir / "test"
        test_images_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    train = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    test = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    train_image_id_offset = 0
    train_annotation_id_offset = 0
    test_image_id_offset = 0
    test_annotation_id_offset = 0

    def process_task(task_path, merged_json, images_dir, image_id_offset, annotation_id_offset):
        json_files = list(task_path.rglob("annotations/*.json"))
        if not json_files:
            print(f"⚠️ Brak pliku JSON w {task_path}")
            return image_id_offset, annotation_id_offset
        json_file = json_files[0]

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not merged_json["categories"]:
            merged_json["categories"] = data["categories"]
        else:
            if data["categories"] != merged_json["categories"]:
                raise ValueError(f"❌ Różne kategorie w pliku: {json_file}")

        images = data["images"]
        annotations = data["annotations"]

        image_folders = [f for f in task_path.rglob("images/*") if f.is_dir()]
        if not image_folders:
            print(f"⚠️ Brak folderu z obrazami w {task_path}")
            return image_id_offset, annotation_id_offset

        if len(image_folders) > 1:
            print(f"ℹ️ Wykryto wiele folderów z obrazami: {[f.name for f in image_folders]} — używam pierwszego.")

        images_folder = image_folders[0]

        old_to_new_ids = {}

        for img in images:
            orig_name = img["file_name"]
            src = images_folder / orig_name
            if not src.exists():
                print(f"⚠️ Nie znaleziono obrazu: {src}")
                continue

            new_name = f"{task_path.name}__{orig_name}"
            dst = images_dir / new_name
            shutil.copy(src, dst)

            new_img_id = image_id_offset + img["id"]
            old_to_new_ids[img["id"]] = new_img_id

            new_img = img.copy()
            new_img["id"] = new_img_id
            new_img["file_name"] = new_name
            merged_json["images"].append(new_img)

        for ann in annotations:
            if ann["image_id"] in old_to_new_ids:
                new_ann = ann.copy()
                new_ann["id"] = annotation_id_offset + ann["id"]
                new_ann["image_id"] = old_to_new_ids[ann["image_id"]]
                merged_json["annotations"].append(new_ann)

        if old_to_new_ids:
            image_id_offset = max(old_to_new_ids.values()) + 1
        if merged_json["annotations"]:
            annotation_id_offset = max(a["id"] for a in merged_json["annotations"]) + 1

        return image_id_offset, annotation_id_offset

    # ✅ TEST (jeśli nie tylko treningowy)
    if not only_train:
        test_tasks = []
        for d in test_input_dir.iterdir():
            if not d.is_dir():
                continue
            inner = d / d.name
            if inner.exists() and inner.is_dir():
                test_tasks.append(inner)
            else:
                test_tasks.append(d)

        for test_task in tqdm(test_tasks, desc="Przetwarzanie testowych tasków"):
            test_image_id_offset, test_annotation_id_offset = process_task(
                test_task, test, test_images_dir, test_image_id_offset, test_annotation_id_offset)

        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump(test, f, indent=2, ensure_ascii=False)
        print(f"✅ Test JSON: {output_dir / 'test.json'}")
        print(f"✅ Obrazy test w: {test_images_dir}")

    # ✅ TRAIN
    train_tasks = []
    for d in train_input_dir.iterdir():
        if not d.is_dir():
            continue
        inner = d / d.name
        if inner.exists() and inner.is_dir():
            train_tasks.append(inner)
        else:
            train_tasks.append(d)

    for train_task in tqdm(train_tasks, desc="Przetwarzanie treningowych tasków"):
        train_image_id_offset, train_annotation_id_offset = process_task(
            train_task, train, train_images_dir, train_image_id_offset, train_annotation_id_offset)

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dane zapisane do folderu: {output_dir}")
    print(f"✅ Train JSON: {output_dir / 'train.json'}")
    print(f"✅ Obrazy train w: {train_images_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scal COCO JSON-y z osobnych folderów train/test")
    parser.add_argument("input_dir", help="Folder wejściowy")
    parser.add_argument("output_dir", help="Folder wyjściowy")
    parser.add_argument("--only-train", action="store_true", help="Tylko przetwarzanie treningowe (pomija testowe)", default=True)

    args = parser.parse_args()
    merge_coco_jsons_train_test(args.input_dir, args.output_dir, only_train=args.only_train)
