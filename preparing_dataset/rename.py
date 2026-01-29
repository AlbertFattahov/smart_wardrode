import shutil
from pathlib import Path

OUT = Path("output_yolo") 
IMG_ROOT = OUT / "images"
LBL_ROOT = OUT / "labels"

# Настройки
DRY_RUN = False     # True — только показать изменения; False — выполнить
COPY_INSTEAD = False  # True — копировать файлы в новые имена; False — переименовывать (move)
REPLACE_FROM = "'"
REPLACE_TO = ""

def make_target_name(path: Path) -> Path:
    """Создаёт целевое имя, заменяя REPLACE_FROM на REPLACE_TO, сохраняя расширение."""
    new_name = path.name.replace(REPLACE_FROM, REPLACE_TO)
    return path.with_name(new_name)

def resolve_collision(target: Path) -> Path:
    """Если target существует, добавляет суффикс _1, _2 ... чтобы избежать перезаписи."""
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def process_pair(old_img: Path, new_img: Path):
    """Переименовать/скопировать изображение и соответствующий label."""
    # определяем соответствующие пути меток по stem
    old_lbl = LBL_ROOT / old_img.parent.name / (old_img.stem + ".txt")
    new_lbl = LBL_ROOT / new_img.parent.name / (new_img.stem + ".txt")

    # разрешаем коллизии для изображений и меток отдельно
    final_img = resolve_collision(new_img)
    final_lbl = resolve_collision(new_lbl)

    if DRY_RUN:
        print(f"[DRY] {old_img} -> {final_img}")
        if old_lbl.exists():
            print(f"[DRY] {old_lbl} -> {final_lbl}")
        else:
            print(f"[DRY] label not found for {old_img.name}")
        return

    # выполняем операцию: копировать или переименовать
    if COPY_INSTEAD:
        shutil.copy2(old_img, final_img)
    else:
        old_img.rename(final_img)

    # метка
    if old_lbl.exists():
        if COPY_INSTEAD:
            shutil.copy2(old_lbl, final_lbl)
        else:
            old_lbl.rename(final_lbl)
    else:
        # если метки нет — ничего не делаем
        pass

def main():
    if not OUT.exists():
        print("Папка OUT не найдена:", OUT)
        return

    mappings = []
    for part in ("train", "val"):
        img_dir = IMG_ROOT / part
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if not img_path.is_file():
                continue
            if REPLACE_FROM in img_path.name:
                new_img = make_target_name(img_path)
                mappings.append((img_path, new_img))

    if not mappings:
        print("Нет файлов с символом", REPLACE_FROM)
        return

    print("Найдено файлов для обработки:", len(mappings))
    for old_img, new_img in mappings:
        process_pair(old_img, new_img)

    if DRY_RUN:
        print("Dry run завершён. Установи DRY_RUN = False чтобы выполнить изменения.")
    else:
        print("Переименование/копирование завершено.")

if __name__ == "__main__":
    main()