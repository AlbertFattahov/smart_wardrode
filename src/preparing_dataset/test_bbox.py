
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random, sys

#  Настройки 
IMAGES_DIR = Path('output_yolo/images')   
LABELS_DIR = Path('output_yolo/labels')   
SAMPLE_N = 12                             # сколько примеров показать
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

 
def read_label_file(lbl_path):
    """Читает файл и возвращает список списков строковых токенов."""
    try:
        text = lbl_path.read_text(encoding='utf-8')
    except Exception:
        try:
            text = lbl_path.read_text(encoding='cp1251', errors='ignore')
        except Exception:
            return []
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        lines.append(parts)
    return lines

def detect_label_format(parts):
    """
    Определяет формат по одной строке:
    - YOLO norm: len>=5, parts[1..4] в диапазоне [0,1] (или близко)
    - PIXELS xyxy: len>=4, parts числа, часто >1 и <=max(img_dim)
    Возвращает 'yolo', 'xyxy', или None.
    """
    try:
        nums = list(map(float, parts[1:5]))
    except Exception:
        return None
    # если все в [0,1] -> yolo
    if all(-0.01 <= v <= 1.01 for v in nums):
        return 'yolo'
    # если хотя бы один >1 -> возможно пиксели (xyxy or xc yc w h in px)
    if any(v > 1 for v in nums):
        # если parts length == 4 (без класса) возможно pure xyxy; if len>=5 and first token class -> still pixels
        return 'pixels'
    return None

def parse_label_line(parts, img_w, img_h, fmt_hint=None):
    """
    Возвращает (cls, xc, yc, w_norm, h_norm) или None при ошибке.
    Поддерживает:
      - YOLO normalized: cls xc yc w h
      - pixels xyxy: cls x1 y1 x2 y2  (или без cls: x1 y1 x2 y2)
      - pixels xc yc w h (если явно >1)
    """
    
    if len(parts) >= 5:
        cls = parts[0]
        vals = list(map(float, parts[1:5]))
    elif len(parts) == 4:
        cls = '0'
        vals = list(map(float, parts[0:4]))
    else:
        return None

    # detect
    fmt = fmt_hint or detect_label_format(parts)
    if fmt == 'yolo':
        xc, yc, ww, hh = vals
        return cls, float(xc), float(yc), float(ww), float(hh)
    else:
        # pixels:  x1 y1 x2 y2 or xc yc w h in px
        a,b,c,d = vals
        #  if c > a and d > b and c <= img_w and d <= img_h -> treat as x1,y1,x2,y2
        if c > a and d > b and c <= img_w + 1 and d <= img_h + 1:
            x1,y1,x2,y2 = a,b,c,d
            
            x1 = max(0, min(img_w, x1)); x2 = max(0, min(img_w, x2))
            y1 = max(0, min(img_h, y1)); y2 = max(0, min(img_h, y2))
            if x2 <= x1 or y2 <= y1:
                return None
            xc = (x1 + x2) / 2.0 / img_w
            yc = (y1 + y2) / 2.0 / img_h
            ww = (x2 - x1) / img_w
            hh = (y2 - y1) / img_h
            return cls, xc, yc, ww, hh
        else:
            #  px center + w,h
            xc_px, yc_px, ww_px, hh_px = a,b,c,d
            xc = xc_px / img_w
            yc = yc_px / img_h
            ww = ww_px / img_w
            hh = hh_px / img_h
            return cls, xc, yc, ww, hh

def draw_boxes_on_image(img_path, labels, show=True):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    drawn = 0
    for cls, xc, yc, ww, hh in labels:
        # конвертация в нормализованные коордитаны 
        x1 = max(0, (xc - ww/2) * w)
        y1 = max(0, (yc - hh/2) * h)
        x2 = min(w, (xc + ww/2) * w)
        y2 = min(h, (yc + hh/2) * h)
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1,y1,x2,y2], outline='red', width=2)
        draw.text((x1+3, y1+3), str(cls), fill='yellow', font=font)
        drawn += 1
    return img, drawn

#  Основная логика 
def find_label_for_image(img_path):
    
    candidate = LABELS_DIR / (img_path.stem + '.txt')
    if candidate.exists():
        return candidate
    # рекурсивный поиск
    found = list(LABELS_DIR.rglob(img_path.stem + '.txt'))
    return found[0] if found else None

# def main():
#     imgs = [p for p in IMAGES_DIR.rglob('*') if p.suffix.lower() in IMG_EXTS]

#     if not imgs:
#         print("No images found in", IMAGES_DIR.resolve())
#         return
#     random.shuffle(imgs)
#     selected = imgs[:SAMPLE_N]
#     rows = (len(selected) + 2) // 3
#     plt.figure(figsize=(15, 5*rows))
#     idx = 1
#     total_drawn = 0
#     for img_path in selected:
#         lbl_path = find_label_for_image(img_path)
#         if not lbl_path:
#             print(f"[NO LABEL] {img_path.name}")
#             continue
#         parts_list = read_label_file(lbl_path)
#         if not parts_list:
#             print(f"[EMPTY LABEL] {lbl_path}")
#             continue
#         img = Image.open(img_path)
#         w,h = img.size
#         parsed = []
#         problems = []
#         for parts in parts_list:
#             fmt_hint = detect_label_format(parts)
#             parsed_line = parse_label_line(parts, w, h, fmt_hint)
#             if parsed_line is None:
#                 problems.append(parts)
#             else:
#                 parsed.append(parsed_line)
#         if problems:
#             print(f"[PARSE ISSUES] {lbl_path} lines: {len(problems)} problematic")
#         img_with_boxes, drawn = draw_boxes_on_image(img_path, parsed, show=False)
#         total_drawn += drawn
#         plt.subplot(rows, 3, idx)
#         plt.imshow(img_with_boxes)
#         plt.title(f"{img_path.name}\nlabels={len(parts_list)} drawn={drawn}")
#         plt.axis('off')
#         idx += 1
#     plt.tight_layout()
#     plt.show()
#     print("Total boxes drawn in sample:", total_drawn)

def main():
    # Если нужно выбирать только одно изображение, иначе None
    IMAGE_NAME_FILTER = "trunks"   # или None

    # Фильтрация боксов по ID
    FILTER_IDS = {35}             # пример: {0, 2}


    # сбор изображений (опционально фильтруем по имени файла) 
    imgs = []
    for p in IMAGES_DIR.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if IMAGE_NAME_FILTER:
            if IMAGE_NAME_FILTER.lower() not in p.name.lower():
                continue
        imgs.append(p)

    if not imgs:
        print("No images found in", IMAGES_DIR.resolve())
        return

    random.shuffle(imgs)
    selected = imgs[:SAMPLE_N]
    rows = (len(selected) + 2) // 3
    plt.figure(figsize=(15, 5*rows))
    idx = 1
    total_drawn = 0

    for img_path in selected:
        lbl_path = find_label_for_image(img_path)
        if not lbl_path:
            print(f"[NO LABEL] {img_path.name}")
            continue

        parts_list = read_label_file(lbl_path)
        if not parts_list:
            print(f"[EMPTY LABEL] {lbl_path}")
            continue

        img = Image.open(img_path)
        w,h = img.size
        parsed = []
        problems = []
        for parts in parts_list:
            fmt_hint = detect_label_format(parts)
            parsed_line = parse_label_line(parts, w, h, fmt_hint)
            if parsed_line is None:
                problems.append(parts)
            else:
                parsed.append(parsed_line)

        if problems:
            print(f"[PARSE ISSUES] {lbl_path} lines: {len(problems)} problematic")

        # ФИЛЬТРАЦИЯ БОКСОВ ПО ID
        def is_trunks_label(item):
            cls_token = str(item[0]).strip()
            # если класс — число 
            if cls_token.isdigit():
                idx_cls = int(cls_token)
                if FILTER_IDS:
                    return idx_cls in FILTER_IDS

                # если нет фильтров — по умолчанию не фильтруем по ID
                return True
            else:

                # если указаны числовые фильтры, строковый токен не подходит
                if FILTER_IDS:
                    return False
                return True

        # применяем фильтр: если ни один фильтр не задан — parsed_trunks == parsed
        if FILTER_IDS:
            parsed_trunks = [it for it in parsed if is_trunks_label(it)]
        else:
            parsed_trunks = parsed

        if not parsed_trunks:
            print(f"[NO MATCHING BOXES] {img_path.name} (labels found: {len(parsed)})")
            continue

        img_with_boxes, drawn = draw_boxes_on_image(img_path, parsed_trunks, show=False)
        total_drawn += drawn

        plt.subplot(rows, 3, idx)
        plt.imshow(img_with_boxes)
        plt.title(f"{img_path.name}\nlabels={len(parts_list)} drawn={drawn}")
        plt.axis('off')
        idx += 1

    plt.tight_layout()
    plt.show()
    print("Total boxes drawn in sample:", total_drawn)

if __name__ == '__main__':
    main()
