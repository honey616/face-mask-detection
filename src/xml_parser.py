import xml.etree.ElementTree as ET

LABEL_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes, labels = [], []

    for obj in root.findall("object"):
        labels.append(LABEL_MAP[obj.find("name").text])

        b = obj.find("bndbox")
        boxes.append([
            int(b.find("xmin").text),
            int(b.find("ymin").text),
            int(b.find("xmax").text),
            int(b.find("ymax").text)
        ])

    return boxes, labels
