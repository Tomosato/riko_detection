from effdet import create_model, create_dataset, create_loader
from effdet.data import resolve_input_config
from effdet.data.transforms import transforms_coco_eval
import torch.utils.data as data
import torch
from PIL import Image
from pathlib import Path
import cv2

class RikoDataset(data.Dataset):
    """
    Datasetの拡張
    """
    def __init__(self, image_paths, transform=None):
        super().__init__()

        self.image_paths = image_paths
        self._transform = transform

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        target = dict(img_idx=index, img_size=(img.width, img.height))
        if self.transform is not None:
            img, target = self.transform(img, target)
    
        return img, target

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


def riko_create_model():
    """
    Efficent Netのモデルを構築する
    """

    bench = create_model(
        "efficientdet_d1", # d0 ~ d7
        bench_task="predict",
        num_classes=20, # 人かそれ以外
    )

    return bench


if __name__ == "__main__":
    bench = riko_create_model()
    bench = bench.cuda()
    bench.eval()
    image_path = "test_riko.png"
    test_dataset = RikoDataset([Path(image_path)])
    input_config = resolve_input_config({}, bench.config)
    loader = create_loader(
            test_dataset,
            input_size=input_config['input_size'],
            batch_size=1,
            use_prefetcher=True,
            interpolation='bilinear',
            fill_color=input_config['fill_color'],
            mean=input_config['mean'],
            std=input_config['std'],
            num_workers=1,
            pin_mem=False)

    with torch.no_grad():
        for input_image, target in loader:
            output = bench(input_image, target)[0]
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i in range(output.size(0)):
                xmin, ymin, xmax, ymax, pred, label = output[i]
                #3,8, 11, 13,15,16
                if label.item() == 21:
                    cv2.rectangle(img, pt1=(int(xmin), int(ymin)),
                                    pt2=(int(xmax.item()), int(ymax.item())),
                                    color=(255, 0, 0), thickness=4)
                    tl = round(0.002 * max(img.shape[0:2])) + 1
                    tf = max(tl - 1, 1)
                    cv2.putText(img, str(label.item()), (int(xmin.item()), int(ymin.item()) - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite("test_result.jpg", img)

