from effdet import create_model, create_loader
from effdet.data import resolve_input_config
import torch.utils.data as data
import torch
from PIL import Image


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
        "efficientdet_d1",  # d0 ~ d7
        bench_task="predict",
        num_classes=20  # 人かそれ以外
    )

    return bench


def get_bounding_box(image_paths):
    """
    受け取ったパスの画像から、人っぽいもののバウンディングボックスの
    座標を返す

    image_paths: 対象の画像のパスのリスト

    return
        切り出し候補の座標と画像のパス
    """

    bench = riko_create_model()
    bench = bench.cuda()
    bench.eval()

    test_dataset = RikoDataset(image_paths)
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

    result = {}

    with torch.no_grad():
        for input_image, target in loader:
            target_image_path_index = target["img_idx"].item()
            image_path = image_paths[target_image_path_index]
            output = bench(input_image, target)[0]
            for i in range(output.size(0)):
                xmin, ymin, xmax, ymax, pred, label = output[i]
                #3,8, 11, 13,15,16がよさそう
                if label.item() == 8:
                    if image_path in result:
                        control_list = result[image_path]
                        control_list.append([xmin.item(),
                                            ymin.item(),
                                            xmax.item(),
                                            ymax.item()])
                        result[image_path] = control_list
                    else:
                        result[image_path] = [[xmin.item(),
                                                ymin.item(),
                                                xmax.item(),
                                                ymax.item()]]

    return result
