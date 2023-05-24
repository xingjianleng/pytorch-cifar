import subprocess


if __name__ == "__main__":
    all_models = [
        ("DenseNet121", "DenseNet169", "DPN26", "DPN92"),
        ("EfficientNetB0", "GoogLeNet", "MobileNet", "MobileNetV2"),
        ("PNASNetA", "PNASNetB", "RegNetX_200MF", "RegNetX_400MF"),
        ("RegNetY_400MF", "ResNet18", "ResNet34", "ResNet50"),
        ("ResNet101", "ResNeXt29_2x64d", "ResNeXt29_4x64d", "ResNeXt29_8x64d"),
        ("ResNeXt29_32x4d", "SENet18", "ShuffleNetG2", "ShuffleNetG3"),
        ("ShuffleNetV2_0_5", "ShuffleNetV2_1", "ShuffleNetV2_1_5", "ShuffleNetV2_2"),
        ("VGG11", "VGG13", "VGG16", "VGG19"),
    ]

    seed = 42

    for model_batch in all_models:
        commands = [
            f"CUDA_VISIBLE_DEVICES=0 python main.py --lr=0.01 --model-name {model_batch[0]} --seed {seed}",
            f"CUDA_VISIBLE_DEVICES=1 python main.py --lr=0.01 --model-name {model_batch[1]} --seed {seed}",
            f"CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.01 --model-name {model_batch[2]} --seed {seed}",
            f"CUDA_VISIBLE_DEVICES=3 python main.py --lr=0.01 --model-name {model_batch[3]} --seed {seed}",
        ]
        processes = []
        for command in commands:
            process = subprocess.subprocess.Popen(command, shell=True)
            processes.append(process)

        for process in processes:
            process.wait()
