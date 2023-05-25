import os
import queue
import subprocess
import time


if __name__ == "__main__":
    all_models = [
        "DenseNet121", "DenseNet169", "DPN26", "DPN92",
        "EfficientNetB0", "GoogLeNet", "MobileNet", "MobileNetV2",
        "PNASNetA", "PNASNetB", "RegNetX_200MF", "RegNetX_400MF",
        "RegNetY_400MF", "ResNet18", "ResNet34", "ResNet50",
        "ResNet101", "ResNeXt29_2x64d", "ResNeXt29_4x64d", "ResNeXt29_8x64d",
        "ResNeXt29_32x4d", "SENet18", "ShuffleNetG2", "ShuffleNetG3",
        "ShuffleNetV2_0_5", "ShuffleNetV2_1", "ShuffleNetV2_1_5", "ShuffleNetV2_2",
        "VGG11", "VGG13", "VGG16", "VGG19",
    ]

    processes = []
    seed = 42

    # Assume we have 4 GPUs available, identified by IDs 0, 1, 2, and 3.
    gpu_queue = queue.Queue()
    for i in range(4):
        gpu_queue.put(i)
    
    # Queue to hold scripts
    script_arg_queue = queue.Queue()
    for model in all_models:
        script_arg = ("main.py", "--lr", "0.01", "--model-name", model, "--seed", str(seed))
        script_arg_queue.put(script_arg)

    while not script_arg_queue.empty():
        if gpu_queue.empty():
            # No GPU is available, wait for a process to finish
            for process in processes:
                if process.poll() is not None:  # A None value indicates that the process is still running
                    processes.remove(process)
                    gpu_queue.put(process.gpu_id)
                    break
            else:
                # No process has finished, wait a bit before checking again
                time.sleep(10)
                continue

        gpu_id = gpu_queue.get()
        script_arg = script_arg_queue.get()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        process = subprocess.Popen(["python"] + list(script_arg), env=env)
        process.gpu_id = gpu_id
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()
