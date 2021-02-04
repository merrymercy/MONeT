import os

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


if __name__ == "__main__":
    batch_sizes = list(range(32, 256, 16)) + list(range(256, 1024, 32))

    for network in ['resnet50']:
        for alg in ['5', '6', '7', '8', '9', '10']:
            for batch_size in batch_sizes:
                ret_code = run_cmd(
                    f"python3 imagenet.py ~/imagenet -a {network} --gpu 0 --b {batch_size} "
                    f"--sol ../data/monet_r50_184_24hr/"
                    f"solution_resnet50_184_inplace_conv_multiway_newnode_{alg}.00.pkl")
                if ret_code != 0:
                    break

