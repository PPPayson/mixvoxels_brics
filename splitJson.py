import json
import os
from argparse import ArgumentParser


def splitFiles(src):
    f = open(os.path.join(src, "transforms.json"), "r")

    data = json.load(f)

    #print(data)


    dataA = data


    train = []
    val = []
    test = []
    for i in range(53):
        if i!=0:
            train.append(data["frames"][i])
        if i==0:
            val.append(data["frames"][i])
            test.append(data["frames"][i])

    dataA["frames"] = train
    d = json.dumps(dataA)
    with open(os.path.join(src, "mixvoxels", "transforms_train.json"), "w") as out:
        out.write(d)

    dataA["frames"] = val
    d = json.dumps(dataA)
    with open(os.path.join(src, "mixvoxels", "transforms_val.json"), "w") as out:
        out.write(d)

    dataA["frames"] = test
    d = json.dumps(dataA)
    with open(os.path.join(src, "mixvoxels", "transforms_test.json"), "w") as out:
        out.write(d)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src", required=True, help="Source of BRICS dataset")
    args = parser.parse_args()
    splitFiles(args.src)
