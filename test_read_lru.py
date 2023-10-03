import os

path = r'C:\Users\PlasticDiscobolus\work\alif_sg\good_experiments'

outs = [d for d in os.listdir(path) if 'out' in d]


# out  = outs[0]

for out in outs:
    try:
        out_path = os.path.join(path, out)
        # load
        with open(out_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        conf_results  = [line for line in lines if ' =>> Initializing...' in line][-1]

        test_results  = [line for line in lines if 'Test Accuracy' in line][-1]

        print('-'*20)
        print(out)
        print(conf_results)
        print(test_results)
    except Exception as e:
        print(e)