import os
from methods.proposed import Proposed


def main():
    print('--------------------------------------------')
    print('Method 3: proposed method')
    method3 = Proposed()
    train, val = method3.train()
    test, _, _ = method3.test()
    print(' training metrics: \n {} \n validation metrics: \n {} \n test metrics: {}'.format(train, val, test))


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    main()