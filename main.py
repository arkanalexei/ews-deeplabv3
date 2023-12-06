import os
from methods.sadeghi_et_al_2017 import SadeghiEtAl2017
from methods.rico_fernandez_et_al_2019 import RicoFernandezEtAl2019
from methods.yu_et_al_2017 import YuEtAl2017
from methods.proposed import Proposed
from methods.dlv3plus import DLv3Plus

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def main():
    print('--------------------------------------------')
    print('Method 3: proposed method')
    method3 = Proposed()
    train, val = method3.train()
    test, _, _ = method3.test()
    print(' training metrics: \n {} \n validation metrics: \n {} \n test metrics: {}'.format(train, val, test))
    
    # print('--------------------------------------------')
    # print('Method 5: DeepLab v3+ trained from scratch')
    # method5 = DLv3Plus(pretrained=False)
    # train, val = method5.train()
    # test, _, _ = method5.test()
    # print(' training metrics: \n {} \n validation metrics: \n {} \n test metrics: {}'.format(train, val, test))

    # print('--------------------------------------------')
    # print('Method 6: DeepLab v3+ pretrained on imagenet')
    # method6 = DLv3Plus(pretrained=True)
    # train, val = method6.train()
    # test, _, _ = method6.test()
    # print(' training metrics: \n {} \n validation metrics: \n {} \n test metrics: {}'.format(train, val, test))

if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    main()