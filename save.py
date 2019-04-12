# -*- coding: utf-8 -*-
# @Time    : 2/28/19 10:14 AM
# @Author  : zhoujun
import os
import pathlib
import shutil
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool

def chunkIt(seq, num):
    """
    将list进行等分
    :param seq: list
    :param num: num
    :return:
    """
    len_seq = len(seq)
    avg = len_seq / float(num)
    out = []
    last = 0.0

    while last < len_seq:
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return [x for x in out if len(x)]


# def copy_list(output_path,txt_path,file_list):
#     with open(txt_path,mode='w',encoding='utf8') as fw:
#         for line in file_list:
#             line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
#             src_path = pathlib.Path(line[0])
#             if not src_path.exists():
#                 continue
#             new_path = str(pathlib.Path(output_path) / src_path.parents._parts[-2])
#             if not os.path.exists(new_path):
#                 os.makedirs(new_path)
#             le = len(os.listdir(new_path))
#             new_path = os.path.join(new_path, str(le) + src_path.suffix)
#             shutil.copy(str(src_path),new_path)
#             fw.write(new_path + '\t' + line[1] + '\n')


# def copy_all(output_path,txt_path):
#     num = mp.cpu_count()
#     with open(txt_path,mode='r',encoding='utf8') as fr:
#         lists = chunkIt(fr.readlines(),num)
#     pool = Pool(processes=num)
#     pbar = tqdm(total=num)
#     for idx in range(len(lists)):
#         pool.apply_async(func=copy_list,args=(output_path,"{}/train_{}.txt".format(output_path,idx),lists[idx]))
#         pbar.update(1)
#     pbar.close()
#     pool.close()
#     pool.join()


def copy_all(output_path,txt_path):
    with open(output_path + '/train.txt',mode='w',encoding='utf8') as fw:
        with open(txt_path,mode='r',encoding='utf8') as fr:
            i = 0
            for line in tqdm(fr.readlines()):
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                src_path = pathlib.Path(line[0])
                if not os.path.exists(str(src_path)):
                    continue
                new_path = str(pathlib.Path(output_path) / src_path.parents._parts[-2])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = os.path.join(new_path, str(i) + src_path.suffix)
                shutil.copy(str(src_path),new_path)
                fw.write(new_path + '\t' + line[1] + '\n')
                i+=1

if __name__ == '__main__':
    import config
    output_path = '/data1/zj/data/crnn/all_test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    s = config.testfile
    copy_all(output_path,s)
