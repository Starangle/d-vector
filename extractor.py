import librosa
import os
import numpy as np
import threadpool

def convertor(npy_file):
    return converted

if __name__=='__main__':
    clean100=open('/home/sub18/code/data_catalogue/train-clean-100.list').read().split()
    clean360=open('/home/sub18/code/data_catalogue/train-clean-360.list').read().split()
    save_path='/home/sub18/code/data/mfcc'
    clean_cata_100='/home/sub18/code/data_catalogue/mfcc_clean_100.list'
    clean_cata_360='/home/sub18/code/data_catalogue/mfcc_clean_360.list'
    
    def extract_from_file(src,dst):
        frame_length=2048
        hop_length=512
        y,sr=librosa.load(src)
        mfcc=librosa.feature.mfcc(y=y,n_mfcc=40,n_fft=frame_length,
                                  hop_length=hop_length).T
        np.save(dst,mfcc)

    def make_params(src):
        base=os.path.basename(src).split('.')[0]
        save_name=os.path.join(save_path,base)+'.npy'
        return ([src,save_name],None)
    
    def write_cata(params,cata):
        tmp=list(zip(*params))[0]
        tmp=list(zip(*tmp))[1]
        f=open(cata,'w')
        for item in tmp:
            f.write(item+'\n')
        f.close()

    params100=list(map(make_params,clean100))
    params360=list(map(make_params,clean360))

    write_cata(params100,clean_cata_100)
    write_cata(params360,clean_cata_360)
    
    params=params100+params360

#   it seemed thread not safe
#    pool=threadpool.ThreadPool(16)
#    requests=threadpool.makeRequests(extract_from_file,params)
#    [pool.putRequest(req) for req in requests]
#    pool.wait()

    for p,_ in params:
        src,dst=p
        extract_from_file(src,dst)
        
