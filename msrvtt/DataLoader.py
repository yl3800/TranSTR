import torch
import os
import h5py
import os.path as osp
import numpy as np
from torch.utils import data
from utils.util import load_file, pause
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast

class VideoQADataset(Dataset):
    def __init__(self, split, obj_num=1, sample_list_path="/data/vqa/msrvtt/anno",\
         video_feature_path="/VideoQA/data/msrvtt" ):
        super(VideoQADataset, self).__init__()
        # 读取dataset
        self.sample_list_file = osp.join(sample_list_path, "{}.csv".format(split))
        self.sample_list = load_file(self.sample_list_file)
        ans=load_file('/msrvtt/anno/ans_word_5000.json')
        self.ans2idx={a:idx for idx,a in enumerate(ans)}
        print(len(ans))
        self.sample_list['answer_id'] = self.sample_list['answer'].map(self.ans2idx)
        if split=='train':
            self.sample_list['answer_id'].fillna(len(self.ans2idx), inplace=True)
            self.ans_group=self.sample_list.groupby("answer_id")
        else:
            self.sample_list['answer_id'].fillna(-1, inplace=True)

        self.split = split

        # 读取video feature
        frame_feat_file = osp.join('/model_base_vqa_capfilt_large', '{}.h5'.format(split))
        object_feat_file = osp.join(video_feature_path, "region_feat_n/acregion_8c10b_{}.h5".format(split))
        print("Loading {} ...".format(frame_feat_file))
        print("Loading {} ...".format(object_feat_file))
        self.frame_feats = {}
        self.obj_feats = {}
        self.frame_vid2idx = {}
        self.obj_vid2idx = {}

        # frame feature
        with h5py.File(frame_feat_file, "r") as fp:
            vids = fp["ids"]
            feats = fp["feat"][:, :, :] 
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.frame_feats[str(vid)] = feat
                self.frame_vid2idx[str(vid)] = id

        # object feature
        with h5py.File(object_feat_file, "r") as fp:
            vids = fp["ids"]
            feats = fp["feat"][:, ::2, :, :obj_num, :]
            bboxes = fp["bbox"][:, ::2, :, :obj_num, :]
            for id, (vid, feat, bbox) in enumerate(zip(vids, feats, bboxes)):
                self.obj_feats[str(vid)] = np.concatenate((feat, bbox), axis=-1)  # (8,2,obj_num,2048+4)
                self.obj_vid2idx[str(vid)] = id


    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]
        video_name, ans_id, qns_key, qns_word  = \
            str(cur_sample['video_id']), int(cur_sample['answer_id']), str(cur_sample['qid']), str(cur_sample['question'])
        
        vid_frame_feat = torch.from_numpy(self.frame_feats[video_name]).type(torch.float32)
        vid_obj_feat = torch.from_numpy(self.obj_feats[video_name]).type(torch.float32)

        return vid_frame_feat, vid_obj_feat.flatten(0,1), qns_word, ans_id, qns_key


    def __len__(self):
        return len(self.sample_list)



if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser(description="next logger")

    train_dataset=VideoQADataset('val', 10)

    train_loader = DataLoader(dataset=train_dataset,batch_size=2,shuffle=False,num_workers=0)

    for sample in train_loader:
        vid_frame_feat, vid_obj_feat, qns_word, ans_id, qns_key = sample
        print("frame feat: ")
        print(vid_frame_feat.size())
        print("object feat: ")
        print(vid_obj_feat.size())
        print("qns_word: ")
        print(qns_word)
        print("ans id: ")
        print(ans_id)
        print("qns key: ")
        print(qns_key)
        break
    print('done')