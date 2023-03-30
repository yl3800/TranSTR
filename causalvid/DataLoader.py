import torch
import os
import h5py
import os.path as osp
import numpy as np
from torch.utils import data
from utils.util import load_file, pause, transform_bb
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast

class VideoQADataset(Dataset):
    def __init__(self, split, n_query=5, obj_num=1, sample_list_path="/data/vqa/causal/anno",\
         video_feature_path="/region_feat_aln" ):
        super(VideoQADataset, self).__init__()
        # 读取dataset
        self.sample_list_file = osp.join(sample_list_path, "{}.csv".format(split))
        self.sample_list = load_file(self.sample_list_file)
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        # 读取video feature
        
        frame_feat_file = osp.join('/model_base_vqa_capfilt_large', '{}.h5'.format(split))
        # object_feat_file = osp.join(video_feature_path, "region_feat_n/acregion_8c20b_{}.h5".format(split))
        self.map_dir = load_file(osp.join(sample_list_path, 'map_dir_caul.json'))
        self.obj_feat_dir = video_feature_path
        
        print("Loading {} ...".format(frame_feat_file))
        # print("Loading {} ...".format(object_feat_file))
        self.frame_feats = {}
        self.obj_feats = {}
        self.frame_vid2idx = {}
        self.obj_vid2idx = {}

        # frame feature
        with h5py.File(frame_feat_file, "r") as fp:
            vids = fp["ids"]
            feats = fp["feat"][:, :, :] 
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.frame_feats[str(vid.decode())] = feat
                self.frame_vid2idx[str(vid.decode())] = id

        # # object feature
        # with h5py.File(object_feat_file, "r") as fp:
        #     vids = fp["ids"]
        #     feats = fp["feat"][:, ::2, :, :obj_num, :]
        #     bboxes = fp["bbox"][:, ::2, :, :obj_num, :]
        #     for id, (vid, feat, bbox) in enumerate(zip(vids, feats, bboxes)):
        #         self.obj_feats[str(vid)] = np.concatenate((feat, bbox), axis=-1)  # (8,2,obj_num,2048+4)
        #         self.obj_vid2idx[str(vid)] = id


    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]
        width, height = cur_sample['width'], cur_sample['height']
        video_name = str(cur_sample["video_id"])
        qns_word = str(cur_sample["question"])
        # ans_id = int(cur_sample['answer'])
        ans_id = self.find_answer_num(cur_sample)
        # video_name, qns_word, qid, ans_id = str(cur_sample['video_id']), str(cur_sample['question']), str(cur_sample['qid']), int(cur_sample['answer'])
        ans_word = ['[CLS] ' + qns_word+' [SEP] '+ str(cur_sample["a" + str(i)]) for i in range(self.mc)]

        vid_frame_feat = torch.from_numpy(self.frame_feats[video_name]).type(torch.float32)
        # vid_obj_feat = torch.from_numpy(self.obj_feats[video_name]).type(torch.float32)
        qns_key = video_name + '_' + str(cur_sample["type"])

        region_feat = np.load(osp.join(self.obj_feat_dir, self.map_dir[video_name]+'.npz'))
        roi_feat, roi_bbox = region_feat['feat'][::2, :, :self.obj_num, :], region_feat['bbox'][::2, :, :self.obj_num, :]
        bbox_feat = transform_bb(roi_bbox, width, height)
        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)
        vid_obj_feat = torch.cat((roi_feat, bbox_feat), dim=-1)

        return vid_frame_feat, vid_obj_feat.flatten(0,1), qns_word, ans_word, ans_id, qns_key


    def __len__(self):
        return len(self.sample_list)

    def find_answer_num(self, cur_sample):
        # to find the answer num according to the given answer text
        answer = str(cur_sample["answer"])  # this is the text of the answer
        answer_list = []  # to store all the answer
        for i in range(self.mc):
            answer_list.append(str(cur_sample["a" + str(i)]))
        # answer text match
        for i in range(self.mc):
            if (answer == answer_list[i]):
                return int(i)
        return None  # fail in matching

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="next logger")
    parser.add_argument('-dataset', default='nextqa',choices=['nextqa'], type=str)
    args = parser.parse_args()
    # video_feature_path = '/storage_fast/jbxiao/workspace/VideoQA/data/nextqa'
    # sample_list_path = '/storage_fast/ycli/data/vqa/next/anno'
    train_dataset=VideoQADataset('val', 5, 3)

    train_loader = DataLoader(dataset=train_dataset,batch_size=2,shuffle=False,num_workers=0)

    for sample in train_loader:
        vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key = sample
        print("frame feat: ")
        print(vid_frame_feat.size())
        print("object feat: ")
        print(vid_obj_feat.size())
        print("qns_word: ")
        print(qns_word)
        print("ans_word")
        print(ans_word)
        print("ans id: ")
        print(ans_id)
        print("qns key: ")
        print(qns_key)
        break
    print('done')