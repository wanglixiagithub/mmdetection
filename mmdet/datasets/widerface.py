import numpy as np
#from pycocotools.cocoper import COCO

from .custom import CustomDataset


class WiderFaceDataset(CustomDataset):
# class WiderFaceDataset():

    def load_annotations(self, ann_file):
        with open(ann_file) as file:
            lines = file.readlines()
        img_infos = []
        i = 0
        while i < len(lines):
            if lines[i].strip()[-3:] == 'jpg':
                img = {}
                img['filename'] = lines[i].strip()
                bbnum = int(lines[i+1])
                i = i+1
                img['ann'] = {}
                for j in range(bbnum):
                    bbx = list(map(int,lines[i+j+1].strip().split()))
                    if bbx[-3] == 0:
                        if 'bboxes' in img['ann']:
                            img['ann']['bboxes'].append([bbx[0], bbx[1], bbx[0] + bbx[2] - 1, bbx[1] + bbx[3] - 1])
                            img['ann']['labels'].append([1])
                        else:
                            img['ann']['bboxes'] = []
                            img['ann']['labels'] = []
                            img['ann']['bboxes'].append([bbx[0], bbx[1], bbx[0]+bbx[2]-1, bbx[1]+bbx[3]-1])
                            img['ann']['labels'].append([1])
                    else:
                        if 'bboxes_ignore' in img['ann']:
                            if 'bboxes' in img['ann']:
                                img['ann']['bboxes_ignore'].append([bbx[0], bbx[1], bbx[0] + bbx[2] - 1, bbx[1] + bbx[3] - 1])
                            else:
                                img['ann']['bboxes_ignore'] = []
                                img['ann']['bboxes_ignore'].append([bbx[0], bbx[1], bbx[0] + bbx[2] - 1, bbx[1] + bbx[3] - 1])
                if 'bboxes' in img['ann']:
                    img['ann']['bboxes'] = np.array(img['ann']['bboxes'],dtype=np.float32)
                    img['ann']['labels'] = np.array(img['ann']['labels'],dtype=np.int64)
                else:
                    img['ann']['bboxes'] = np.zeros((0,4),dtype=np.float32)
                    img['ann']['labels'] = np.array([],dtype=np.int64)

                if 'bboxes_ignore' in img['ann']:
                    img['ann']['bboxes_ignore'] = np.array(img['ann']['bboxes_ignore'],dtype=np.float32)
                else:
                    img['ann']['bboxes_ignore'] = np.zeros((0,4),dtype=np.float32)
                i = i+bbnum+1
                img_infos.append(img)
        return img_infos

# if __name__ == '__main__':
#     dataset = WiderFaceDataset()
#     img_info = dataset.load_annotations('/home/wlx/Detection/mmdetection/data/widerface/wider_face_split/wider_face_train_bbx_gt.txt')

