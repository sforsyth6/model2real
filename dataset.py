import torchvision.datasets as datasets
from PIL import Image

import torch

class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def get_concat_v(self, im1, im2):
        
        dst = torch.cat((im1,im2), dim=0)
        
        #dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        #dst.paste(im1, (0, 0))
        #dst.paste(im2, (0, im1.height))

        return dst

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

    #    imgs = img

        name = path.split("/")[-1]
        num = int(name.split("-")[-1].split(".jpg")[0])

    #    tmp_name = "-".join(name.split("-")[:-1])
    #    tmp = []

    #    tnum = num

        '''
        for i in range(num+1,num+5):
            try:
                tmp_path = "/".join(path.split("/")[:-1]) + "/" + tmp_name +"-{}.jpg".format(i)
                tmp_img = self.loader(tmp_path)
                if self.transform is not None:
                    tmp_img = self.transform(tmp_img)
                imgs = self.get_concat_v(imgs,tmp_img)

                tnum = i
            except:
                tmp_img = torch.zeros(img.shape)
                imgs = self.get_concat_v(imgs,tmp_img)
        '''     

        tpath = path.split("train/a310")
        tpath[0] += "masks"
        tpath = "".join(tpath)

#        tpath = "/".join(path.split("/")[:3]) + "/masks/" + name # + "-{}.jpg".format(num)
        target = self.loader(tpath)

        
        if self.target_transform is not None:
            target = self.target_transform(target)/255.0

        return img, target