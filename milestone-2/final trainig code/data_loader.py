import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # Sort batch by descending caption lengths
    batch.sort(key=lambda x: x[2], reverse=True)
    
    images, captions, lengths = zip(*batch)
    
    # Convert to tensors and pad sequences
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    lengths = torch.LongTensor(lengths)
    
    return images, captions, lengths



class CocoDataset(data.Dataset):
    
    def __init__(self, root, json, vocab, transform=None, ids_subset=None):
      self.root = root
      self.coco = COCO(json)
      self.ids = ids_subset if ids_subset is not None else list(self.coco.anns.keys())
      self.vocab = vocab
      self.transform = transform

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
      """Returns one data pair (image and caption)."""
      vocab = self.vocab
      ann_id = self.ids[index]
    
      # Get caption and image id
      caption = self.coco.anns[ann_id]['caption']
      img_id = self.coco.anns[ann_id]['image_id']
    
      # Get the path to the image
      img_info = self.coco.loadImgs(img_id)[0]
      img_path = os.path.join(self.root, img_info['file_name'])
    
      # Load and transform image
      image = Image.open(img_path).convert('RGB')
      if self.transform is not None:
        image = self.transform(image)
    
      # Convert caption to word ids
      tokens = nltk.tokenize.word_tokenize(str(caption).lower())
      caption = []
      caption.append(vocab('<start>'))
      caption.extend([vocab(token) for token in tokens])
      caption.append(vocab('<end>'))
      caption = torch.Tensor(caption).long()
      
      return image, caption, len(caption)


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, ids_subset=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform, ids_subset=ids_subset)
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader


