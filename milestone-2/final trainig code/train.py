# import argparse
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import pickle
# from data_loader import get_loader 
# from build_vocab import Vocabulary
# from model import EncoderCNN, DecoderRNN
# from torch.nn.utils.rnn import pack_padded_sequence
# from torchvision import transforms


# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def main(args):
#     # Create model directory
#     if not os.path.exists(args.model_path):
#         os.makedirs(args.model_path)

    
#     print("load vocabulary ...")    
#     # Load vocabulary wrapper
#     with open(args.vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
        
#       # Image preprocessing, normalization for the pretrained resnet
#     transform = transforms.Compose([
#     transforms.Resize(256),  
#     transforms.RandomCrop(args.crop_size),
#     transforms.RandomHorizontalFlip(), 
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    
#     print("build data loader ...")
#     # Build data loader
#     data_loader = get_loader(args.image_dir, args.caption_path, vocab,transform, args.batch_size,
#                              shuffle=True, num_workers=args.num_workers) 
    
#     print("build the models ...")
#     # Build the models
#     encoder = EncoderCNN(args.embed_size).to(device)
#     decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
#     #encoder.load_state_dict(torch.load("models/encoder-2-1000.ckpt"))
#     #decoder.load_state_dict(torch.load("models/decoder-2-1000.ckpt")) 


#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     params = list(decoder.parameters())# + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
#     optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
#     # Train the models
#     total_step = len(data_loader)
#     for epoch in range(args.num_epochs):
#         for i, (images, captions, lengths) in enumerate(data_loader):
            
#             # Set mini-batch dataset
#             images = images.to(device)
#             captions = captions.to(device)
#             # Add this before the pack_padded_sequence call
#             # Sort sequences by length in decreasing order
#             lengths, sort_idx = lengths.sort(0, descending=True)
#             captions = captions[sort_idx]
#             images = images[sort_idx]
              
            
#             # Forward, backward and optimize
#             features = encoder(images)
#             outputs = decoder(features, captions, lengths)
#             # outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
#             targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
#             loss = criterion(outputs, targets)
#             decoder.zero_grad()
#             encoder.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # Print log info
#             if i % args.log_step == 0:
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
#                       .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
#             # Save the model checkpoints
#             if (i+1) % args.save_step == 0:
#                 torch.save(decoder.state_dict(), os.path.join(
#                     args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
#                 torch.save(encoder.state_dict(), os.path.join(
#                     args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
#     parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
#     parser.add_argument('--vocab_path', type=str, default='data/zh_vocab.pkl', help='path for vocabulary wrapper')
#     parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
#     parser.add_argument('--caption_path', type=str, default='data/annotations/img_tag.txt', help='path for train annotation json file')
#     parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
#     parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
#     # Model parameters
#     parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
#     parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
#     parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    
#     parser.add_argument('--num_epochs', type=int, default=5)
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--num_workers', type=int, default=1)
#     parser.add_argument('--learning_rate', type=float, default=0.001)
#     args = parser.parse_args()
#     print(args)
#     main(args)


import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Subset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print("Load vocabulary...")    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    print("Build data loader...")
    # Get the full dataset first
    from pycocotools.coco import COCO
    coco = COCO(args.caption_path)
    ids = list(coco.anns.keys())
    

    subset_size = int(len(ids) * 0.25)  # Using 25% of data
    subset_ids = random.sample(ids, subset_size)
    print(f"Training on {subset_size} samples (15% of full dataset)")
    
    # Build data loader with the subset
    data_loader = get_loader(
        args.image_dir, 
        args.caption_path, 
        vocab, 
        transform, 
        args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        ids_subset=subset_ids
    )
    
    print("Build the models...")
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Train the models
    total_step = len(data_loader)
    print(f"Total steps per epoch: {total_step}")
    
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            
            # Sort sequences by length in decreasing order
            lengths, sort_idx = lengths.sort(0, descending=True)
            captions = captions[sort_idx]
            images = images[sort_idx]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                # Forward pass
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                loss = criterion(outputs, targets)
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/zh_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/img_tag.txt', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
