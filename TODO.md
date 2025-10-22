# TODO: Task Aware Compressor

## Group 1: Dataset Loading & Filtering [DONE]
- [x] Load and preprocess CIFAR-10 images (resize, convert to tensor, normalize).  
- [x] Filter dataset to include only 'cat' and 'dog', remapping labels (`cat -> 1`, `dog -> 0`).  
- [x] Implement PyTorch `Dataset` class for the filtered subset.  
- [x] Convert full training set into the specialized Cat/Dog dataset instance.  

## Group 2: Data Loading & ResNet Setup [DONE]
- [x] Set up DataLoaders with the desired batch size for training and testing.  
- [x] Load pretrained ResNet18.  
- [x] Modify the final fully connected layer of ResNet18 for binary classification.  
- [x] Freeze all layers except `layer3`, `layer4`, and `fc`.  

## Group 3: Fine-Tuning & Latent Module Prep
- [ ] Implement the fine-tuning loop (zero gradients, forward pass, loss computation, backprop, parameter update).  
- [ ] Record training and validation loss and accuracy after fine-tuning.  
- [ ] Implement `ResNet18_ImageLatent` forward pass to extract features up to `layer3`.  
- [ ] Implement `ResNet18_SemanticLatent` module (uses `layer4` and `fc`) for classification.  
- [ ] Verify that `image_latent` stops at `layer3` and check input channels of first conv in `layer4` of `semantic_latent`.  

## Group 4: Decoder & Evaluation
- [ ] Define Decoder architecture (ConvTranspose layers and final activation).  
- [ ] Instantiate Decoder and set core hyperparameters (ALPHA, LR, NUM_EPOCHS).  
- [ ] Implement utility functions: denormalize, save tensor as JPEG, reconstruct image, predict class.  
- [ ] Set models to evaluation/training mode, train Decoder jointly with frozen ResNet.  
- [ ] Compute latent code, reconstructed image, classification logits, CE/MSE loss, and total loss.  
- [ ] Evaluate latent and reconstructed accuracy, save reconstructed images, and calculate file sizes.  
- [ ] Compress reconstructed images to target size and evaluate compressed accuracy.  
- [ ] Display final results tables for Tanh and Sigmoid runs, including average file sizes and accuracy metrics.  
