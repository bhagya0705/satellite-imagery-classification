# Study guide - how to explain this project

## 60-second explanation
"I built a classifier that looks at a satellite image and says what the land is used for - forest, river, highway, residential and so on - using the EuroSAT dataset of 27,000 images. Instead of training a network from scratch, I used transfer learning: a ResNet18 already trained on ImageNet, with its last layer replaced for my 10 classes. I first trained only that new layer, then fine-tuned the whole network with a smaller learning rate. I used flips and rotations as augmentation, split the data into train/validation/test, picked the best epoch on validation, and evaluated once on the test set with a confusion matrix and per-class F1 to see which classes get confused."

## Concepts to understand (in this order)
1. **Image = numbers.** A 64x64 RGB image is a 3x64x64 grid of numbers; the model turns it into 10 class scores.
2. **CNN.** Layers of small filters slide over the image; early layers find edges, later layers find shapes and textures.
3. **Transfer learning.** Re-use features learned on a huge dataset (ImageNet) instead of learning from nothing. Needs less data and time.
4. **Freezing / fine-tuning.** Frozen = weights not updated. Stage 1 trains only the new head; stage 2 unfreezes everything with a small learning rate.
5. **Loss, optimizer, epoch.** Cross-entropy measures how wrong the predictions are; Adam adjusts the weights to reduce it; one epoch = one pass over the training data.
6. **Train / validation / test.** Train = learn; validation = choose settings and the best epoch; test = final honest score, used once.
7. **Overfitting.** Train accuracy keeps rising but validation stops improving. Augmentation helps against it.
8. **Metrics.** Accuracy; precision (of predicted X, how many are X); recall (of real X, how many found); F1 (balance); confusion matrix (which classes get mixed up).

## Likely interview questions
- **Why transfer learning?** Less data and training time, and better accuracy than training from scratch on 27k small images.
- **Why freeze the backbone first?** The new head starts random; training it alone first avoids large, noisy gradients wrecking the pretrained features.
- **Why a lower learning rate for fine-tuning?** To nudge pretrained weights gently instead of destroying them.
- **Why these augmentations?** Satellite images are rotation/flip invariant, so the transformed images are still valid examples.
- **Why normalise with ImageNet mean/std?** The pretrained network was trained with inputs in that range.
- **Why stratified split?** Keeps class proportions equal in train/val/test.
- **Why a separate test set?** The validation set is used to pick the best epoch, so it is slightly optimistic; the test set gives an unbiased estimate.
- **Why not just accuracy?** It can hide weak classes; per-class F1 and the confusion matrix show them.
- **What is the bootstrap confidence interval?** Resample the test predictions many times to see how much the accuracy could vary by chance.
- **Which classes were confused and why?** (Answer from your own confusion matrix - typically the crop/pasture/vegetation classes, which look similar from above.)
- **How would you improve it?** Bigger backbone, class weights, more augmentation, test-time augmentation, Grad-CAM to inspect failures, multispectral bands instead of RGB.
- **What are the limitations?** Small low-resolution images, single sensor/region; would need retraining for other imagery.

## Code map (notebook cells)
1. Setup/config -> 2. Transforms -> 3. Load data -> 4. EDA -> 5. Split + loaders -> 6. Model -> 7. Training loop -> 8. Evaluation -> 9. Save -> 10. Single-image prediction.

Read the notebook once top to bottom, then re-run it changing one thing (e.g. remove augmentation, or `EPOCHS_FINETUNE = 0`) and note how the results change. That experiment alone gives you a great interview answer.
