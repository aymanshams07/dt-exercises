

class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # our dataset has two classes only - background and person
        num_classes = 2

        # get the model using our helper function
        model = get_instance_segmentation_model(num_classes)
        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.00001,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)


        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
        # TODO If no GPU is available, raise the NoGPUAvailable exception
        pass
    def predict(self, batch_or_image):
        # TODO Make your model predict here!
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

        return model

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!

            box, label, score = self.model.predict(img) # TODO you probably need to send the image to a tensor, etc.
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores

class Model():    # TODO probably extend a TF or Pytorch class!
    def __init__(self):
        # TODO Instantiate your weights etc here!
        pass
    # TODO add your own functions if need be!