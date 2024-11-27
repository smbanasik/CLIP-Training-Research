def getDataFromSet(dataset):
    images = []
    targets = []

    for img, targ in dataset:
        images.append(img)
        targets.append(targ)

    return np.array(images), np.array(targets)

class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=28, crop_size=24, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        self.transform_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop((crop_size, crop_size), padding=None),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((image_size, image_size)),
                                ])
        self.transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size))
                                ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target

class ImageDataset_APLoss(Dataset):
    def __init__(self, images, targets, image_size=28, crop_size=24, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        self.transform_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop((crop_size, crop_size), padding=None),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((image_size, image_size)),
                                ])
        self.transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size))
                                ])
        # "For loss function"
        self.pos_indices = np.flatnonzero(targets==1)
        self.pos_index_map = {}
        for i, idx in enumerate(self.pos_indices):
            self.pos_index_map[idx] = i
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            idx = self.pos_index_map[idx] if idx in self.pos_indices else -1
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return idx, image, target


def load_data(paramaters):
    # Load the Data
    train_dataset = PneumoniaMNIST(split="train", download=False)
    eval_dataset = PneumoniaMNIST(split="val")
    eval_dataset = PneumoniaMNIST(split="test")

    tr_imgs, tr_labels = getDataFromSet(train_dataset)
    e_imgs, e_labels = getDataFromSet(eval_dataset)
    te_imgs, te_labels = getDataFromSet(train_dataset)

    # Augment data
    train_dataset = ImageDataset(tr_imgs, tr_labels)
    eval_dataset = ImageDataset(e_imgs, e_labels)
    test_dataset = ImageDataset(te_imgs, te_labels)

    sampler = DualSampler(train_dataset, parameters.batch_size, sampling_rate=0.2)
    trainloader = DataLoader(train_dataset, batch_size=parameters.batch_size, sampler=sampler, num_workers=2)
    evalloader = DataLoader(eval_dataset, batch_size=parameters.batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters.batch_size, shuffle=False, num_workers=2)

    return sampler, trainloader, evalloader, testloader