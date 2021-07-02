def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        # Reshape tensor to chunk of 1-d vectors.  28 * 28 --> 784
        x = x.view(x.size(0), -1)

    return x, y
