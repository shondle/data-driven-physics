import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import tensorboard


# Things to try to get loss to converge better
# 1. Larger network?
# 2. Better normalization with BatchNorm
# 3. Different learning rate
# 4. Change architecture to a CNN

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # making this a VERY simple model
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid() # to get values between 0 and 1
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    # z dim - dimension of the input latent noise vector
    def __init__(self, z_dim, img_dim):
        super().__init__()
        # making this a VERY simple model
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # 28 * 28 * 1 ---> 784 (size of MNIST image)
            nn.Tanh() # to get values between -1 and 1 (because the input is normalized this way as well)
        )

    def forward(self, x):
        return self.gen(x)
    

num_epochs = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
disc = Discriminator(784).to(device)
gen = Generator(z_dim=64, image_dim=784).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=3e-4)
opt_gen = optim.Adam(gen.parameters(), lr=3e-4)
criterion = nn.BCELoss()
writer_fake = tensorboard.SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = tensorboard.SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # train discriminator: max log(D(real)) + log(1 - D(G(z))) (this is BCE loss)
        # z is the random noise
        # we want a 1 in the first log, and a 0 in the second log

        # randn is a gaussian distribution
        noise = torch.randn(batch_size, z_dim=64).to(device)

        fake = gen(noise)
        disc_real = disc(real).view(-1)

        # we want it to be able to tell that it is real
        # aka min -log(D(real))
        # bc is D(real) is less than 1 -> log(D(real)) is negative (e^x where x is negative is less than 1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # now fake discriminator
        # aka min -log(1 - D(G(z)))

        # you can add fake.detach() to keep the fake gen(noise) for next step
        # we do this to keep fake gen(noise) for next step
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True) # we do this to keep fake gen(noise) for next step
        opt_disc.step()

        # so BCE loss equation is 
        # - wn [ yn log xn + (1 - yn) log (1 - xn) ]
        # the first term cancels out when yn is zeros (testing fake)
        # second terms cancels out when yn is 1 (real)
        # this is how we get it to mirror the GAN loss


        # now, on to the generator
        # min log(1 - D(G(z))) which is the same as max log(D(G(z))) [this gives faster gradients]
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        # you can add tensorboard additional code here