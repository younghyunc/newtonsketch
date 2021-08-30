import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

def load_data(dataset, n=2**14, d=2**8, df=2, nth=0):
    if dataset == 'synthetic_orthogonal':
        return generate_orthogonal(n=n, d=d)
    elif dataset == 'synthetic_high_coherence':
        return generate_high_coherence(n=n, d=d, df=df)
    elif dataset == 'cifar-10':
        return load_cifar(n=n, d=d)
    elif dataset == 'mnist_10Kn':
        return load_mnist_10Kn()
    elif dataset == 'susy_10Kn':
        return load_susy_10Kn()
    elif dataset == 'susy_100Kn':
        return load_susy_100Kn(nth=nth)
    elif dataset == 'epsilon_normalized_10Kn':
        return load_epsilon_normalized_10Kn(nth=nth)
    elif dataset == 'musk':
        data = np.load('./data/musk.npz')
        A, b = data['A'], data['b']
        return torch.tensor(A), torch.tensor(b)
    elif dataset == 'wesad':
        data = np.load('./data/wesad.npz')
        A, b = data['A'], data['b']
        return torch.tensor(A), torch.tensor(b)



def generate_high_coherence(n=2**12, d=2**10, c=1, df=1, lr=False):
    def cov_mat(d):
        Sigma = np.zeros((d,d))
        for ii in range(d):
            for jj in range(d):
                Sigma[ii,jj] = 2 * 0.5**(np.abs(ii-jj))
        return Sigma 
    def mvt(n_samples, Sigma, df):
        d = len(Sigma)
        g = np.tile(np.random.gamma(df/2., 2./df, n_samples), (d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d), Sigma, n_samples)
        return Z / np.sqrt(g)
    A = mvt(n, cov_mat(d), df=df)
    x_pl = np.ones((d,1))
    x_pl[10:-10] = 0.1
    b = A @ x_pl + 0.09 * np.random.randn(n,1)
    _, sigma, _ = np.linalg.svd(A, full_matrices=False)
    condition_number = sigma[0] / sigma[-1]
    A = A / sigma[0]
    b = b / sigma[0]
    if lr:
        b = np.sign(b)
    return torch.tensor(A), torch.tensor(b)


def generate_orthogonal(n=2**12, d=2**10):
    A = np.random.randn(n,d)
    A, _, _ = np.linalg.svd(A, full_matrices=False)
    b = 1./np.sqrt(n) * np.random.randn(n,1)

    return torch.tensor(A), torch.tensor(b)



def load_cifar(n=2**13, d=2**8):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)
    
    A_, b_ = iter(trainloader).next()
    A_ = torch.tensor(A_.reshape((A_.shape[0], -1)), dtype=torch.float64)[:,:d]
    b = torch.tensor([-1 if b_[ii] % 2 == 0 else 1 for ii in range(len(b_))], dtype=torch.float64).reshape((-1,1))

    print ("A: ", A_)
    print ("b: ", b)

    return A_, b

def load_mnist_10Kn():
    # read data
    fname = './data/mnist_10Kn'
    prec = np.float64
    train_points = np.genfromtxt(
        fname + '_train.csv', delimiter=",", dtype=prec
    )
    train_labels = np.genfromtxt(
        fname + '_train_label.csv', delimiter=",", dtype=prec
    )
    #test_points = np.genfromtxt(
    #    fname + '_test.csv', delimiter=",", dtype=prec
    #)
    #test_labels = np.genfromtxt(
    #    fname + '_test_label.csv', delimiter=",", dtype=prec
    #)

    A = torch.tensor(train_points)
    d = torch.tensor(np.reshape(train_labels, (-1,1)))

    print ("A: ", A)
    print ("d: ", d)

    return A, d

def load_susy_10Kn():
    # read data
    fname = './data/susy_10Kn'
    prec = np.float64
    train_points = np.genfromtxt(
        fname + '_train.csv', delimiter=",", dtype=prec
    )
    train_labels = np.genfromtxt(
        fname + '_train_label.csv', delimiter=",", dtype=prec
    )
    #test_points = np.genfromtxt(
    #    fname + '_test.csv', delimiter=",", dtype=prec
    #)
    #test_labels = np.genfromtxt(
    #    fname + '_test_label.csv', delimiter=",", dtype=prec
    #)

    A = torch.tensor(train_points)
    d = torch.tensor(np.reshape(train_labels, (-1,1)))

    print ("A: ", A)
    print ("d: ", d)

    return A, d

def load_susy_100Kn(nth=0):
    # read data
    fname = './data/susy_1Mn'
    prec = np.float64
    train_points = np.genfromtxt(
        fname + '_train.csv', delimiter=",", dtype=prec
    )
    train_labels = np.genfromtxt(
        fname + '_train_label.csv', delimiter=",", dtype=prec
    )

    train_points = train_points[nth*100000:(nth+1)*100000]
    train_labels = train_labels[nth*100000:(nth+1)*100000]

    #test_points = np.genfromtxt(
    #    fname + '_test.csv', delimiter=",", dtype=prec
    #)
    #test_labels = np.genfromtxt(
    #    fname + '_test_label.csv', delimiter=",", dtype=prec
    #)

    A = torch.tensor(train_points)
    d = torch.tensor(np.reshape(train_labels, (-1,1)))

    print ("A: ", A)
    print ("d: ", d)

    return A, d

def load_epsilon_normalized_10Kn(nth=0):
    # read data
    fname = './data/epsilon_normalized_10Kn'
    prec = np.float64
    train_points = np.genfromtxt(
        fname + '_train_'+str(nth)+'.csv', delimiter=",", dtype=prec
    )
    train_labels = np.genfromtxt(
        fname + '_train_label_'+str(nth)+'.csv', delimiter=",", dtype=prec
    )

    A = torch.tensor(train_points)
    d = torch.tensor(np.reshape(train_labels, (-1,1)))

    print ("A: ", A)
    print ("d: ", d)

    return A, d
