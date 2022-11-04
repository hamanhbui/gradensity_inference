import logging
import math
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.VAE.src.dataloaders import dataloader_factory
from algorithms.VAE.src.models import model_factory
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.load_metadata import set_test_samples_labels, set_tr_val_samples_labels
from torch.optim.lr_scheduler import MultiStepLR
import xlwt
from xlwt import Workbook


class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, z):
        y = self.classifier(z)
        return y


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.relu(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def gaussian_likelihood(self, x_hat, x):
        dist = torch.distributions.Normal(x, torch.tensor([1.0]).cuda())
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x_hat)
        return log_pxz


def vae_loss_function(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction="mean")
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


class Trainer_VAE:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.exp_idx = exp_idx
        self.checkpoint_name = (
            "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + self.exp_idx
        )
        self.plot_dir = (
            "algorithms/" + self.args.algorithm + "/results/plots/" + self.args.exp_name + "_" + self.exp_idx + "/"
        )
        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(self.args.feature_dim, self.args.n_classes).to(self.device)
        self.vae = VAE(x_dim=self.args.feature_dim, h_dim1=512, h_dim2=256, z_dim=2).to(self.device)

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def init_training(self):
        logging.basicConfig(
            filename="algorithms/"
            + self.args.algorithm
            + "/results/logs/"
            + self.args.exp_name
            + "_"
            + self.exp_idx
            + ".log",
            filemode="w",
            level=logging.INFO,
        )
        self.writer = self.set_writer(
            log_dir="algorithms/"
            + self.args.algorithm
            + "/results/tensorboards/"
            + self.args.exp_name
            + "_"
            + self.exp_idx
            + "/"
        )
        (
            tr_sample_paths,
            tr_class_labels,
            val_sample_paths,
            val_class_labels,
        ) = set_tr_val_samples_labels(self.args.train_meta_filenames, self.args.val_size)
        self.train_loader = DataLoader(
            dataloader_factory.get_train_dataloader(self.args.dataset)(
                path=self.args.train_path,
                sample_paths=tr_sample_paths,
                class_labels=tr_class_labels,
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        if self.args.val_size != 0:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    path=self.args.train_path,
                    sample_paths=val_sample_paths,
                    class_labels=val_class_labels,
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
        else:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    path=self.args.train_path, sample_paths=test_sample_paths, class_labels=test_class_labels
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
        optimizer_params = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.SGD(optimizer_params, lr=self.args.learning_rate, momentum=0.9, nesterov=True)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.decay_interations, gamma=0.2)
        self.density_optimizer = torch.optim.SGD(self.vae.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.val_loss_min = np.Inf
        self.val_acc_max = 0

    def train(self):
        self.init_training()
        self.model.train()
        self.classifier.train()
        n_class_corrected, total_classification_loss, total_samples = 0, 0, 0
        self.train_iter_loader = iter(self.train_loader)
        for iteration in range(self.args.iterations):
            if (iteration % len(self.train_iter_loader)) == 0:
                self.train_iter_loader = iter(self.train_loader)
            samples, labels = self.train_iter_loader.next()
            samples, labels = samples.to(self.device), labels.to(self.device)
            predicted_classes = self.classifier(self.model(samples))
            classification_loss = self.criterion(predicted_classes, labels)
            total_classification_loss += classification_loss.item()
            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)
            self.optimizer.zero_grad()
            classification_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if iteration % self.args.step_eval == (self.args.step_eval - 1):
                self.writer.add_scalar("Accuracy/train", 100.0 * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar("Loss/train", total_classification_loss / self.args.step_eval, iteration)
                logging.info(
                    "Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                        iteration + 1,
                        self.args.iterations,
                        n_class_corrected,
                        total_samples,
                        100.0 * n_class_corrected / total_samples,
                        total_classification_loss / self.args.step_eval,
                    )
                )
                self.evaluate(iteration)
                n_class_corrected, total_classification_loss, total_samples = 0, 0, 0
        self.estimate_density()

    def estimate_density(self):
        self.model.eval()
        self.vae.train()
        total_estimation_loss = 0
        self.train_iter_loader = iter(self.train_loader)
        for iteration in range(self.args.density_estimation_iterations):
            if (iteration % len(self.train_iter_loader)) == 0:
                self.train_iter_loader = iter(self.train_loader)
            samples, labels = self.train_iter_loader.next()
            samples, labels = samples.to(self.device), labels.to(self.device)
            latents = self.model(samples)
            recon_batch, mu, log_var = self.vae(latents.detach())
            density_loss = vae_loss_function(recon_batch, latents, mu, log_var)
            total_estimation_loss += density_loss
            self.density_optimizer.zero_grad()
            density_loss.backward()
            self.density_optimizer.step()
            if iteration % self.args.step_eval == (self.args.step_eval - 1):
                self.writer.add_scalar("Loss/density", total_estimation_loss / self.args.step_eval, iteration)
                logging.info(
                    "Train set: Iteration: [{}/{}]\tLoss: {:.6f}".format(
                        iteration + 1,
                        self.args.density_estimation_iterations,
                        total_estimation_loss / self.args.step_eval,
                    )
                )
                total_estimation_loss = 0
        torch.save(
            {
                "score_state_dict": self.vae.state_dict(),
            },
            self.checkpoint_name + "_score.pt",
        )

    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()
        n_class_corrected, total_classification_loss = 0, 0
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.val_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        self.writer.add_scalar("Accuracy/validate", 100.0 * n_class_corrected / len(self.val_loader.dataset), n_iter)
        self.writer.add_scalar("Loss/validate", total_classification_loss / len(self.val_loader), n_iter)
        logging.info(
            "Val set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                n_class_corrected,
                len(self.val_loader.dataset),
                100.0 * n_class_corrected / len(self.val_loader.dataset),
                total_classification_loss / len(self.val_loader),
            )
        )
        val_acc = n_class_corrected / len(self.val_loader.dataset)
        val_loss = total_classification_loss / len(self.val_loader)
        self.model.train()
        self.classifier.train()
        if self.args.val_size != 0:
            if self.val_loss_min > val_loss:
                self.val_loss_min = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )
        else:
            if self.val_acc_max < val_acc:
                self.val_acc_max = val_acc
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )

    def test(self):
        self.wb = Workbook()
        self.sheet = self.wb.add_sheet("test_path")
        test_sample_paths, test_class_labels = set_test_samples_labels(self.args.test_meta_filenames)
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()
        for i in range(len(self.args.test_paths)):
            test_path = self.args.test_paths[i]
            self.test_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    path=test_path, sample_paths=test_sample_paths, class_labels=test_class_labels
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
            n_class_corrected = 0
            with torch.no_grad():
                for iteration, (samples, labels) in enumerate(self.test_loader):
                    samples, labels = samples.to(self.device), labels.to(self.device)
                    predicted_classes = self.classifier(self.model(samples))
                    _, predicted_classes = torch.max(predicted_classes, 1)
                    n_class_corrected += (predicted_classes == labels).sum().item()
            test_acc = 100.0 * n_class_corrected / len(self.test_loader.dataset)
            print(
                test_path + "\tTest set: Accuracy: {}/{} ({:.2f}%)".format(
                    n_class_corrected,
                    len(self.test_loader.dataset),
                    test_acc,
                )
            )
            self.sheet.write(0, i, test_acc)
            adapt_acc = self.adapt_test()
            self.sheet.write(1, i, adapt_acc)
        self.wb.save('xlwt example.xls')

    def adapt_test(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        checkpoint_score = torch.load(self.checkpoint_name + "_score.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.vae.load_state_dict(checkpoint_score["score_state_dict"])
        self.model.eval()
        self.classifier.eval()
        self.vae.eval()
        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                latents = self.model(samples)
                latents_s = latents
                with torch.enable_grad():
                    for i in range(self.args.adaptive_iterations):
                        z = latents.clone().detach().requires_grad_(True)
                        recon_batch, mu, log_var = self.vae(z)
                        log_pxz = self.vae.gaussian_likelihood(recon_batch, latents_s)
                        grad1 = torch.autograd.grad(log_pxz, z, grad_outputs=torch.ones_like(log_pxz))[0]
                        latents = latents.add(grad1 * self.args.adaptive_rate)
                predicted_classes = self.classifier(latents)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        test_acc = 100.0 * n_class_corrected / len(self.test_loader.dataset)
        print(
            "Test set: Accuracy: {}/{} ({:.2f}%)".format(
                n_class_corrected,
                len(self.test_loader.dataset),
                test_acc,
            )
        )
        return test_acc

    def save_plot(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        checkpoint_score = torch.load(self.checkpoint_name + "_score.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.vae.load_state_dict(checkpoint_score["score_state_dict"])
        self.model.eval()
        self.classifier.eval()
        self.vae.eval()
        Z_train, Y_train, Z_test, Y_test, Z_adapt = [], [], [], [], []
        tr_nlls, tr_entropies, te_nlls, te_entropies, adapt_nlls, adapt_entropies = [], [], [], [], [], []
        nn_softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.train_loader):
                b, c, h, w = samples.shape
                samples, labels = samples.to(self.device), labels.to(self.device)
                z = self.model(samples)
                z_s = z
                predicted_classes = self.classifier(z)
                predicted_softmaxs = nn_softmax(predicted_classes)
                for predicted_softmax in predicted_softmaxs:
                    tr_entropies.append(entropy(predicted_softmax.cpu().detach().numpy()))
                classification_loss = self.criterion(predicted_classes, labels)
                recon_batch, mu, log_var = self.vae(z)
                log_pxz = self.vae.gaussian_likelihood(recon_batch, z_s)
                bpd = (log_pxz.cpu().detach().sum()) / b
                tr_nlls.append(bpd.numpy())
                Z_train += z.tolist()
                Y_train += labels.tolist()
            for iteration, (samples, labels) in enumerate(self.test_loader):
                b, c, h, w = samples.shape
                samples, labels = samples.to(self.device), labels.to(self.device)
                z = self.model(samples)
                z_s = z
                predicted_classes = self.classifier(z)
                predicted_softmaxs = nn_softmax(predicted_classes)
                for predicted_softmax in predicted_softmaxs:
                    te_entropies.append(entropy(predicted_softmax.cpu().detach().numpy()))
                classification_loss = self.criterion(predicted_classes, labels)
                recon_batch, mu, log_var = self.vae(z)
                log_pxz = self.vae.gaussian_likelihood(recon_batch, z_s)
                bpd = (log_pxz.cpu().detach().sum()) / b
                te_nlls.append(bpd.numpy())
                Z_test += z.tolist()
                Y_test += labels.tolist()
                with torch.enable_grad():
                    for i in range(self.args.adaptive_iterations):
                        latents = z.clone().detach().requires_grad_(True)
                        recon_batch, mu, log_var = self.vae(latents)
                        log_pxz = self.vae.gaussian_likelihood(recon_batch, z_s)
                        grad1 = torch.autograd.grad(log_pxz, latents, grad_outputs=torch.ones_like(log_pxz))[0]
                        z = z.add(grad1 * self.args.adaptive_rate)
                predicted_classes = self.classifier(z)
                predicted_softmaxs = nn_softmax(predicted_classes)
                for predicted_softmax in predicted_softmaxs:
                    adapt_entropies.append(entropy(predicted_softmax.cpu().detach().numpy()))
                classification_loss = self.criterion(predicted_classes, labels)
                recon_batch, mu, log_var = self.vae(z)
                log_pxz = self.vae.gaussian_likelihood(recon_batch, z_s)
                bpd = (log_pxz.cpu().detach().sum()) / b
                adapt_nlls.append(bpd.numpy())
                Z_adapt += z.tolist()
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        with open(self.plot_dir + "Z_train.pkl", "wb") as fp:
            pickle.dump(Z_train, fp)
        with open(self.plot_dir + "Y_train.pkl", "wb") as fp:
            pickle.dump(Y_train, fp)
        with open(self.plot_dir + "Z_test.pkl", "wb") as fp:
            pickle.dump(Z_test, fp)
        with open(self.plot_dir + "Z_adapt.pkl", "wb") as fp:
            pickle.dump(Z_adapt, fp)
        with open(self.plot_dir + "Y_test.pkl", "wb") as fp:
            pickle.dump(Y_test, fp)
        with open(self.plot_dir + "tr_nlls.pkl", "wb") as fp:
            pickle.dump(tr_nlls, fp)
        with open(self.plot_dir + "tr_entropies.pkl", "wb") as fp:
            pickle.dump(tr_entropies, fp)
        with open(self.plot_dir + "te_nlls.pkl", "wb") as fp:
            pickle.dump(te_nlls, fp)
        with open(self.plot_dir + "te_entropies.pkl", "wb") as fp:
            pickle.dump(te_entropies, fp)
        with open(self.plot_dir + "adapt_nlls.pkl", "wb") as fp:
            pickle.dump(adapt_nlls, fp)
        with open(self.plot_dir + "adapt_entropies.pkl", "wb") as fp:
            pickle.dump(adapt_entropies, fp)
