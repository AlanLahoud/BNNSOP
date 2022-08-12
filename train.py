import torch

class TrainDecoupledElbo():
    
    def __init__(self, model, opt, loss_data, K, training_loader, validation_loader):
        self.model = model
        self.opt = opt
        self.loss_data = loss_data
        self.K = K
        self.training_loader = training_loader
        self.validation_loader = validation_loader
    
    
    def train_one_epoch(self):
        data_running_loss = 0.
        kl_running_loss = 0.

        n = len(self.training_loader.dataset)
        n_batches = len(self.training_loader)
        
        for i, data in enumerate(self.training_loader):

            x_batch, y_batch = data
            self.opt.zero_grad()
            
            y_preds = self.model(x_batch)
            y_preds_mean = y_preds.mean(axis=0)

            loss_data_ = self.loss_data(y_preds_mean, y_batch)
            kl_loss_ = self.K*self.model.kl_divergence_NN()/n_batches

            elbo_loss = loss_data_ + kl_loss_
            elbo_loss.backward()

            self.opt.step()

            data_running_loss += loss_data_.item()
            kl_running_loss += kl_loss_.item()

        loss_data = data_running_loss/n_batches
        kl = kl_running_loss

        return loss_data, kl
    
    
    def train(self, EPOCHS=150):
        epoch_number = 0

        for epoch in range(EPOCHS):
            print('------------------EPOCH {}------------------'.format(
                epoch_number + 1))

            self.model.train(True)
            avg_loss_data_loss, avg_kl_loss = self.train_one_epoch()
            avg_loss = avg_loss_data_loss + avg_kl_loss

            self.model.train(False)
            loss_data_running_loss_v = 0.0

            n = len(self.validation_loader.dataset)
            n_batches = len(self.validation_loader)
            
            kl_loss_ = self.K*self.model.kl_divergence_NN()

            for i, vdata in enumerate(self.validation_loader):

                x_val_batch, y_val_batch = vdata

                y_val_preds = self.model(x_val_batch)
                y_val_preds_mean = y_val_preds.mean(axis=0)

                loss_data_ = self.loss_data(y_val_preds_mean, y_val_batch)
                loss_data_running_loss_v += loss_data_

            avg_vloss_data = (loss_data_running_loss_v/n_batches).item()
            avg_vklloss = (kl_loss_).item()

            avg_vloss = avg_vloss_data + avg_vklloss

            print('DATA LOSS \t train {} valid {}'.format(
                round(avg_loss_data_loss, 2), round(avg_vloss_data, 2)))
            print('KL LOSS \t train {} valid {}'.format(
                round(avg_kl_loss, 2), round(avg_vklloss, 2)))
            print('ELBO LOSS \t train {} valid {}'.format(
                round(avg_loss, 2), round(avg_vloss, 2)))

            epoch_number += 1